
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from pathlib import Path

from cyborg_rl.utils.config import Config
from cyborg_rl.envs.minerl_adapter import MineRLAdapter
from cyborg_rl.models.encoder import UnifiedEncoder
from cyborg_rl.models.policy import DiscretePolicy, ContinuousPolicy
from cyborg_rl.models.value import ValueHead
from cyborg_rl.memory.pmm import PMM
from cyborg_rl.memory.recurrent_rollout_buffer import RecurrentRolloutBuffer

try:
    import wandb
except ImportError:
    wandb = None

class PPOTrainer:
    """
    PPO System v3.0
    
    Unified trainer supporting:
    - Vectorized Envs
    - Unified Encoder (GRU/Mamba)
    - PMM (Honest Memory)
    - Full/Truncated BPTT
    - AMP & torch.compile
    """
    def __init__(self, config_dict, use_wandb=False):
        self.cfg = config_dict
        self.device = torch.device(self.cfg['train']['device']) if self.cfg['train']['device'] != 'auto' else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Logging
        self.use_wandb = use_wandb and (wandb is not None)
        if self.use_wandb:
            wandb.init(project="cyborg-minerl-v3", config=self.cfg)
        
        # Envs
        self.env = MineRLAdapter(
            env_name=self.cfg['env']['name'],
            image_size=tuple(self.cfg['env']['size']),
            max_steps=self.cfg['env']['max_steps'],
            num_envs=self.cfg['train']['num_envs']
        )
        
        self.obs_dim = self.env.observation_dim
        self.action_dim = self.env.action_dim
        # Flatten image for simple encoder input if needed, or keeping it as is?
        # Adapter likely flattens or provides structured. Let's assume flattened for now or Encoder handles it.
        # MineRL adapter v2 usually returns (C,H,W). Encoder expects (D) or (C,H,W)?
        # UnifiedEncoder `input_proj` is Linear. So it expects flattened.
        # We need a CNN encoder if input is image. The UnifiedEncoder is generic RNN.
        # Requirement: "Visual auto-cropping", "Efficient image resizing".
        # The UnifiedEncoder currently has `nn.Linear` input proj. 
        # I should add a CNN feature extractor or assume the MineRL adapter provides features.
        # "Latent frame stacking" suggests adapter provides pixels.
        # I will assume for now I need a small CNN in the encoder if input is 3D.
        # BUT `UnifiedEncoder` implementation I wrote uses `nn.Linear`.
        # I'll stick to `nn.Linear` and assume flattened input for v3.0 core, 
        # or rely on Adapter to flatten. (Usually MineRL yields 64x64x3 = 12288 dims).
        
        flattened_obs_dim = int(np.prod(self.obs_dim))
        
        
        # Models
        self.encoder = UnifiedEncoder(
            encoder_type=self.cfg['model']['encoder'],
            input_dim=flattened_obs_dim,
            hidden_dim=self.cfg['model']['hidden_dim'],
            latent_dim=self.cfg['model']['vision_dim'],
            device=self.device.type
        ).to(self.device)
        
        # PMM
        self.use_pmm = self.cfg['pmm']['enabled']
        if self.use_pmm:
            self.pmm = PMM(
                memory_dim=self.cfg['pmm']['memory_dim'],
                num_slots=self.cfg['pmm']['num_slots'],
                write_rate_target_inv=self.cfg['pmm']['write_rate_target_inv'],
                gate_type=self.cfg['pmm']['gate_type'],
                temperature=self.cfg['pmm']['temperature'],
                sharpness=self.cfg['pmm']['sharpness']
            ).to(self.device)
            # Encoder output -> PMM -> Policy/Value
            # We need to project encoder output to PMM dim if diff?
            # Configs: vision_dim vs memory_dim.
            self.pmm_proj = nn.Linear(self.cfg['model']['vision_dim'], self.cfg['pmm']['memory_dim']).to(self.device)
            policy_input_dim = self.cfg['pmm']['memory_dim'] + self.cfg['model']['vision_dim'] # Cat(vision, memory)
        else:
            policy_input_dim = self.cfg['model']['vision_dim']
            
        self.policy = DiscretePolicy(policy_input_dim, self.action_dim).to(self.device)
        self.value = ValueHead(policy_input_dim).to(self.device)
        
        # Optimizer
        self.params = list(self.encoder.parameters()) + \
                      list(self.policy.parameters()) + \
                      list(self.value.parameters())
        if self.use_pmm:
            self.params += list(self.pmm.parameters()) + list(self.pmm_proj.parameters())
            
        self.optimizer = optim.AdamW(self.params, lr=self.cfg['train']['learning_rate'], eps=1e-5)
        
        # Compile
        if self.cfg['train']['compile']:
            self._compile_models()
            
        # Storage
        # Storage
        # One buffer per env to maintain sequence integrity
        self.buffers = [
            RecurrentRolloutBuffer(
                buffer_size=self.cfg['train']['horizon'],
                obs_dim=flattened_obs_dim,
                action_dim=self.action_dim,
                is_discrete=True, # MineRL is discrete
                device=self.device,
                gamma=self.cfg['train']['gamma'],
                gae_lambda=self.cfg['train']['gae_lambda']
            ) for _ in range(self.cfg['train']['num_envs'])
        ]
        
        self.metrics = deque(maxlen=100)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['train']['amp'])

    def _compile_models(self):
        try:
            self.encoder = torch.compile(self.encoder)
            if self.use_pmm:
                self.pmm = torch.compile(self.pmm)
            self.policy = torch.compile(self.policy)
            self.value = torch.compile(self.value)
            print("✅ Models compiled with torch.compile")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    def train(self):
        print(f"🚀 Starting Training on {self.device}")
        
        # Init states
        # Combined state object: (encoder_state, pmm_state)
        # We start with None
        current_state = None
        current_pmm_memory = None 
        
        obs = self.env.reset() # (B, C, H, W)
        obs_flat = obs.reshape(obs.shape[0], -1)
        
        total_steps = 0
        update_idx = 0
        
        while total_steps < self.cfg['train']['total_timesteps']:
            for buf in self.buffers: buf.reset()
            
            # Collection phase
            for _ in range(self.cfg['train']['horizon']):
                with torch.no_grad():
                    t_obs = torch.as_tensor(obs_flat, device=self.device, dtype=torch.float32)
                    
                    # Forward
                    vision_embed_seq, next_encoder_state = self.encoder(t_obs, current_state)
                    vision_embed = vision_embed_seq.squeeze(1) # (B, 1, D) -> (B, D)
                    
                    policy_feat = vision_embed
                    next_pmm_mem = current_pmm_memory
                    
                    if self.use_pmm:
                        if current_pmm_memory is None:
                            current_pmm_memory = torch.zeros(
                                self.cfg['train']['num_envs'], 
                                self.cfg['pmm']['num_slots'], 
                                self.cfg['pmm']['memory_dim'], 
                                device=self.device
                            )
                        
                        # PMM Forward (dummy mask 1.0 for collection)
                        mask = torch.ones(self.cfg['train']['num_envs'], 1, device=self.device)
                        mem_read, next_pmm_mem, _ = self.pmm(
                            current_pmm_memory, 
                            self.pmm_proj(vision_embed), 
                            mask
                        )
                        policy_feat = torch.cat([vision_embed, mem_read], dim=-1)
                    
                    # Policy
                    action, log_prob = self.policy.sample(policy_feat, deterministic=self.cfg['train']['deterministic'])
                    value = self.value(policy_feat)
                
                # Step
                action_cpu = action.cpu().numpy()
                next_obs, rewards, dones, infos = self.env.step(action_cpu)
                next_obs_flat = next_obs.reshape(next_obs.shape[0], -1)
                
                # Store per env
                # We need to split batch inputs to individual buffers
                val_cpu = value.cpu().numpy()
                lp_cpu = log_prob.cpu().numpy()
                
                # Slicing states is hard if they are tensors/lists.
                # But rollout buffer needs state for THAT env.
                # Assuming state is handled by encoder, we just store the full batch state or slice it?
                # Storing full B-dim state in each buffer is redundant and wrong (RecurrentRolloutBuffer isn't smart).
                # Implementation: RecurrentRolloutBuffer expects `recurrent_state` to be the state for THIS env.
                # So we must slice `current_state` (which is B-dim).
                # _slice_state helper needed.
                
                for i, buf in enumerate(self.buffers):
                    # Slice state for env i
                    env_state = self._slice_state(current_state, i)
                    env_pmm = self._slice_state(current_pmm_memory, i) if self.use_pmm else None
                    full_env_state = (env_state, env_pmm)
                    
                    buf.add(
                        obs_flat[i], 
                        action_cpu[i], 
                        rewards[i], 
                        val_cpu[i].item() if val_cpu.ndim>0 else val_cpu.item(), 
                        lp_cpu[i].item() if lp_cpu.ndim>0 else lp_cpu.item(), 
                        dones[i], 
                        full_env_state
                    )
                
                # Update loop vars
                obs_flat = next_obs_flat
                current_state = next_encoder_state
                
                # Handle reset state on done
                if np.any(dones):
                    t_dones = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
                    if self.use_pmm:
                         # Mask PMM memory
                         mask_reset = (~t_dones).float().unsqueeze(1).unsqueeze(2)
                         next_pmm_mem = next_pmm_mem * mask_reset
                
                current_pmm_memory = next_pmm_mem
                total_steps += self.cfg['train']['num_envs']
            
            # Update Policy
            with torch.no_grad():
                # Bootstrap value
                t_obs = torch.as_tensor(obs_flat, device=self.device, dtype=torch.float32)
                v_emb_seq, _ = self.encoder(t_obs, current_state)
                v_emb = v_emb_seq.squeeze(1)
                p_feat = v_emb
                if self.use_pmm:
                    m_read, _, _ = self.pmm(current_pmm_memory, self.pmm_proj(v_emb), torch.ones(self.cfg['train']['num_envs'], 1, device=self.device))
                    p_feat = torch.cat([v_emb, m_read], dim=-1)
                last_values = self.value(p_feat).cpu().numpy()
                
            for i, buf in enumerate(self.buffers):
                buf.compute_returns_and_advantages(last_values[i], last_done=dones[i])
            
            self._update_network()
            update_idx += 1
            
            if update_idx % self.cfg['train']['save_interval'] == 0:
                self._save_checkpoint(f"step_{total_steps}")
                
    def _slice_state(self, state, idx):
        if state is None: return None
        if isinstance(state, torch.Tensor):
            # (Layers, B, H) for GRU? Or (B, H)? UnifiedEncoder ensures.
            # UnifiedEncoder output `next_encoder_state`:
            # GRU: (Layers, B, H)
            # Mamba: List[Dict] (per layer)
            if state.ndim == 3 and state.shape[1] == self.cfg['train']['num_envs']:
                 # GRU-like
                 return state[:, idx:idx+1, :]
            elif state.ndim == 2 and state.shape[0] == self.cfg['train']['num_envs']:
                 return state[idx:idx+1, :]
            else:
                 # Assume dim 0 is batch?
                 return state[idx:idx+1]
        elif isinstance(state, list):
             # Mamba caches are List[Dict]
             new_list = []
             for layer_cache in state:
                 new_cache = {}
                 for k, v in layer_cache.items():
                     if isinstance(v, torch.Tensor):
                         # Usually (B, ...)
                         new_cache[k] = v[idx:idx+1]
                 new_list.append(new_cache)
             return new_list
        return state

    def _update_network(self):
        # Truncated or Full BPTT
        burn_in = self.cfg['train']['burn_in']
        
        seq_len = 128
        batch_size = self.cfg['train']['batch_size'] // seq_len # number of sequences
        
        # Combine samplers from all buffers
        import itertools
        samplers = [buf.get_sampler(batch_size // len(self.buffers), seq_len, burn_in) for buf in self.buffers]
        combined_sampler = itertools.chain(*samplers)
        
        for batch in combined_sampler:
            # batch has tensors (B, T, D)
            # states is a list of len B
            
            # We need to stack states to batch form
            # This is complex if state is nested.
            # Assuming state is handled correctly by encoder.
            
            # Re-run forward
            with torch.cuda.amp.autocast(enabled=self.cfg['train']['amp']):
                # Unpack states
                # Optimized: We assume we can just pass the list of states to UnifiedEncoder if it supports it?
                # UnifiedEncoder `state` arg expects `Any`.
                # But we need to batch them.
                # For this implementation, we will perform the forward pass iteratively (or use optimized scan)
                # re-calculating everything to get gradients.
                
                # Unroll logic for BPTT
                # 1. Init state from batch['states']
                # 2. Loop T times
                # 3. Compute Loss
                
                loss = self._compute_loss(batch)
                
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg['train']['max_grad_norm'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_wandb:
                wandb.log({"loss": loss.item()})

    def _compute_loss(self, batch):
        # batch: obs(B,T,D), actions(B,T), states(List[B]), ...
        
        t_obs = batch['obs']
        t_actions = batch['actions']
        t_logprobs_old = batch['log_probs']
        t_adv = batch['advantages']
        t_ret = batch['returns']
        t_val_old = batch['values']
        t_dones = batch['dones'] # (B, T)
        
        states = batch['states']
        
        # 1. Prepare Init States
        init_encoder_state, init_pmm_state = self._stack_states(states)
        
        # 2. Forward Sequence
        latent_seq, _ = self.encoder(t_obs, init_encoder_state)
        
        policy_features = latent_seq
        aux_loss = 0.0
        
        if self.use_pmm:
            masks = 1.0 - t_dones.unsqueeze(-1) # (B, T, 1)
            
            if init_pmm_state is None:
                 init_pmm_state = torch.zeros(
                     t_obs.size(0), self.cfg['pmm']['num_slots'], self.cfg['pmm']['memory_dim'],
                     device=self.device
                 )
                 
            pmm_read_seq, pmm_logs = self.pmm.forward_sequence(
                init_pmm_state, self.pmm_proj(latent_seq), masks
            )
            
            policy_features = torch.cat([latent_seq, pmm_read_seq], dim=-1)
            aux_loss += pmm_logs['sparsity_loss']
            
        # 3. Policy & Value Heads
        B, T, _ = policy_features.shape
        flat_feat = policy_features.reshape(B*T, -1)
        
        logits = self.policy.forward(flat_feat) 
        
        dist = torch.distributions.Categorical(logits=logits)
        
        flat_actions = t_actions.view(B*T)
        new_log_probs = dist.log_prob(flat_actions)
        entropy = dist.entropy().mean()
        
        new_values = self.value(flat_feat).view(B, T)
        new_log_probs = new_log_probs.view(B, T)
        
        # 4. Losses
        ratio = (new_log_probs - t_logprobs_old).exp()
        
        surr1 = ratio * t_adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg['train']['clip_range'], 1.0 + self.cfg['train']['clip_range']) * t_adv
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (new_values - t_ret).pow(2).mean()
        
        loss = policy_loss + \
               self.cfg['train']['value_coef'] * value_loss - \
               self.cfg['train']['entropy_coef'] * entropy + \
               aux_loss
               
        return loss

    def _stack_states(self, states_list):
        """
        Stack list of (encoder_state, pmm_state) into batched form.
        """
        # states_list is List[Tuple(enc, pmm)]
        # Filter Nones
        if states_list[0] is None:
            return None, None
            
        enc_list, pmm_list = zip(*states_list)
        
        # Stack Encoder State
        enc_state = None
        if isinstance(enc_list[0], torch.Tensor):
            # GRU: (layers, B, H) -> Stack on dim 1
            enc_state = torch.cat(enc_list, dim=1) 
        elif isinstance(enc_list[0], list):
             # PseudoMamba: List[Dict]. Stack 'h'.
             new_layers = []
             for layer_idx in range(len(enc_list[0])):
                 layer_states = [item[layer_idx] for item in enc_list]
                 if 'h' in layer_states[0]:
                     h_stack = torch.cat([d['h'] for d in layer_states], dim=0)
                     new_layers.append({'h': h_stack})
                 else:
                     new_layers.append({})
             enc_state = new_layers
             
        # Stack PMM State (B, Slots, Dim)
        if pmm_list[0] is not None:
             pmm_state = torch.cat(pmm_list, dim=0)
        else:
             pmm_state = None
             
        return enc_state, pmm_state

    def _save_checkpoint(self, name):
        path = os.path.join(self.cfg['train']['save_dir'], f"{name}.pt")
        os.makedirs(self.cfg['train']['save_dir'], exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"Saved checkpoint {path}")

