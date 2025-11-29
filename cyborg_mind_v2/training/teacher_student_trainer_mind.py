"""
Teacher–Student Distillation for Cyborg Mind v2.0

This script trains a `BrainCyborgMind` using a mock teacher.  It
extends the teacher–student example found in the provided apex build
training loop【940122878883148†L20-L79】 by adding targets for the
emotion and workspace heads.  In production you would replace
`MockMindTeacher` with a model that produces appropriate actions,
value estimates, emotional targets and workspace targets for each
observation.  Training uses a weighted sum of cross‑entropy (for
actions) and mean squared error (for value, emotion and workspace).

Run this file to perform a simple distillation run.  It saves the
trained brain to ``models/student_brain_mind.pt``.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind


class MockMindTeacher:
    """
    A placeholder teacher for the Cyborg Mind.  It returns random
    action labels, value estimates, emotion targets and workspace
    targets.  Replace this with a real teacher—e.g., a language
    model that produces emotional annotations or a pre‑trained agent
    trained on human demonstrations.
    """

    def __init__(self, num_actions: int, emotion_dim: int, workspace_dim: int):
        self.num_actions = num_actions
        self.emotion_dim = emotion_dim
        self.workspace_dim = workspace_dim

    def predict(self, pixels: torch.Tensor, scalars: torch.Tensor, goals: torch.Tensor):
        B = pixels.size(0)
        # Random actions in the range [0, num_actions)
        target_action = torch.randint(0, self.num_actions, (B,))
        # Random value targets
        target_value = torch.randn(B, 1)
        # Random emotion and workspace targets in [-1,1]
        target_emotion = torch.randn(B, self.emotion_dim).tanh()
        target_workspace = torch.randn(B, self.workspace_dim).tanh()
        return target_action, target_value, target_emotion, target_workspace


def train_student_mind(num_steps: int = 500, lr: float = 1e-4, weight_decay: float = 1e-5):
    """
    Train a ``BrainCyborgMind`` using a mock teacher for a given number of steps.

    This training loop monitors the memory utilisation of the student's PMM and
    automatically expands the memory when pressure exceeds 85%.  When an
    expansion occurs the optimiser is reinitialised so that the new key/value
    parameters are trainable【531940505518075†L293-L344】.  Without this step
    the optimiser would continue to update stale parameter references leading
    to frozen memory slots during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = BrainCyborgMind().to(device)
    student.train()
    teacher = MockMindTeacher(
        num_actions=student.num_actions,
        emotion_dim=student.emotion_dim,
        workspace_dim=student.workspace_dim,
    )

    # Initialise optimiser and loss functions
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_action = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    criterion_emotion = nn.MSELoss()
    criterion_workspace = nn.MSELoss()

    print("=== STARTING MIND TEACHER-STUDENT DISTILLATION ===")
    for step in range(num_steps):
        # Generate a mock batch of observations
        B = 32
        pixels = torch.randn(B, 3, 128, 128).to(device)
        scalars = torch.randn(B, student.scalar_dim).to(device)
        goals = torch.randn(B, student.goal_dim).to(device)
        thoughts = torch.zeros(B, student.thought_dim).to(device)
        emotions = torch.zeros(B, student.emotion_dim).to(device)
        workspaces = torch.zeros(B, student.workspace_dim).to(device)
        # Get teacher targets
        with torch.no_grad():
            target_action, target_value, target_emotion, target_workspace = teacher.predict(
                pixels, scalars, goals
            )
            ta = target_action.to(device)
            tv = target_value.to(device)
            te = target_emotion.to(device)
            tw = target_workspace.to(device)

        # Forward pass
        out = student(
            pixels,
            scalars,
            goals,
            thoughts,
            emotion=emotions,
            workspace=workspaces,
        )
        # Compute weighted losses
        loss_act = criterion_action(out.action_logits, ta)
        loss_val = criterion_value(out.value, tv)
        loss_emo = criterion_emotion(out.emotion, te)
        loss_ws = criterion_workspace(out.workspace, tw)
        total_loss = loss_act + 0.5 * loss_val + 0.3 * loss_emo + 0.3 * loss_ws

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # Monitor memory pressure and expand if necessary
        pressure = student.pmm.get_pressure()
        if pressure > 0.85:
            expanded = student.pmm.expand()
            if expanded:
                # Reinitialise the optimiser so it sees the new key/value parameters
                optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
                print(f"[Trainer] Memory expanded to {student.pmm.mem_slots} slots at step {step}.")

        # Periodic logging
        if step % 50 == 0:
            print(
                f"Step {step}: Total {total_loss.item():.4f} | Act {loss_act.item():.3f} | "
                f"Val {loss_val.item():.3f} | Emo {loss_emo.item():.3f} | WS {loss_ws.item():.3f}"
            )

    # Save the trained weights
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(models_dir, "student_brain_mind.pt"))
    print("Mind Distillation Complete.")


if __name__ == "__main__":
    import os
    train_student_mind()