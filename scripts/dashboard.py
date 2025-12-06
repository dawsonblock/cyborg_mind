"""
Interactive Dashboard for CyborgMind.
Run with: streamlit run scripts/dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import numpy as np

st.set_page_config(
    page_title="CyborgMind Control",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 CyborgMind Neural Interface")

# --- Sidebar ---
with st.sidebar:
    st.header("Connection Settings")
    api_url = st.text_input("API URL", "http://localhost:8000")
    token = st.text_input("Auth Token", os.environ.get("CYBORG_AUTH_TOKEN", ""), type="password")
    
    if st.button("Check Connection"):
        try:
            resp = requests.get(f"{api_url}/health")
            if resp.status_code == 200:
                st.success("Connected!")
                st.json(resp.json())
            else:
                st.error(f"Error: {resp.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    st.divider()
    st.header("Control Mode")
    auto_refresh = st.checkbox("Auto-Refresh (Monitoring)", value=False)
    refresh_rate = st.slider("Refresh Rate (s)", 0.5, 5.0, 1.0)

# --- Main Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Sensory Input (CartPole)")
    
    with st.form("observation_form"):
        cart_pos = st.slider("Cart Position", -2.4, 2.4, 0.0)
        cart_vel = st.slider("Cart Velocity", -2.0, 2.0, 0.0)
        pole_angle = st.slider("Pole Angle", -0.2, 0.2, 0.0)
        pole_vel = st.slider("Pole Velocity", -2.0, 2.0, 0.0)
        
        agent_id = st.text_input("Agent ID", "streamlit_user")
        deterministic = st.checkbox("Deterministic Action", value=True)
        
        submitted = st.form_submit_button("Send Observation")

    if submitted or auto_refresh:
        payload = {
            "observation": [cart_pos, cart_vel, pole_angle, pole_vel],
            "agent_id": agent_id,
            "deterministic": deterministic
        }
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            # If auto-refreshing, we might want to randomize input slightly to see changes
            if auto_refresh and not submitted:
                payload["observation"] = [
                    cart_pos + np.random.normal(0, 0.1),
                    cart_vel,
                    pole_angle + np.random.normal(0, 0.05),
                    pole_vel
                ]

            resp = requests.post(f"{api_url}/step", json=payload, headers=headers)
            if resp.status_code == 200:
                st.session_state['last_response'] = resp.json()
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            if submitted: # Only show error on manual submit to avoid spam
                st.error(f"Request failed: {e}")

with col2:
    st.subheader("Brain State")
    
    if 'last_response' in st.session_state:
        data = st.session_state['last_response']
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Action", data['action'])
        m2.metric("Value Estimate", f"{data['value']:.3f}")
        m3.metric("Memory Pressure", f"{data['pressure']:.3f}")
        
        # Thought Stream
        st.info(f"💭 **Internal Monologue:** {data['thought']}")
        
        # Visualizations
        tab1, tab2 = st.tabs(["Memory State", "Decision Confidence"])
        
        with tab1:
            st.caption("PMM Attention Map (Simulated Visualization)")
            # Mockup: In production, API should return attention weights
            # Generating a heatmap that reacts to pressure
            pressure = data['pressure']
            memory_state = np.random.rand(4, 16) * (1.0 + pressure)
            
            fig = px.imshow(
                memory_state,
                labels=dict(x="Memory Slot", y="Read Head", color="Activation"),
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.caption("Value Estimation History")
            if 'value_history' not in st.session_state:
                st.session_state['value_history'] = []
            
            st.session_state['value_history'].append(data['value'])
            if len(st.session_state['value_history']) > 50:
                st.session_state['value_history'].pop(0)
                
            chart_data = pd.DataFrame({
                "Step": range(len(st.session_state['value_history'])),
                "Value": st.session_state['value_history']
            })
            
            st.line_chart(chart_data, x="Step", y="Value", height=300)

    else:
        st.info("Waiting for input... Send an observation to activate the brain.")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
