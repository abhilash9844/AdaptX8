"""
AdaptX - Resource Optimization & Efficiency Prediction
=======================================================
Day/Night Theme System + Fixed Analysis Background
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import time
from datetime import datetime

from utils.simulation import InfrastructureSimulator
from utils.optimizer import AIOptimizer
from utils.server_manager import ServerManager
from utils.alarm import AlarmSystem

st.set_page_config(
    page_title="AdaptX - Resource Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def get_theme_css(is_dark_mode):
    """Generate CSS based on theme."""
    
    if is_dark_mode:
        # DARK THEME (Night Mode)
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
            /* ROOT VARIABLES - DARK THEME */
            :root {
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
                --bg-tertiary: #334155;
                --bg-card: #1e293b;
                --bg-hover: #334155;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --border-color: #334155;
                --border-light: #475569;
                --shadow-color: rgba(0, 0, 0, 0.3);
                --accent-ai: #818cf8;
                --accent-human: #34d399;
                --success: #22c55e;
                --warning: #f59e0b;
                --danger: #ef4444;
                --info: #3b82f6;
            }
            
            * { font-family: 'Inter', sans-serif; }
            
            .stApp {
                background-color: var(--bg-primary) !important;
            }
            
            .main-header {
                background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #1e293b 100%);
                padding: 1.5rem 2rem;
                border-radius: 16px;
                margin-bottom: 1.5rem;
                text-align: center;
                box-shadow: 0 10px 40px var(--shadow-color);
                border: 1px solid var(--border-color);
            }
            
            .main-title {
                font-size: 2rem;
                font-weight: 800;
                color: var(--text-primary);
                margin: 0;
                letter-spacing: -0.5px;
            }
            
            .main-subtitle {
                font-size: 0.85rem;
                color: var(--text-secondary);
                margin-top: 0.5rem;
            }
            
            .team-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                padding: 0.25rem 0.8rem;
                border-radius: 20px;
                font-size: 0.7rem;
                font-weight: 600;
                margin-top: 0.5rem;
            }
            
            .theme-indicator {
                display: inline-block;
                margin-left: 0.5rem;
                font-size: 0.7rem;
            }
            
            .scoreboard-container {
                background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
                border-radius: 16px;
                padding: 1.2rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px var(--shadow-color);
                display: flex;
                justify-content: space-around;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
                border: 1px solid var(--border-color);
            }
            
            .score-card {
                text-align: center;
                padding: 1rem 2rem;
                border-radius: 12px;
                color: white;
                min-width: 160px;
            }
            
            .ai-bg { 
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); 
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35);
            }
            
            .human-bg { 
                background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35);
            }
            
            .score-label { font-size: 0.8rem; opacity: 0.9; }
            .score-val { font-size: 2rem; font-weight: 800; }
            .score-trend { font-size: 0.75rem; opacity: 0.85; margin-top: 0.3rem; }
            
            .gap-card {
                text-align: center;
                padding: 0.8rem 1.5rem;
                background: var(--bg-tertiary);
                border-radius: 12px;
                box-shadow: 0 2px 10px var(--shadow-color);
                border: 1px solid var(--border-color);
            }
            
            .gap-round { font-size: 0.85rem; color: var(--text-secondary); }
            .gap-value { font-size: 1.8rem; font-weight: 800; color: var(--text-primary); }
            .gap-label { font-size: 0.8rem; margin-top: 0.3rem; font-weight: 600; }
            
            .section-card {
                background: var(--bg-card);
                border-radius: 14px;
                padding: 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 15px var(--shadow-color);
                border: 1px solid var(--border-color);
            }
            
            .ai-border { border-left: 5px solid #818cf8; }
            .human-border { border-left: 5px solid #34d399; }
            
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 1rem;
                padding-bottom: 0.6rem;
                border-bottom: 2px solid var(--border-color);
            }
            
            .eff-container {
                display: flex;
                gap: 0.8rem;
                margin-bottom: 1rem;
            }
            
            .eff-card {
                flex: 1;
                text-align: center;
                padding: 0.8rem;
                border-radius: 10px;
                background: var(--bg-tertiary);
            }
            
            .eff-curr { 
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%); 
                border: 2px solid #10b981; 
            }
            
            .eff-pred { 
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.2) 100%); 
                border: 2px solid #6366f1; 
            }
            
            .eff-val { font-size: 1.6rem; font-weight: 800; color: var(--text-primary); }
            .eff-lbl { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
            
            .metrics-container {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .metric-item {
                background: var(--bg-tertiary);
                border-radius: 8px;
                padding: 0.5rem;
                text-align: center;
                border: 1px solid var(--border-color);
            }
            
            .metric-val { font-size: 1rem; font-weight: 700; color: var(--text-primary); }
            .metric-lbl { font-size: 0.6rem; color: var(--text-secondary); text-transform: uppercase; }
            
            .server-grid-container {
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 0.5rem;
            }
            
            .server-item {
                background: var(--bg-tertiary);
                border-radius: 10px;
                padding: 0.6rem;
                border: 1px solid var(--border-color);
                text-align: center;
                min-height: 80px;
            }
            
            .server-on { border-left: 4px solid #22c55e; }
            .server-off { border-left: 4px solid #64748b; background: var(--bg-secondary); }
            .server-hot { border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.1); }
            .server-crit { border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.1); }
            
            .server-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; color: var(--text-primary); }
            .server-stats { font-size: 0.7rem; color: var(--text-secondary); line-height: 1.4; }
            
            .action-log-container {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 10px;
                padding: 0.8rem;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 0.75rem;
                color: #94a3b8;
                max-height: 150px;
                overflow-y: auto;
                border: 1px solid var(--border-color);
            }
            
            /* FIXED ANALYSIS CARD - DARK THEME */
            .analysis-card {
                background: var(--bg-tertiary);
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid var(--border-color);
            }
            
            .analysis-title {
                font-weight: 700;
                font-size: 0.9rem;
                color: var(--text-primary);
                margin-bottom: 0.8rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-color);
            }
            
            .analysis-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--border-color);
            }
            
            .analysis-item:last-of-type { border-bottom: none; }
            
            .analysis-label { font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }
            .analysis-value { font-size: 0.85rem; font-weight: 600; }
            
            .suggestions-container {
                margin-top: 0.8rem;
                padding-top: 0.8rem;
                border-top: 1px solid var(--border-color);
            }
            
            .suggestions-title {
                font-weight: 600;
                font-size: 0.8rem;
                color: var(--text-primary);
                margin-bottom: 0.5rem;
            }
            
            .suggestion-item {
                background: var(--bg-secondary);
                border-radius: 6px;
                padding: 0.5rem 0.8rem;
                margin-bottom: 0.4rem;
                font-size: 0.8rem;
                color: var(--text-secondary);
                border-left: 3px solid #818cf8;
            }
            
            .tips-container {
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(234, 179, 8, 0.15) 100%);
                border-radius: 10px;
                padding: 1rem;
                border-left: 4px solid #f59e0b;
            }
            
            .tips-title { font-weight: 700; color: #fbbf24; margin-bottom: 0.5rem; font-size: 0.9rem; }
            .tips-item { font-size: 0.8rem; color: #fcd34d; margin: 0.3rem 0; line-height: 1.4; }
            
            /* Streamlit overrides for dark theme */
            .stExpander {
                background-color: var(--bg-secondary) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 10px !important;
            }
            
            .stExpander > div > div {
                background-color: var(--bg-secondary) !important;
            }
            
            .stMarkdown, .stText, p, span, label {
                color: var(--text-primary) !important;
            }
            
            .stSlider > div > div {
                background-color: var(--bg-tertiary) !important;
            }
            
            .stCheckbox > label {
                color: var(--text-primary) !important;
            }
            
            .stButton > button {
                background-color: var(--bg-tertiary) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
            }
            
            .stButton > button:hover {
                background-color: var(--bg-hover) !important;
                border-color: var(--accent-ai) !important;
            }
            
            .stMetric {
                background-color: var(--bg-secondary) !important;
                padding: 0.5rem !important;
                border-radius: 8px !important;
            }
            
            .stMetric label {
                color: var(--text-secondary) !important;
            }
            
            .stMetric > div {
                color: var(--text-primary) !important;
            }
            
            /* Sidebar dark */
            section[data-testid="stSidebar"] {
                background-color: var(--bg-secondary) !important;
            }
            
            section[data-testid="stSidebar"] .stMarkdown {
                color: var(--text-primary) !important;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
        </style>
        """
    else:
        # LIGHT THEME (Day Mode)
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
            /* ROOT VARIABLES - LIGHT THEME */
            :root {
                --bg-primary: #f8fafc;
                --bg-secondary: #ffffff;
                --bg-tertiary: #f1f5f9;
                --bg-card: #ffffff;
                --bg-hover: #e2e8f0;
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --text-muted: #94a3b8;
                --border-color: #e2e8f0;
                --border-light: #cbd5e1;
                --shadow-color: rgba(0, 0, 0, 0.08);
                --accent-ai: #6366f1;
                --accent-human: #10b981;
                --success: #22c55e;
                --warning: #f59e0b;
                --danger: #ef4444;
                --info: #3b82f6;
            }
            
            * { font-family: 'Inter', sans-serif; }
            
            .stApp {
                background-color: var(--bg-primary) !important;
            }
            
            .main-header {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #1e3a5f 100%);
                padding: 1.5rem 2rem;
                border-radius: 16px;
                margin-bottom: 1.5rem;
                text-align: center;
                box-shadow: 0 10px 40px rgba(30, 58, 95, 0.3);
            }
            
            .main-title {
                font-size: 2rem;
                font-weight: 800;
                color: white;
                margin: 0;
                letter-spacing: -0.5px;
            }
            
            .main-subtitle {
                font-size: 0.85rem;
                color: rgba(255, 255, 255, 0.8);
                margin-top: 0.5rem;
            }
            
            .team-badge {
                display: inline-block;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                padding: 0.25rem 0.8rem;
                border-radius: 20px;
                font-size: 0.7rem;
                font-weight: 600;
                margin-top: 0.5rem;
            }
            
            .theme-indicator {
                display: inline-block;
                margin-left: 0.5rem;
                font-size: 0.7rem;
            }
            
            .scoreboard-container {
                background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
                border-radius: 16px;
                padding: 1.2rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px var(--shadow-color);
                display: flex;
                justify-content: space-around;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
                border: 1px solid var(--border-color);
            }
            
            .score-card {
                text-align: center;
                padding: 1rem 2rem;
                border-radius: 12px;
                color: white;
                min-width: 160px;
            }
            
            .ai-bg { 
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); 
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35);
            }
            
            .human-bg { 
                background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35);
            }
            
            .score-label { font-size: 0.8rem; opacity: 0.9; }
            .score-val { font-size: 2rem; font-weight: 800; }
            .score-trend { font-size: 0.75rem; opacity: 0.85; margin-top: 0.3rem; }
            
            .gap-card {
                text-align: center;
                padding: 0.8rem 1.5rem;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 10px var(--shadow-color);
                border: 1px solid var(--border-color);
            }
            
            .gap-round { font-size: 0.85rem; color: var(--text-secondary); }
            .gap-value { font-size: 1.8rem; font-weight: 800; color: var(--text-primary); }
            .gap-label { font-size: 0.8rem; margin-top: 0.3rem; font-weight: 600; }
            
            .section-card {
                background: var(--bg-card);
                border-radius: 14px;
                padding: 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 15px var(--shadow-color);
                border: 1px solid var(--border-color);
            }
            
            .ai-border { border-left: 5px solid #6366f1; }
            .human-border { border-left: 5px solid #10b981; }
            
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 1rem;
                padding-bottom: 0.6rem;
                border-bottom: 2px solid var(--border-color);
            }
            
            .eff-container {
                display: flex;
                gap: 0.8rem;
                margin-bottom: 1rem;
            }
            
            .eff-card {
                flex: 1;
                text-align: center;
                padding: 0.8rem;
                border-radius: 10px;
            }
            
            .eff-curr { 
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
                border: 2px solid #10b981; 
            }
            
            .eff-pred { 
                background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); 
                border: 2px solid #6366f1; 
            }
            
            .eff-val { font-size: 1.6rem; font-weight: 800; }
            .eff-lbl { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
            
            .metrics-container {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .metric-item {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 8px;
                padding: 0.5rem;
                text-align: center;
                border: 1px solid var(--border-color);
            }
            
            .metric-val { font-size: 1rem; font-weight: 700; color: var(--text-primary); }
            .metric-lbl { font-size: 0.6rem; color: var(--text-secondary); text-transform: uppercase; }
            
            .server-grid-container {
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 0.5rem;
            }
            
            .server-item {
                background: #f8fafc;
                border-radius: 10px;
                padding: 0.6rem;
                border: 1px solid var(--border-color);
                text-align: center;
                min-height: 80px;
            }
            
            .server-on { border-left: 4px solid #10b981; }
            .server-off { border-left: 4px solid #94a3b8; background: #f1f5f9; }
            .server-hot { border-left: 4px solid #f59e0b; background: #fffbeb; }
            .server-crit { border-left: 4px solid #ef4444; background: #fef2f2; }
            
            .server-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; color: var(--text-primary); }
            .server-stats { font-size: 0.7rem; color: var(--text-secondary); line-height: 1.4; }
            
            .action-log-container {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 10px;
                padding: 0.8rem;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 0.75rem;
                color: #94a3b8;
                max-height: 150px;
                overflow-y: auto;
            }
            
            /* FIXED ANALYSIS CARD - LIGHT THEME */
            .analysis-card {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid var(--border-color);
            }
            
            .analysis-title {
                font-weight: 700;
                font-size: 0.9rem;
                color: var(--text-primary);
                margin-bottom: 0.8rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-color);
            }
            
            .analysis-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid #e5e7eb;
            }
            
            .analysis-item:last-of-type { border-bottom: none; }
            
            .analysis-label { font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }
            .analysis-value { font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }
            
            .suggestions-container {
                margin-top: 0.8rem;
                padding-top: 0.8rem;
                border-top: 1px solid var(--border-color);
            }
            
            .suggestions-title {
                font-weight: 600;
                font-size: 0.8rem;
                color: var(--text-primary);
                margin-bottom: 0.5rem;
            }
            
            .suggestion-item {
                background: white;
                border-radius: 6px;
                padding: 0.5rem 0.8rem;
                margin-bottom: 0.4rem;
                font-size: 0.8rem;
                color: #475569;
                border-left: 3px solid #6366f1;
            }
            
            .tips-container {
                background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                border-radius: 10px;
                padding: 1rem;
                border-left: 4px solid #f59e0b;
            }
            
            .tips-title { font-weight: 700; color: #92400e; margin-bottom: 0.5rem; font-size: 0.9rem; }
            .tips-item { font-size: 0.8rem; color: #78350f; margin: 0.3rem 0; line-height: 1.4; }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
        </style>
        """


# ============================================================
# INITIALIZATION
# ============================================================

@st.cache_resource
def load_model():
    if os.path.exists("models/efficiency_model.pkl") and os.path.exists("models/scaler.pkl"):
        try:
            return joblib.load("models/efficiency_model.pkl"), joblib.load("models/scaler.pkl")
        except:
            return None, None
    return None, None


def init():
    if 'init' not in st.session_state:
        st.session_state.ai_sim = InfrastructureSimulator()
        st.session_state.human_sim = InfrastructureSimulator()
        st.session_state.ai_mgr = ServerManager()
        st.session_state.human_mgr = ServerManager()
        st.session_state.optimizer = AIOptimizer()
        st.session_state.alarm = AlarmSystem(50.0)
        
        st.session_state.ai_hist = {
            't': [], 'eff': [], 'workload': [], 'energy': [],
            'temps': [[] for _ in range(10)]
        }
        st.session_state.human_hist = {
            't': [], 'eff': [], 'workload': [], 'energy': [],
            'temps': [[] for _ in range(10)]
        }
        
        st.session_state.ai_scores = []
        st.session_state.human_scores = []
        st.session_state.ai_log = []
        st.session_state.rounds = 0
        st.session_state.dark_mode = True  # Default to dark mode
        st.session_state.init = True


def update_hist(h, s, max_len=80):
    h['t'].append(time.time())
    h['eff'].append(s['current_efficiency'])
    h['workload'].append(s['workload'])
    h['energy'].append(s['energy'])
    
    for i in range(10):
        h['temps'][i].append(s['server_temperatures'][i])
    
    for k in ['t', 'eff', 'workload', 'energy']:
        if len(h[k]) > max_len:
            h[k].pop(0)
    for i in range(10):
        if len(h['temps'][i]) > max_len:
            h['temps'][i].pop(0)


def predict(sim, model, scaler):
    if model is None:
        return sim.calculate_current_efficiency()
    return np.clip(model.predict(scaler.transform(sim.get_features_for_prediction()))[0], 20, 92)


def get_chart_template():
    """Get Plotly template based on theme."""
    if st.session_state.get('dark_mode', True):
        return "plotly_dark"
    return "plotly_white"


def get_chart_colors():
    """Get chart colors based on theme."""
    if st.session_state.get('dark_mode', True):
        return {
            'bg': '#1e293b',
            'grid': '#334155',
            'text': '#f1f5f9'
        }
    return {
        'bg': '#ffffff',
        'grid': '#e2e8f0',
        'text': '#1e293b'
    }


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    is_dark = st.session_state.get('dark_mode', True)
    theme_icon = "🌙" if is_dark else "☀️"
    
    html = f"""
    <div class="main-header">
        <div class="main-title">⚡ Resource Optimization & Efficiency Prediction</div>
        <div class="main-subtitle">ML-Driven Predictive Infrastructure Management</div>
        <div class="team-badge">🚀 Team AdaptX</div>
        <span class="theme-indicator">{theme_icon} {'Night Mode' if is_dark else 'Day Mode'}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_scoreboard():
    r = max(1, st.session_state.rounds)
    
    ai_scores = st.session_state.ai_scores
    human_scores = st.session_state.human_scores
    
    ai_avg = sum(ai_scores) / len(ai_scores) if ai_scores else 50
    human_avg = sum(human_scores) / len(human_scores) if human_scores else 50
    
    ai_recent = ai_scores[-15:] if len(ai_scores) >= 15 else ai_scores
    human_recent = human_scores[-15:] if len(human_scores) >= 15 else human_scores
    
    ai_trend = sum(ai_recent) / len(ai_recent) if ai_recent else 50
    human_trend = sum(human_recent) / len(human_recent) if human_recent else 50
    
    gap = ai_avg - human_avg
    
    if gap > 1.5:
        winner, color = "🤖 AI Leads", "#818cf8"
    elif gap < -1.5:
        winner, color = "👤 Human Leads", "#34d399"
    else:
        winner, color = "🤝 Tied", "#94a3b8"
    
    ai_arrow = "📈" if ai_trend > ai_avg else ("📉" if ai_trend < ai_avg - 0.5 else "➡️")
    human_arrow = "📈" if human_trend > human_avg else ("📉" if human_trend < human_avg - 0.5 else "➡️")
    
    html = f"""
    <div class="scoreboard-container">
        <div class="score-card ai-bg">
            <div class="score-label">🤖 AI (ML-Driven)</div>
            <div class="score-val">{ai_avg:.1f}%</div>
            <div class="score-trend">{ai_arrow} Recent: {ai_trend:.1f}%</div>
        </div>
        <div class="gap-card">
            <div class="gap-round">Round {r}</div>
            <div class="gap-value" style="color:{color};">{abs(gap):.1f}%</div>
            <div class="gap-label" style="color:{color};">{winner}</div>
        </div>
        <div class="score-card human-bg">
            <div class="score-label">👤 Human Control</div>
            <div class="score-val">{human_avg:.1f}%</div>
            <div class="score-trend">{human_arrow} Recent: {human_trend:.1f}%</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_efficiency_cards(curr, pred):
    c_col = "#22c55e" if curr >= 70 else ("#f59e0b" if curr >= 50 else "#ef4444")
    
    html = f"""
    <div class="eff-container">
        <div class="eff-card eff-curr">
            <div class="eff-lbl">Current Efficiency</div>
            <div class="eff-val" style="color:{c_col};">{curr:.1f}%</div>
        </div>
        <div class="eff-card eff-pred">
            <div class="eff-lbl">ML Predicted</div>
            <div class="eff-val" style="color:#818cf8;">{pred:.1f}%</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metrics(s):
    html = f"""
    <div class="metrics-container">
        <div class="metric-item"><div class="metric-val">{s['servers']}/10</div><div class="metric-lbl">Servers</div></div>
        <div class="metric-item"><div class="metric-val">{s['workload']:.0f}%</div><div class="metric-lbl">Workload</div></div>
        <div class="metric-item"><div class="metric-val">{s['cpu']:.0f}%</div><div class="metric-lbl">Avg CPU</div></div>
        <div class="metric-item"><div class="metric-val">{s['energy']:.0f}W</div><div class="metric-lbl">Energy</div></div>
        <div class="metric-item"><div class="metric-val">{s['temperature']:.1f}°</div><div class="metric-lbl">Avg Temp</div></div>
        <div class="metric-item"><div class="metric-val">{s['max_temperature']:.1f}°</div><div class="metric-lbl">Max Temp</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_server_grid(sim, prefix, editable=False, mgr=None):
    fan_names = {0: "OFF", 1: "LOW", 2: "MED", 3: "HIGH"}
    fan_symbols = {0: "⭘", 1: "▪", 2: "▪▪", 3: "▪▪▪"}
    
    for row in range(2):
        cols = st.columns(5)
        for col_idx in range(5):
            i = row * 5 + col_idx
            
            with cols[col_idx]:
                on = sim.server_states[i]
                temp = sim.server_temperatures[i]
                cpu = sim.server_cpu_loads[i]
                fan = sim.fan_states[i]
                
                if not on:
                    cls, icon = "server-off", "⚫"
                elif temp >= 55:
                    cls, icon = "server-crit", "🔴"
                elif temp >= 45:
                    cls, icon = "server-hot", "🟠"
                else:
                    cls, icon = "server-on", "🟢"
                
                html = f"""
                <div class="server-item {cls}">
                    <div class="server-name">{icon} S{i+1}</div>
                    <div class="server-stats">
                        🌡️ {temp:.1f}°C<br>
                        💻 {cpu:.0f}% | 🌀 {fan_symbols[fan]}
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                
                if editable and mgr is not None:
                    new_power = st.toggle("Power", value=on, key=f"{prefix}_pwr_{i}")
                    if new_power != on:
                        success, _ = mgr.set_server_online(i, new_power, "Human")
                        if success:
                            sim.set_server_state(i, new_power)
                            st.rerun()
                    
                    if sim.server_states[i]:
                        new_fan = st.select_slider(
                            "Fan", options=[0, 1, 2, 3], value=sim.fan_states[i],
                            format_func=lambda x: fan_names[x], key=f"{prefix}_fan_{i}"
                        )
                        if new_fan != sim.fan_states[i]:
                            mgr.set_fan_speed(i, new_fan, "Human")
                            sim.set_fan_state(i, new_fan)
                            st.rerun()


def render_workload_chart(h, prefix):
    if len(h['t']) < 3:
        st.info("📊 Collecting workload data...")
        return
    
    t0 = h['t'][0]
    times = [t - t0 for t in h['t']]
    colors = get_chart_colors()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=h['workload'], mode='lines', name='Workload',
        line=dict(color='#8b5cf6', width=2.5),
        fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.15)'
    ))
    
    fig.update_layout(
        height=200, margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False, xaxis_title="Time (s)", yaxis_title="Workload %",
        template=get_chart_template(), yaxis=dict(range=[0, 100]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_workload")


def render_energy_chart(h, prefix):
    if len(h['t']) < 3:
        st.info("⚡ Collecting energy data...")
        return
    
    t0 = h['t'][0]
    times = [t - t0 for t in h['t']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=h['energy'], mode='lines', name='Energy',
        line=dict(color='#f59e0b', width=2.5),
        fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.15)'
    ))
    
    fig.update_layout(
        height=200, margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False, xaxis_title="Time (s)", yaxis_title="Energy (W)",
        template=get_chart_template(),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_energy")


def render_temp_chart(h, prefix, threshold):
    if len(h['t']) < 3:
        st.info("🌡️ Collecting temperature data...")
        return
    
    t0 = h['t'][0]
    times = [t - t0 for t in h['t']]
    
    colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e',
              '#14b8a6', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899']
    
    fig = go.Figure()
    for i in range(10):
        fig.add_trace(go.Scatter(
            x=times, y=h['temps'][i], mode='lines', name=f'S{i+1}',
            line=dict(color=colors[i], width=1.5),
            hovertemplate=f'Server {i+1}: %{{y:.1f}}°C<extra></extra>'
        ))
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Alarm: {threshold}°C", annotation_position="right")
    
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        xaxis_title="Time (s)", yaxis_title="Temperature (°C)",
        template=get_chart_template(),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_temp")


def render_efficiency_chart(h, prefix):
    if len(h['t']) < 3:
        st.info("📈 Collecting efficiency data...")
        return
    
    t0 = h['t'][0]
    times = [t - t0 for t in h['t']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=h['eff'], mode='lines', name='Efficiency',
        line=dict(color='#10b981', width=2.5),
        fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.15)'
    ))
    
    fig.update_layout(
        height=200, margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False, xaxis_title="Time (s)", yaxis_title="Efficiency %",
        template=get_chart_template(), yaxis=dict(range=[0, 100]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_eff")


def render_action_log(log):
    if not log:
        html = "<div class='action-log-container'>No actions yet...</div>"
    else:
        entries = "<br>".join(log[-10:])
        html = f"<div class='action-log-container'>{entries}</div>"
    
    st.markdown(html, unsafe_allow_html=True)


def render_analysis(state, predicted_eff):
    servers = state.get('servers', 10)
    workload = state.get('workload', 50)
    curr_eff = state.get('current_efficiency', 50)
    max_temp = state.get('max_temperature', 35)
    temps = state.get('server_temperatures', [35] * 10)
    server_states = state.get('server_states', [True] * 10)
    
    # Efficiency status
    if curr_eff >= 80:
        eff_status, eff_color = "🌟 Excellent", "#22c55e"
    elif curr_eff >= 65:
        eff_status, eff_color = "✅ Good", "#10b981"
    elif curr_eff >= 50:
        eff_status, eff_color = "📊 Average", "#f59e0b"
    else:
        eff_status, eff_color = "⚠️ Needs Work", "#ef4444"
    
    # Temperature status
    critical_count = sum(1 for i, t in enumerate(temps) if server_states[i] and t > 55)
    hot_count = sum(1 for i, t in enumerate(temps) if server_states[i] and 45 <= t <= 55)
    
    if critical_count > 0:
        temp_status, temp_color = f"🔴 {critical_count} Critical", "#ef4444"
    elif hot_count > 0:
        temp_status, temp_color = f"🟠 {hot_count} Hot", "#f59e0b"
    elif max_temp > 38:
        temp_status, temp_color = "🟡 Warm", "#eab308"
    else:
        temp_status, temp_color = "🟢 Normal", "#22c55e"
    
    # Load status
    load_per_server = workload / servers if servers > 0 else 0
    if load_per_server > 75:
        load_status, load_color = "🔴 Overloaded", "#ef4444"
    elif load_per_server > 60:
        load_status, load_color = "🟠 High", "#f59e0b"
    elif load_per_server >= 35:
        load_status, load_color = "🟢 Optimal", "#22c55e"
    else:
        load_status, load_color = "🔵 Light", "#3b82f6"
    
    pred_gap = predicted_eff - curr_eff
    gap_sign = "+" if pred_gap >= 0 else ""
    
    # Suggestions
    suggestions = []
    if critical_count > 0:
        suggestions.append(f"🚨 {critical_count} server(s) need immediate cooling!")
    if hot_count > 0:
        suggestions.append(f"🌡️ Increase fans on {hot_count} hot server(s)")
    if load_per_server < 30 and servers > 2:
        suggestions.append("💤 Consider turning off idle servers")
    if load_per_server > 70:
        suggestions.append("📈 Consider adding more servers")
    if max_temp < 30 and servers > 0:
        suggestions.append("💚 Reduce fans to save energy")
    if not suggestions:
        suggestions.append("✅ System running optimally!")
    
    suggestions_html = ""
    for s in suggestions[:3]:
        suggestions_html += f'<div class="suggestion-item">{s}</div>'
    
    html = f"""
    <div class="analysis-card">
        <div class="analysis-title">📊 System Analysis</div>
        <div class="analysis-item">
            <span class="analysis-label">Efficiency</span>
            <span class="analysis-value" style="color:{eff_color};">{eff_status}</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">Temperature</span>
            <span class="analysis-value" style="color:{temp_color};">{temp_status}</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">Load Balance</span>
            <span class="analysis-value" style="color:{load_color};">{load_status}</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">Load/Server</span>
            <span class="analysis-value">{load_per_server:.1f}%</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">Prediction Gap</span>
            <span class="analysis-value">{gap_sign}{pred_gap:.1f}%</span>
        </div>
        <div class="suggestions-container">
            <div class="suggestions-title">💡 Suggestions</div>
            {suggestions_html}
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_tips():
    html = """
    <div class="tips-container">
        <div class="tips-title">💡 Optimization Guide</div>
        <div class="tips-item">🔴 <b>Hot (>45°C)</b>: Increase fan to HIGH</div>
        <div class="tips-item">🟢 <b>Cool (<32°C)</b>: Reduce fan to save energy</div>
        <div class="tips-item">📈 <b>High CPU (>70%)</b>: Turn ON more servers</div>
        <div class="tips-item">📉 <b>Low CPU (<30%)</b>: Turn OFF idle servers</div>
        <div class="tips-item">🎯 <b>Target</b>: 50-60% CPU per server</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

def main():
    init()
    
    # Apply theme CSS
    is_dark = st.session_state.get('dark_mode', True)
    st.markdown(get_theme_css(is_dark), unsafe_allow_html=True)
    
    model, scaler = load_model()
    
    if model is not None:
        st.session_state.ai_mgr.set_ml_model(model, scaler)
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Theme Toggle
        st.markdown("#### 🎨 Theme")
        theme_mode = st.toggle(
            "🌙 Night Mode" if is_dark else "☀️ Day Mode",
            value=is_dark,
            key="theme_toggle"
        )
        
        if theme_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = theme_mode
            st.rerun()
        
        st.markdown("---")
        
        if model:
            st.success("✅ ML Model Active")
            st.caption("RandomForest Regressor")
        else:
            st.warning("⚠️ No ML Model")
        
        st.markdown("---")
        
        thresh = st.slider("🌡️ Temp Alarm", 30, 70, int(st.session_state.alarm.get_threshold()))
        st.session_state.alarm.set_threshold(thresh)
        
        auto = st.checkbox("🔄 Auto-Refresh", True)
        
        st.markdown("---")
        
        if st.button("🔄 Reset", use_container_width=True):
            dark_mode = st.session_state.dark_mode
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.markdown("---")
        st.metric("Rounds", st.session_state.rounds)
        
        st.markdown("---")
        st.markdown("### 🚀 Team AdaptX")
        st.caption("ML-Driven Predictive Optimization")
    
    render_scoreboard()
    
    # Main columns
    c1, c2 = st.columns(2)
    
    # === AI SECTION ===
    with c1:
        st.markdown("""
        <div class="section-card ai-border">
            <div class="section-title">🤖 AI Control (ML-Driven Predictive)</div>
        </div>
        """, unsafe_allow_html=True)
        
        ai_state = st.session_state.ai_sim.step()
        ai_pred = predict(st.session_state.ai_sim, model, scaler)
        ai_curr = ai_state['current_efficiency']
        
        decision = st.session_state.ai_mgr.ai_smart_control(st.session_state.ai_sim, ai_curr)
        
        if decision['should_act']:
            actions = st.session_state.ai_mgr.apply_ai_actions(decision, st.session_state.ai_sim)
            ts = datetime.now().strftime("%H:%M:%S")
            for a in actions:
                st.session_state.ai_log.append(f"[{ts}] {a}")
            st.session_state.ai_log = st.session_state.ai_log[-15:]
        
        st.session_state.ai_mgr.sync_with_simulator(st.session_state.ai_sim)
        update_hist(st.session_state.ai_hist, ai_state)
        st.session_state.ai_scores.append(ai_curr)
        
        render_efficiency_cards(ai_curr, ai_pred)
        render_metrics(ai_state)
        
        with st.expander("🖥️ Server Status", expanded=True):
            render_server_grid(st.session_state.ai_sim, "ai", editable=False)
        
        with st.expander("🧠 AI Decision Log", expanded=False):
            for r in decision['reasoning'][-6:]:
                st.text(r)
        
        with st.expander("📋 AI Actions", expanded=False):
            render_action_log(st.session_state.ai_log)
        
        with st.expander("📊 System Analysis", expanded=False):
            render_analysis(ai_state, ai_pred)
        
        with st.expander("📈 Workload", expanded=False):
            render_workload_chart(st.session_state.ai_hist, "ai")
        
        with st.expander("⚡ Energy", expanded=False):
            render_energy_chart(st.session_state.ai_hist, "ai")
        
        with st.expander("🌡️ Temperature", expanded=True):
            render_temp_chart(st.session_state.ai_hist, "ai", thresh)
        
        with st.expander("📈 Efficiency", expanded=False):
            render_efficiency_chart(st.session_state.ai_hist, "ai")
    
    # === HUMAN SECTION ===
    with c2:
        st.markdown("""
        <div class="section-card human-border">
            <div class="section-title">👤 Human Control (Manual)</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.human_sim.copy_workload_pattern_from(st.session_state.ai_sim)
        
        human_state = st.session_state.human_sim.step()
        human_pred = predict(st.session_state.human_sim, model, scaler)
        human_curr = human_state['current_efficiency']
        
        st.session_state.human_mgr.sync_with_simulator(st.session_state.human_sim)
        update_hist(st.session_state.human_hist, human_state)
        st.session_state.human_scores.append(human_curr)
        
        render_efficiency_cards(human_curr, human_pred)
        render_metrics(human_state)
        
        with st.expander("🖥️ Server Controls", expanded=True):
            render_server_grid(st.session_state.human_sim, "human", editable=True, mgr=st.session_state.human_mgr)
        
        with st.expander("💡 Optimization Tips", expanded=False):
            render_tips()
        
        with st.expander("📊 System Analysis", expanded=False):
            render_analysis(human_state, human_pred)
        
        with st.expander("📈 Workload", expanded=False):
            render_workload_chart(st.session_state.human_hist, "human")
        
        with st.expander("⚡ Energy", expanded=False):
            render_energy_chart(st.session_state.human_hist, "human")
        
        with st.expander("🌡️ Temperature", expanded=True):
            render_temp_chart(st.session_state.human_hist, "human", thresh)
        
        with st.expander("📈 Efficiency", expanded=False):
            render_efficiency_chart(st.session_state.human_hist, "human")
    
    st.session_state.rounds += 1
    
    max_t = max(ai_state['max_temperature'], human_state['max_temperature'])
    alarm = st.session_state.alarm.check_temperature(max_t)
    if alarm['exceeded']:
        st.warning(alarm['message'])
    
    if auto:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()