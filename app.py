"""
LoanGuard AI - Intelligent Credit Risk & Loan Default Prediction
Main Application Entry Point & Top Horizontal Navbar Orchestrator
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st

from config import (
    APP_ICON,
    CSS_PATH,
    JS_PATH,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    VERSION,
)
from utils import render_html

# ==========================================
# 1. STREAMLIT GLOBAL CONFIGURATION
# ==========================================
st.set_page_config(
    page_title=f"{PROJECT_NAME} - {PROJECT_TAGLINE}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================
# 2. INJECT DESIGN SYSTEM & ASSETS
# ==========================================
def inject_assets():
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
            render_html(f"<style>{css_content}</style>")

    if JS_PATH.exists():
        with open(JS_PATH, "r", encoding="utf-8") as f:
            js_content = f.read()
            render_html(f"<script>{js_content}</script>")


inject_assets()

# ==========================================
# 3. BIDIRECTIONAL PAGE STATE SYNC
# ==========================================
VALID_PAGES = ["Home", "Analytics", "Models", "Prediction"]

# Check query parameters first
query_page = st.query_params.get("page", None)

if query_page and query_page in VALID_PAGES:
    st.session_state["current_page"] = query_page
elif "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"
    st.query_params["page"] = "Home"

current_page = st.session_state.get("current_page", "Home")


# ==========================================
# 4. TOP HORIZONTAL NAVBAR COMPONENT
# ==========================================
def render_top_navbar(active_page: str):
    nav_home_active = "active" if active_page == "Home" else ""
    nav_analytics_active = "active" if active_page == "Analytics" else ""
    nav_models_active = "active" if active_page == "Models" else ""
    nav_prediction_active = "active" if active_page == "Prediction" else ""

    navbar_html = f"""
    <nav class="lg-navbar">
        <div class="lg-navbar-container">
            <div class="lg-brand">
                <a href="?page=Home" target="_self">
                    <div class="lg-brand-icon">
                        <i class="bi bi-shield-shaded"></i>
                    </div>
                    <div class="lg-brand-text">
                        <div class="lg-brand-title">LoanGuard<span>AI</span></div>
                        <small class="lg-brand-sub">INTELLIGENT CREDIT RISK & LOAN DEFAULT PREDICTION</small>
                    </div>
                </a>
            </div>

            <div class="lg-navigation">
                <a href="?page=Home" target="_self" class="lg-nav-item {nav_home_active}">
                    <span class="nav-icon">🏠</span> <span>Home</span>
                </a>
                <a href="?page=Analytics" target="_self" class="lg-nav-item {nav_analytics_active}">
                    <span class="nav-icon">📊</span> <span>Analytics</span>
                </a>
                <a href="?page=Models" target="_self" class="lg-nav-item {nav_models_active}">
                    <span class="nav-icon">🤖</span> <span>Models</span>
                </a>
                <a href="?page=Prediction" target="_self" class="lg-nav-item {nav_prediction_active}">
                    <span class="nav-icon">🎯</span> <span>Prediction</span>
                </a>
            </div>

            <div class="lg-navbar-action">
                <a href="?page=Prediction" target="_self" class="lg-action-btn">
                    Assess Risk →
                </a>
            </div>
        </div>
    </nav>
    """
    render_html(navbar_html)


render_top_navbar(current_page)


# ==========================================
# 5. PAGE ROUTER (NO SIDEBAR)
# ==========================================
if current_page == "Home":
    from views.home import render_home_page
    render_home_page()

elif current_page == "Analytics":
    from views.analytics import render_analytics_page
    render_analytics_page()

elif current_page == "Models":
    from views.models import render_models_page
    render_models_page()

elif current_page == "Prediction":
    from views.prediction import render_prediction_page
    render_prediction_page()

else:
    st.error(f"Unknown page requested: {current_page}")
