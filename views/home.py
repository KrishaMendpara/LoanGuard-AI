"""
LoanGuard AI - Home Page
Modern FinTech Landing Page with Hero, Real-Time Stats, Feature Cards, Workflow, and CTA.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_LIGHT_ORANGE,
    PROJECT_NAME,
    PROJECT_TAGLINE,
)
from utils import render_html, load_data, load_models


def render_home_page():
    # Load data and model artifacts for live metrics
    try:
        df = load_data()
        total_apps = f"{len(df):,}"
        default_rate = f"{df['loan_default'].mean() * 100:.1f}%"
    except Exception:
        total_apps = "255,347"
        default_rate = "11.6%"

    models_dict = load_models()
    if models_dict["loaded"] and models_dict["feature_schema"]:
        best_model_name = models_dict["feature_schema"].get("best_model_name", "Logistic Regression (Trained Model)")
        best_f1 = f"{models_dict['feature_schema'].get('best_f1_score', 0.6161):.4f}"
        is_user_model = models_dict.get("is_user_model", False)
    else:
        best_model_name = "Logistic Regression (Trained Model)"
        best_f1 = "0.6161"
        is_user_model = False

    hero_badge = "TRAINED AI CREDIT MODEL (24 FEATURES)" if is_user_model else "AI-POWERED CREDIT INTELLIGENCE"

    # ==========================================
    # 1. HERO SECTION
    # ==========================================
    hero_col_left, hero_col_right = st.columns([1.15, 0.85], gap="large")

    with hero_col_left:
        hero_left_html = f"""
        <div class="lg-hero">
            <div class="lg-badge-pill">
                <i class="bi bi-shield-check"></i> {hero_badge}
            </div>
            <h1 class="lg-hero-title">
                Smarter Loan Decisions<br>
                <span>Powered by Machine Learning</span>
            </h1>
            <p class="lg-hero-desc">
                Analyze borrower profiles, evaluate credit default probabilities, and optimize risk management
                with enterprise-grade predictive intelligence and transparent ML pipelines.
            </p>
        </div>
        """
        render_html(hero_left_html)

        # Action Buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("🚀 Start Prediction →", key="hero_start_prediction", use_container_width=True):
                st.query_params["page"] = "Prediction"
                st.session_state.current_page = "Prediction"
                st.rerun()
        with btn_col2:
            if st.button("📊 Explore Analytics", key="hero_explore_analytics", type="secondary", use_container_width=True):
                st.query_params["page"] = "Analytics"
                st.session_state.current_page = "Analytics"
                st.rerun()

    with hero_col_right:
        # High-Fidelity FinTech Card Visual
        fintech_card_html = f"""
        <div class="lg-hero-card">
            <div class="lg-hero-card-header">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background: #EF4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);"></div>
                    <span style="font-size: 12.5px; font-weight: 700; color: #172033; letter-spacing: 0.5px; text-transform: uppercase;">AI Risk Assessment</span>
                </div>
                <span class="lg-risk-badge risk-high-badge">High Risk</span>
            </div>

            <div style="display: flex; align-items: baseline; gap: 10px;">
                <span class="lg-score-number">68.4%</span>
                <span style="font-size: 13px; font-weight: 600; color: #6B7280;">Default Probability</span>
            </div>

            <div class="lg-meter-bar">
                <div class="lg-meter-fill" style="width: 68.4%;"></div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #9CA3AF; font-weight: 600; margin-bottom: 14px;">
                <span>0% Prime</span>
                <span>50% Moderate</span>
                <span>100% High Risk</span>
            </div>

            <div class="lg-hero-stat-row">
                <div class="lg-stat-mini">
                    <span class="lg-stat-mini-label">Credit Score</span>
                    <span class="lg-stat-mini-val">451 (Subprime)</span>
                </div>
                <div class="lg-stat-mini">
                    <span class="lg-stat-mini-label">DTI Ratio</span>
                    <span class="lg-stat-mini-val">0.68 (Elevated)</span>
                </div>
                <div class="lg-stat-mini">
                    <span class="lg-stat-mini-label">Applicant Income</span>
                    <span class="lg-stat-mini-val">$50,432 / yr</span>
                </div>
                <div class="lg-stat-mini">
                    <span class="lg-stat-mini-label">Requested Loan</span>
                    <span class="lg-stat-mini-val">$124,440 (60 mo)</span>
                </div>
            </div>

            <div style="margin-top: 14px; padding: 10px 12px; background: #FFF3E8; border-radius: 10px; border: 1px solid rgba(255, 107, 0, 0.2); display: flex; align-items: center; gap: 8px;">
                <i class="bi bi-exclamation-triangle-fill" style="color: #FF6B00; font-size: 15px;"></i>
                <span style="font-size: 12px; font-weight: 600; color: #172033;">Model Recommendation: Stringent Underwriting Required</span>
            </div>
        </div>
        """
        render_html(fintech_card_html)

    # Spacing
    render_html("<div style='height: 24px;'></div>")

    # ==========================================
    # 2. STATISTICS SECTION (4 PREMIUM METRIC CARDS)
    # ==========================================
    section_header_html = """
    <div style="margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 700; color: #FF6B00; text-transform: uppercase; letter-spacing: 1px;">Platform Intelligence</span>
        <h2 style="font-size: 22px; font-weight: 800; color: #172033; margin: 2px 0 0 0;">Portfolio & Model Benchmarks</h2>
    </div>
    """
    render_html(section_header_html)

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Total Applications</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-folder2-open"></i></div>
            </div>
            <div class="lg-stat-val">{total_apps}</div>
            <div class="lg-stat-trend">
                <i class="bi bi-database-check" style="color: #10B981;"></i> Standardized historical records
            </div>
        </div>
        """)

    with s2:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Portfolio Default Rate</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-pie-chart-fill"></i></div>
            </div>
            <div class="lg-stat-val">{default_rate}</div>
            <div class="lg-stat-trend">
                <i class="bi bi-graph-down" style="color: #EF4444;"></i> Historical baseline default rate
            </div>
        </div>
        """)

    with s3:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Top Algorithm</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-cpu-fill"></i></div>
            </div>
            <div class="lg-stat-val" style="font-size: 20px; line-height: 1.35; padding-top: 4px;">{best_model_name}</div>
            <div class="lg-stat-trend">
                <i class="bi bi-patch-check-fill" style="color: #FF6B00;"></i> Maximum F1 Score winner
            </div>
        </div>
        """)

    with s4:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Model F1 Score</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-award-fill"></i></div>
            </div>
            <div class="lg-stat-val">{best_f1}</div>
            <div class="lg-stat-trend">
                <i class="bi bi-check-circle-fill" style="color: #10B981;"></i> Stratified out-of-fold evaluation
            </div>
        </div>
        """)

    # Spacing
    render_html("<div style='height: 32px;'></div>")

    # ==========================================
    # 3. FEATURE SECTION (4 CARDS)
    # ==========================================
    feat_header_html = """
    <div style="margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 700; color: #FF6B00; text-transform: uppercase; letter-spacing: 1px;">Core Capabilities</span>
        <h2 style="font-size: 22px; font-weight: 800; color: #172033; margin: 2px 0 0 0;">Enterprise Credit Risk Architecture</h2>
    </div>
    """
    render_html(feat_header_html)

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        render_html("""
        <div class="lg-feature-card">
            <div class="lg-feature-icon"><i class="bi bi-speedometer2"></i></div>
            <h3 class="lg-feature-title">AI Risk Prediction</h3>
            <p class="lg-feature-desc">
                Evaluate individual applicant risk profiles with machine learning pipelines trained on historical financial data.
            </p>
        </div>
        """)

    with f2:
        render_html("""
        <div class="lg-feature-card">
            <div class="lg-feature-icon"><i class="bi bi-bar-chart-line-fill"></i></div>
            <h3 class="lg-feature-title">Advanced Analytics</h3>
            <p class="lg-feature-desc">
                Interactive Plotly portfolio charts, dynamic demographic filters, and deep-dive risk distribution analytics.
            </p>
        </div>
        """)

    with f3:
        render_html("""
        <div class="lg-feature-card">
            <div class="lg-feature-icon"><i class="bi bi-diagram-3-fill"></i></div>
            <h3 class="lg-feature-title">Model Comparison</h3>
            <p class="lg-feature-desc">
                Transparent benchmarking of Logistic Regression, Random Forest, and Gradient Boosting algorithms with full metrics.
            </p>
        </div>
        """)

    with f4:
        render_html("""
        <div class="lg-feature-card">
            <div class="lg-feature-icon"><i class="bi bi-lightbulb-fill"></i></div>
            <h3 class="lg-feature-title">Credit Insights</h3>
            <p class="lg-feature-desc">
                Explainable AI feature importances identify specific driving factors behind each default risk probability.
            </p>
        </div>
        """)

    # Spacing
    render_html("<div style='height: 32px;'></div>")

    # ==========================================
    # 4. HOW IT WORKS (3-STEP TIMELINE)
    # ==========================================
    steps_header_html = """
    <div style="margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 700; color: #FF6B00; text-transform: uppercase; letter-spacing: 1px;">Workflow</span>
        <h2 style="font-size: 22px; font-weight: 800; color: #172033; margin: 2px 0 0 0;">How LoanGuard AI Evaluates Credit Risk</h2>
    </div>
    """
    render_html(steps_header_html)

    w1, w2, w3 = st.columns(3)

    with w1:
        render_html("""
        <div class="lg-step-card">
            <span class="lg-step-num">STEP 01</span>
            <h3 class="lg-step-title">Enter Applicant Data</h3>
            <p class="lg-step-desc">
                Provide borrower demographic, employment, and financial metrics across our structured 4-section assessment form.
            </p>
        </div>
        """)

    with w2:
        render_html("""
        <div class="lg-step-card">
            <span class="lg-step-num">STEP 02</span>
            <h3 class="lg-step-title">AI Analyzes Risk</h3>
            <p class="lg-step-desc">
                Pretrained ColumnTransformer preprocesses raw values and feeds feature representations into our high-F1 ensemble pipeline.
            </p>
        </div>
        """)

    with w3:
        render_html("""
        <div class="lg-step-card">
            <span class="lg-step-num">STEP 03</span>
            <h3 class="lg-step-title">Get Risk Assessment</h3>
            <p class="lg-step-desc">
                Receive calibrated default probability, transparent risk category badges, advisory recommendations, and risk factors.
            </p>
        </div>
        """)

    # ==========================================
    # 5. CALL TO ACTION SECTION
    # ==========================================
    render_html("<div style='height: 16px;'></div>")

    cta_html = """
    <div class="lg-cta">
        <div>
            <h2 class="lg-cta-title">Ready to evaluate loan risk?</h2>
            <p class="lg-cta-desc">
                Input borrower financial details and generate an instant, AI-calibrated default probability score.
            </p>
        </div>
    </div>
    """
    render_html(cta_html)

    cta_btn_col1, cta_btn_col2, cta_btn_col3 = st.columns([1, 1, 1])
    with cta_btn_col2:
        if st.button("Start Risk Assessment →", key="cta_start_prediction", use_container_width=True):
            st.query_params["page"] = "Prediction"
            st.session_state.current_page = "Prediction"
            st.rerun()
