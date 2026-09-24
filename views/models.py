"""
LoanGuard AI - Model Intelligence Page
Compares machine learning algorithms, visualizes performance metrics, and presents model explanations.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    COLOR_BORDER,
    COLOR_DARK_NAVY,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
)
from utils import apply_fintech_theme, load_models, render_html


def render_models_page():
    # Header
    header_html = """
    <div style="margin-bottom: 24px;">
        <span class="lg-badge-pill" style="margin-bottom: 8px;">
            <i class="bi bi-cpu-fill"></i> MODEL INTELLIGENCE
        </span>
        <h1 style="font-size: 28px; font-weight: 800; color: #172033; margin: 4px 0 6px 0; letter-spacing: -0.5px;">
            Algorithm Benchmarks & Selection
        </h1>
        <p style="font-size: 14.5px; color: #6B7280; margin: 0; max-width: 680px;">
            Compare machine learning models and understand which algorithm performs best for credit risk prediction on imbalanced loan portfolios.
        </p>
    </div>
    """
    render_html(header_html)

    # Load artifacts
    models_dict = load_models()

    if not models_dict["loaded"]:
        st.warning(
            "⚠️ AI models are not trained yet. Please run `python train_model.py` to generate the benchmark artifacts."
        )
        if st.button("🚀 Train Models Now", key="train_models_btn"):
            with st.spinner("Training models..."):
                import train_model
                train_model.train_pipeline()
                st.cache_resource.clear()
                st.rerun()
        return

    results_data = models_dict["model_results"]
    schema = models_dict["feature_schema"]
    df_results = results_data["comparison_table"]
    best_model_name = results_data["best_model_name"]

    # Locate best model metrics
    best_row = df_results[df_results["Model"] == best_model_name].iloc[0]

    # ==========================================
    # 1. MODEL SUMMARY CARDS (5 METRIC CARDS)
    # ==========================================
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Best Model</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-trophy-fill"></i></div>
            </div>
            <div class="lg-stat-val" style="font-size: 18px; line-height: 1.3;">{best_model_name}</div>
            <div class="lg-stat-trend"><i class="bi bi-check2-circle" style="color:#10B981;"></i> Selected pipeline</div>
        </div>
        """)

    with m2:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Accuracy</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-bullseye"></i></div>
            </div>
            <div class="lg-stat-val" style="font-size: 22px;">{best_row['Accuracy'] * 100:.1f}%</div>
            <div class="lg-stat-trend">Overall classification</div>
        </div>
        """)

    with m3:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Precision</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-crosshair"></i></div>
            </div>
            <div class="lg-stat-val" style="font-size: 22px;">{best_row['Precision'] * 100:.1f}%</div>
            <div class="lg-stat-trend">Positive predictive value</div>
        </div>
        """)

    with m4:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">Recall</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-arrow-repeat"></i></div>
            </div>
            <div class="lg-stat-val" style="font-size: 22px;">{best_row['Recall'] * 100:.1f}%</div>
            <div class="lg-stat-trend">Sensitivity to default</div>
        </div>
        """)

    with m5:
        render_html(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-top">
                <span class="lg-stat-label">F1 Score</span>
                <div class="lg-stat-icon-wrap"><i class="bi bi-award-fill"></i></div>
            </div>
            <div class="lg-stat-val" style="font-size: 22px; color:#FF6B00;">{best_row['F1 Score']:.4f}</div>
            <div class="lg-stat-trend">Primary selection metric</div>
        </div>
        """)

    # Spacing
    render_html("<div style='height: 24px;'></div>")

    # ==========================================
    # 2. MODEL COMPARISON TABLE
    # ==========================================
    render_html("""
    <div style="margin-bottom: 12px;">
        <h3 style="font-size: 18px; font-weight: 700; color: #172033; margin: 0 0 4px 0;">
            <i class="bi bi-table" style="color: #FF6B00; margin-right: 6px;"></i> Model Performance Comparison
        </h3>
        <p style="font-size: 13px; color: #6B7280; margin: 0;">
            Evaluated on stratified out-of-sample test split with balanced class weighting.
        </p>
    </div>
    """)

    table_rows = []
    for _, row in df_results.iterrows():
        is_best = row["Model"] == best_model_name
        status_badge = (
            '<span class="lg-best-badge"><i class="bi bi-trophy-fill"></i> BEST MODEL</span>'
            if is_best
            else '<span style="color:#6B7280; font-size:12px; font-weight:600;">Candidate</span>'
        )
        row_bg = "background-color: #FFF9F5;" if is_best else ""
        font_weight = "font-weight: 700;" if is_best else "font-weight: 500;"

        table_rows.append(f"""
        <tr style="{row_bg}">
            <td style="{font_weight} color: #172033;">
                {row['Model']} {'<i class="bi bi-star-fill" style="color:#FF6B00; font-size:12px; margin-left:4px;"></i>' if is_best else ''}
            </td>
            <td><b>{row['Accuracy'] * 100:.2f}%</b></td>
            <td>{row['Precision'] * 100:.2f}%</td>
            <td>{row['Recall'] * 100:.2f}%</td>
            <td style="color: {'#FF6B00' if is_best else '#172033'}; font-weight: 700;">{row['F1 Score']:.4f}</td>
            <td>{status_badge}</td>
        </tr>
        """)

    table_html = f"""
    <div class="lg-table-card">
        <table class="lg-table">
            <thead>
                <tr>
                    <th>Algorithm</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Optimization Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    """
    render_html(table_html)

    # Spacing
    render_html("<div style='height: 20px;'></div>")

    # ==========================================
    # 3. MODEL PERFORMANCE GROUPED BAR CHART
    # ==========================================
    models_list = df_results["Model"].tolist()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig = go.Figure()
    colors = ["#172033", "#FF8A3D", "#10B981", "#FF6B00"]

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=models_list,
            y=df_results[metric],
            marker=dict(color=colors[i], cornerradius=6),
            text=[f"{v:.3f}" for v in df_results[metric]],
            textposition="outside",
            hovertemplate=f"Algorithm: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_fintech_theme(fig, title="Grouped Benchmark: Accuracy vs Precision vs Recall vs F1 Score", height=380)
    fig.update_yaxes(range=[0, 1.05], title="Score (0.0 – 1.0)")
    st.plotly_chart(fig, use_container_width=True)

    # Spacing
    render_html("<div style='height: 24px;'></div>")

    # ==========================================
    # 4. MODEL EXPLANATION CARDS (3 CARDS)
    # ==========================================
    is_user_model = models_dict.get("is_user_model", False)
    lr_badge = '<span class="lg-best-badge"><i class="bi bi-check-circle-fill"></i> ACTIVE MODEL</span>' if is_user_model else ''
    lr_border = 'border: 2px solid rgba(255, 107, 0, 0.5);' if is_user_model else ''
    rf_badge = '' if is_user_model else '<span class="lg-best-badge">BEST MODEL</span>'
    rf_border = '' if is_user_model else 'border: 2px solid rgba(255, 107, 0, 0.4);'

    render_html("""
    <div style="margin-bottom: 14px;">
        <h3 style="font-size: 18px; font-weight: 700; color: #172033; margin: 0 0 4px 0;">
            <i class="bi bi-info-circle-fill" style="color: #FF6B00; margin-right: 6px;"></i> Machine Learning Architectural Analysis
        </h3>
        <p style="font-size: 13px; color: #6B7280; margin: 0;">
            Understanding structural trade-offs between linear baselines and tree-based ensembles.
        </p>
    </div>
    """)

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        render_html(f"""
        <div class="lg-feature-card" style="{lr_border}">
            <div class="lg-feature-icon" style="background:#FFF3E8; color:#FF6B00;"><i class="bi bi-sliders"></i></div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h4 class="lg-feature-title" style="margin: 0;">Logistic Regression</h4>
                {lr_badge}
            </div>
            <p class="lg-feature-desc">
                Interpretable statistical model with standard feature normalization.
                Applies calibrated sigmoid activation over 24 standardized credit attributes for transparent, rapid default risk probability estimation.
            </p>
        </div>
        """)

    with exp2:
        render_html(f"""
        <div class="lg-feature-card" style="{rf_border}">
            <div class="lg-feature-icon"><i class="bi bi-diagram-3-fill"></i></div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h4 class="lg-feature-title" style="margin: 0;">Random Forest</h4>
                {rf_badge}
            </div>
            <p class="lg-feature-desc">
                Ensemble model capable of capturing nonlinear relationships and complex interactions across disparate financial features.
                Constructs diverse bootstrap decision trees to achieve high harmonic balance between false alarms and missed defaults.
            </p>
        </div>
        """)

    with exp3:
        render_html("""
        <div class="lg-feature-card">
            <div class="lg-feature-icon" style="background:#ECFDF5; color:#10B981;"><i class="bi bi-layers-fill"></i></div>
            <h4 class="lg-feature-title">Gradient Boosting</h4>
            <p class="lg-feature-desc">
                Sequential ensemble approach that improves prediction performance by correcting previous model errors iteratively.
                Delivers high specificity on prime borrowers.
            </p>
        </div>
        """)

    # Spacing
    render_html("<div style='height: 24px;'></div>")

    # ==========================================
    # 5. MODEL INSIGHTS & SPECIFICATIONS
    # ==========================================
    train_records = schema.get("total_train_records", 60000)
    test_records = schema.get("total_test_records", 15000)
    total_features = schema.get("total_features", 24)

    pipeline_sub = "Active Trained Model" if is_user_model else "Scikit-Learn Production Pipeline"

    insights_html = f"""
    <div class="lg-card">
        <div class="lg-card-header">
            <h3 class="lg-card-title"><i class="bi bi-gear-wide-connected"></i> Model Training & Deployment Metadata</h3>
            <span style="font-size: 12px; font-weight: 700; color: #FF6B00; background: #FFF3E8; padding: 4px 10px; border-radius: 12px;">
                {pipeline_sub}
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-top: 6px;">
            <div style="padding: 12px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E5E7EB;">
                <div style="font-size: 11px; color: #6B7280; font-weight: 600;">Active Model</div>
                <div style="font-size: 15px; font-weight: 800; color: #172033; margin-top: 4px;">{best_model_name}</div>
            </div>
            <div style="padding: 12px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E5E7EB;">
                <div style="font-size: 11px; color: #6B7280; font-weight: 600;">Feature Scaler</div>
                <div style="font-size: 15px; font-weight: 800; color: #FF6B00; margin-top: 4px;">StandardScaler</div>
            </div>
            <div style="padding: 12px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E5E7EB;">
                <div style="font-size: 11px; color: #6B7280; font-weight: 600;">Input Features</div>
                <div style="font-size: 15px; font-weight: 800; color: #172033; margin-top: 4px;">{total_features} Normalized</div>
            </div>
            <div style="padding: 12px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E5E7EB;">
                <div style="font-size: 11px; color: #6B7280; font-weight: 600;">Model Intercept</div>
                <div style="font-size: 15px; font-weight: 800; color: #172033; margin-top: 4px;">{schema.get('intercept', -2.375):.3f}</div>
            </div>
            <div style="padding: 12px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E5E7EB;">
                <div style="font-size: 11px; color: #6B7280; font-weight: 600;">Decision Boundary</div>
                <div style="font-size: 15px; font-weight: 800; color: #10B981; margin-top: 4px;">Calibrated Tiers</div>
            </div>
        </div>
    </div>
    """
    render_html(insights_html)

    # If user model has coefficients, display interactive coefficient weight chart
    coefs = schema.get("coefficients", {})
    if coefs:
        render_html("<div style='height: 24px;'></div>")
        render_html("""
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 18px; font-weight: 700; color: #172033; margin: 0 0 4px 0;">
                <i class="bi bi-sliders2-vertical" style="color: #FF6B00; margin-right: 6px;"></i> Trained Model Coefficients (Log-Odds Weights)
            </h3>
            <p style="font-size: 13px; color: #6B7280; margin: 0;">
                Positive weights increase default risk probability; negative weights reduce risk (protective factors).
            </p>
        </div>
        """)

        sorted_coef_items = sorted(coefs.items(), key=lambda x: x[1])
        f_names = [x[0] for x in sorted_coef_items]
        f_vals = [x[1] for x in sorted_coef_items]
        f_colors = ["#EF4444" if v > 0 else "#10B981" for v in f_vals]

        fig_coef = go.Figure(data=[
            go.Bar(
                x=f_vals,
                y=f_names,
                orientation="h",
                marker=dict(color=f_colors, cornerradius=4),
                text=[f"{v:+.3f}" for v in f_vals],
                textposition="outside",
                hovertemplate="Feature: %{y}<br>Coefficient: %{x:.4f}<extra></extra>",
            )
        ])
        apply_fintech_theme(fig_coef, title="Learned Feature Coefficients (Risk Drivers vs Protective Offsets)", height=560, show_legend=False)
        fig_coef.update_xaxes(title="Logistic Regression Weight (Log-Odds Impact)")
        st.plotly_chart(fig_coef, use_container_width=True)
