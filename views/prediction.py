"""
LoanGuard AI - Prediction Engine Page
Interactive 4-card applicant risk assessment form, model inference pipeline, animated risk meter, and feature insights.
"""

import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    COLOR_BORDER,
    COLOR_DARK_NAVY,
    COLOR_LIGHT_ORANGE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    EDUCATION_CHOICES,
    EMPLOYMENT_CHOICES,
)
from utils import apply_fintech_theme, evaluate_risk, load_models, render_html


def render_prediction_page():
    # Load artifacts
    models_dict = load_models()
    if not models_dict["loaded"]:
        st.warning("⚠️ AI models are not available. Please ensure model files are present.")
        return

    pipeline = models_dict["best_model"]
    schema = models_dict["feature_schema"]
    best_model_name = schema.get("best_model_name", "Logistic Regression (Trained Model)")
    is_user_model = models_dict.get("is_user_model", False)

    model_badge_text = "Trained Model Active (24 Features)" if is_user_model else "Production Pipeline Active"

    # ==========================================
    # HEADER
    # ==========================================
    header_html = f"""
    <div style="margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="lg-badge-pill">
                <i class="bi bi-shield-lock-fill"></i> AI LOAN RISK ASSESSMENT
            </span>
            <span class="lg-badge-pill" style="background: #FFF3E8; color: #FF6B00; border: 1px solid #FFD8BE;">
                <i class="bi bi-check-circle-fill"></i> {model_badge_text}
            </span>
        </div>
        <h1 style="font-size: 28px; font-weight: 800; color: #172033; margin: 4px 0 6px 0; letter-spacing: -0.5px;">
            Borrower Default Risk Assessment
        </h1>
        <p style="font-size: 14.5px; color: #6B7280; margin: 0; max-width: 680px;">
            Enter applicant demographic, employment, and financial data below to generate an AI-powered loan default risk evaluation using your trained model.
        </p>
    </div>
    """
    render_html(header_html)

    # ==========================================
    # 3-SECTION FORM DESIGN (MATCHING MODEL FEATURES & UI SPEC)
    # ==========================================

    # ----- SECTION 1: Personal Information -----
    render_html("""
    <div class="lg-form-card-header" style="margin-top: 4px;">
        <div class="lg-form-card-icon"><i class="bi bi-person-fill"></i></div>
        <h3 class="lg-form-card-title" style="font-size: 18px;">Personal Information</h3>
    </div>
    """)
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Applicant legal age at time of application.",
        )
        marital_status = st.selectbox(
            "Marital Status",
            options=["Divorced", "Married", "Single"],
            index=0,
            help="Current legal marital status.",
        )
    with p_col2:
        income = st.number_input(
            "Annual Income",
            min_value=1000.0,
            max_value=1000000.0,
            value=50000.00,
            step=1000.0,
            format="%.2f",
            help="Gross verifiable annual income in USD.",
        )
        has_dependents = st.selectbox(
            "Has Dependents",
            options=["No", "Yes"],
            index=0,
            help="Whether the applicant has dependent family members.",
        )
    with p_col3:
        education = st.selectbox(
            "Education",
            options=["Bachelor's", "High School", "Master's", "PhD"],
            index=0,
            help="Highest completed education credential.",
        )
        has_mortgage = st.selectbox(
            "Has Mortgage",
            options=["No", "Yes"],
            index=0,
            help="Whether applicant currently carries an active home mortgage.",
        )

    # Spacing
    render_html("<div style='height: 18px;'></div>")

    # ----- SECTION 2: Employment & Credit -----
    render_html("""
    <div class="lg-form-card-header">
        <div class="lg-form-card-icon"><i class="bi bi-briefcase-fill"></i></div>
        <h3 class="lg-form-card-title" style="font-size: 18px;">Employment & Credit</h3>
    </div>
    """)
    e_col1, e_col2, e_col3 = st.columns(3)
    with e_col1:
        employment_type = st.selectbox(
            "Employment Type",
            options=["Full-time", "Part-time", "Self-employed", "Unemployed"],
            index=0,
            help="Current borrower employment classification.",
        )
        num_credit_lines = st.number_input(
            "Number of Credit Lines",
            min_value=0,
            max_value=50,
            value=3,
            step=1,
            help="Total active and historical credit lines.",
        )
    with e_col2:
        months_employed = st.number_input(
            "Months Employed",
            min_value=0,
            max_value=600,
            value=60,
            step=1,
            help="Continuous tenure at current employment in months.",
        )
        dti_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            format="%.2f",
            help="Total monthly debt obligations divided by gross earnings.",
        )
    with e_col3:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=700,
            step=1,
            help="Standard FICO bureau credit score rating (300–850).",
        )
        has_cosigner = st.selectbox(
            "Has Co-Signer",
            options=["No", "Yes"],
            index=0,
            help="Whether a secondary guarantor is attached to the loan.",
        )

    # Spacing
    render_html("<div style='height: 18px;'></div>")

    # ----- SECTION 3: Loan Information -----
    render_html("""
    <div class="lg-form-card-header">
        <div class="lg-form-card-icon"><i class="bi bi-cash-coin"></i></div>
        <h3 class="lg-form-card-title" style="font-size: 18px;">Loan Information</h3>
    </div>
    """)
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1000.0,
            max_value=1000000.0,
            value=100000.00,
            step=1000.0,
            format="%.2f",
            help="Total principal loan amount requested in USD.",
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=["Business", "Auto", "Education", "Home", "Other"],
            index=0,
            help="Primary allocation category for borrowed funds.",
        )
    with l_col2:
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=8.00,
            step=0.25,
            format="%.2f",
            help="Contracted annual interest percentage rate (APR).",
        )
    with l_col3:
        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=6,
            max_value=120,
            value=36,
            step=1,
            help="Total scheduled amortization repayment duration in months.",
        )

    # Spacing
    render_html("<div style='height: 24px;'></div>")

    # ==========================================
    # PREDICT BUTTON
    # ==========================================
    predict_clicked = st.button(
        "🛡️ Predict Default Risk",
        key="btn_predict_default_risk",
        use_container_width=True,
    )

    # Execute Prediction Logic
    if predict_clicked:
        with st.spinner("Analyzing borrower profile with Trained LoanGuard AI Pipeline..."):
            time.sleep(0.3)  # Professional responsive feedback

            # Assemble complete applicant dictionary matching model schema
            applicant_data = {
                "Age": int(age),
                "Income": float(income),
                "LoanAmount": float(loan_amount),
                "CreditScore": int(credit_score),
                "MonthsEmployed": int(months_employed),
                "NumCreditLines": int(num_credit_lines),
                "InterestRate": float(interest_rate),
                "LoanTerm": int(loan_term),
                "DTIRatio": float(dti_ratio),
                "EmploymentType": str(employment_type),
                "Education": str(education),
                "HasCoSigner": str(has_cosigner),
                "HasMortgage": str(has_mortgage),
                "LoanPurpose": str(loan_purpose),
                "MaritalStatus": str(marital_status),
                "HasDependents": str(has_dependents),
            }

            try:
                predictor = models_dict.get("predictor")
                if predictor is not None:
                    # Direct inference & explanation through trained model predictor
                    explanation = predictor.explain_applicant(applicant_data)
                    raw_prob = float(explanation["probability"])
                    pred_label = int(explanation["pred_label"])
                    top_risk_drivers = explanation.get("top_risk_drivers", [])
                    top_protective_factors = explanation.get("top_protective_factors", [])
                else:
                    # Fallback to general pipeline
                    input_df = pd.DataFrame([applicant_data])
                    raw_prob = float(pipeline.predict_proba(input_df)[0, 1])
                    pred_label = int(pipeline.predict(input_df)[0])
                    top_risk_drivers = []
                    top_protective_factors = []
            except Exception as e:
                st.error(f"Prediction Pipeline Error: {str(e)}")
                return

            # Map to risk tier and recommendation
            risk_eval = evaluate_risk(raw_prob)

            # Store in session state for persistent display
            st.session_state["latest_prediction"] = {
                "eval": risk_eval,
                "input": applicant_data,
                "model_name": best_model_name,
                "pred_label": pred_label,
                "top_risk_drivers": top_risk_drivers,
                "top_protective_factors": top_protective_factors,
            }

    # ==========================================
    # RESULT DASHBOARD SECTION
    # ==========================================
    if "latest_prediction" in st.session_state:
        res = st.session_state["latest_prediction"]
        eval_data = res["eval"]
        prob_pct = eval_data["percentage"]
        risk_lvl = eval_data["level"]
        risk_color = eval_data["color"]
        badge_class = eval_data["class"]
        verdict = eval_data["verdict"]
        model_used = res["model_name"]
        applicant = res["input"]
        top_risk_drivers = res.get("top_risk_drivers", [])
        top_protective_factors = res.get("top_protective_factors", [])

        render_html(f"""
        <div class="lg-result-card">
            <div class="lg-result-header">
                <div>
                    <span style="font-size: 11.5px; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 0.8px;">
                        AI CREDIT EVALUATION REPORT
                    </span>
                    <h2 style="font-size: 22px; font-weight: 800; color: #172033; margin: 2px 0 0 0;">
                        Risk Assessment Summary
                    </h2>
                </div>
                <span class="lg-risk-badge {badge_class}" style="font-size: 13px; padding: 6px 14px;">
                    {risk_lvl}
                </span>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; align-items: center; margin-bottom: 20px;">
                <div>
                    <div style="font-size: 12px; color: #6B7280; font-weight: 600;">Default Probability</div>
                    <div class="lg-prob-number" style="color: {risk_color};">{prob_pct:.1f}%</div>
                    <span style="font-size: 11.5px; color: #6B7280;">Calibrated model output</span>
                </div>
                <div>
                    <div style="font-size: 12px; color: #6B7280; font-weight: 600;">Decision Verdict</div>
                    <div style="font-size: 18px; font-weight: 800; color: #172033; margin-top: 4px;">{verdict}</div>
                    <span style="font-size: 11.5px; color: #6B7280;">Analytical guidance status</span>
                </div>
                <div>
                    <div style="font-size: 12px; color: #6B7280; font-weight: 600;">Predictive Engine</div>
                    <div style="font-size: 17px; font-weight: 800; color: #FF6B00; margin-top: 4px;">{model_used}</div>
                    <span style="font-size: 11.5px; color: #10B981; font-weight: 600;">✓ Active Trained Pipeline</span>
                </div>
            </div>

            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; font-size: 11.5px; font-weight: 600; color: #6B7280; margin-bottom: 6px;">
                    <span>0% Prime (Safe)</span>
                    <span>25% Low</span>
                    <span>50% Moderate</span>
                    <span>75% High</span>
                    <span>100% Critical Risk</span>
                </div>
                <div class="lg-meter-bar" style="height: 12px;">
                    <div class="lg-meter-fill" style="width: {prob_pct}%;"></div>
                </div>
            </div>

            <div class="lg-recommendation-card">
                <div class="lg-recommendation-title">
                    <i class="bi bi-lightbulb-fill" style="color: #FF6B00;"></i> Analytical Risk Recommendation
                </div>
                <p class="lg-recommendation-text">
                    {eval_data['recommendation']}
                </p>
                <div class="lg-disclaimer">
                    * AI-generated risk assessment for analytical and decision support purposes only. Powered by your uploaded Scikit-Learn trained model.
                </div>
            </div>
        </div>
        """)

        # Applicant-Specific Risk Factors Section
        if top_risk_drivers or top_protective_factors:
            render_html("<div style='height: 20px;'></div>")

            # Risk Drivers HTML
            risk_pills = "".join([
                f'<span style="display:inline-flex; align-items:center; gap:6px; background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:600; margin:3px 4px 3px 0;">'
                f'<i class="bi bi-arrow-up-right"></i> {d["label"]} (+{abs(d["impact"]):.2f})'
                f'</span>'
                for d in top_risk_drivers
            ]) if top_risk_drivers else '<span style="color:#6B7280; font-size:12.5px;">No significant high-risk triggers detected.</span>'

            protective_pills = "".join([
                f'<span style="display:inline-flex; align-items:center; gap:6px; background:#ECFDF5; color:#059669; border:1px solid #A7F3D0; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:600; margin:3px 4px 3px 0;">'
                f'<i class="bi bi-shield-check"></i> {d["label"]} (-{abs(d["impact"]):.2f})'
                f'</span>'
                for d in top_protective_factors
            ]) if top_protective_factors else '<span style="color:#6B7280; font-size:12.5px;">No significant protective offsets detected.</span>'

            render_html(f"""
            <div class="lg-card" style="padding: 20px; margin-bottom: 4px;">
                <h3 style="font-size: 16.5px; font-weight: 700; color: #172033; margin: 0 0 14px 0;">
                    <i class="bi bi-pie-chart-fill" style="color: #FF6B00; margin-right: 6px;"></i> Applicant Risk Factor Breakdown
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px;">
                    <div style="background: #FFFBFB; border: 1px solid #FEE2E2; border-radius: 12px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 700; color: #DC2626; margin-bottom: 8px;">
                            <i class="bi bi-exclamation-triangle-fill"></i> Factors Increasing Default Risk
                        </div>
                        <div style="display: flex; flex-wrap: wrap;">
                            {risk_pills}
                        </div>
                    </div>
                    <div style="background: #F6FDF9; border: 1px solid #D1FAE5; border-radius: 12px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 700; color: #059669; margin-bottom: 8px;">
                            <i class="bi bi-shield-fill-check"></i> Protective & Mitigating Factors
                        </div>
                        <div style="display: flex; flex-wrap: wrap;">
                            {protective_pills}
                        </div>
                    </div>
                </div>
            </div>
            """)

        # Spacing
        render_html("<div style='height: 24px;'></div>")

        # ==========================================
        # ADVANCED PREDICTION INSIGHTS
        # ==========================================
        render_html("""
        <div style="margin-bottom: 14px;">
            <h3 style="font-size: 18px; font-weight: 700; color: #172033; margin: 0 0 4px 0;">
                <i class="bi bi-bar-chart-steps" style="color: #FF6B00; margin-right: 6px;"></i> Primary Risk Drivers & Factor Importance
            </h3>
            <p style="font-size: 13px; color: #6B7280; margin: 0;">
                Calculated directly from the model's feature importance distribution.
            </p>
        </div>
        """)

        ins_col1, ins_col2 = st.columns([1.1, 0.9])

        with ins_col1:
            # Model Feature Importance Chart
            raw_importances = schema.get("feature_importances", {})
            if raw_importances:
                # Top 7 factors
                top_items = list(raw_importances.items())[:7]
                features_names = [item[0] for item in reversed(top_items)]
                importance_vals = [item[1] for item in reversed(top_items)]

                fig_fi = go.Figure(data=[
                    go.Bar(
                        x=importance_vals,
                        y=features_names,
                        orientation="h",
                        marker=dict(color=COLOR_PRIMARY, cornerradius=4),
                        text=[f"{v * 100:.1f}%" for v in importance_vals],
                        textposition="outside",
                        hovertemplate="Feature: %{y}<br>Model Weight: %{x:.4f}<extra></extra>",
                    )
                ])
                apply_fintech_theme(fig_fi, title="Model Feature Weights (Top Risk Factors)", height=320, show_legend=False)
                fig_fi.update_xaxes(title="Normalized Importance Share")
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance is not directly available for the selected linear baseline.")

        with ins_col2:
            # Financial Ratio & Profile Breakdown
            lti_ratio = applicant["LoanAmount"] / applicant["Income"] if applicant["Income"] > 0 else 0
            lti_status = "Elevated" if lti_ratio > 1.5 else "Optimal"
            lti_color = "#EF4444" if lti_ratio > 1.5 else "#10B981"

            dti_status = "Elevated" if applicant["DTIRatio"] > 0.43 else "Optimal"
            dti_color = "#EF4444" if applicant["DTIRatio"] > 0.43 else "#10B981"

            cs_status = "Subprime" if applicant["CreditScore"] < 580 else ("Prime" if applicant["CreditScore"] >= 670 else "Near-Prime")
            cs_color = "#EF4444" if applicant["CreditScore"] < 580 else ("#10B981" if applicant["CreditScore"] >= 670 else "#F59E0B")

            render_html(f"""
            <div class="lg-card" style="padding: 20px;">
                <h4 style="font-size: 15px; font-weight: 700; color: #172033; margin: 0 0 14px 0;">
                    <i class="bi bi-calculator" style="color: #FF6B00; margin-right: 6px;"></i> Applicant Ratio Health
                </h4>
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 8px; border-bottom: 1px solid #F1F5F9;">
                        <div>
                            <div style="font-size: 12.5px; font-weight: 600; color: #172033;">Loan-to-Income (LTI)</div>
                            <div style="font-size: 11px; color: #6B7280;">Principal vs Annual Earnings</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14.5px; font-weight: 800; color: #172033;">{lti_ratio:.2f}x</div>
                            <span style="font-size: 10.5px; font-weight: 700; color: {lti_color};">{lti_status}</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 8px; border-bottom: 1px solid #F1F5F9;">
                        <div>
                            <div style="font-size: 12.5px; font-weight: 600; color: #172033;">Debt-to-Income (DTI)</div>
                            <div style="font-size: 11px; color: #6B7280;">Total Debt Obligations</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14.5px; font-weight: 800; color: #172033;">{applicant['DTIRatio'] * 100:.1f}%</div>
                            <span style="font-size: 10.5px; font-weight: 700; color: {dti_color};">{dti_status}</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 8px; border-bottom: 1px solid #F1F5F9;">
                        <div>
                            <div style="font-size: 12.5px; font-weight: 600; color: #172033;">Credit Score Rating</div>
                            <div style="font-size: 11px; color: #6B7280;">FICO Bureau Bracket</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14.5px; font-weight: 800; color: #172033;">{applicant['CreditScore']}</div>
                            <span style="font-size: 10.5px; font-weight: 700; color: {cs_color};">{cs_status}</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 12.5px; font-weight: 600; color: #172033;">Contract Interest Rate</div>
                            <div style="font-size: 11px; color: #6B7280;">Annual Percentage Rate</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14.5px; font-weight: 800; color: #FF6B00;">{applicant['InterestRate']:.2f}%</div>
                            <span style="font-size: 10.5px; font-weight: 700; color: #6B7280;">Fixed APR</span>
                        </div>
                    </div>
                </div>
            </div>
            """)
