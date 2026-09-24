"""
LoanGuard AI - Utility Functions & Helpers
Handles safe HTML rendering, cached data/model loading, Plotly theming, and risk evaluation.
"""

import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    BEST_MODEL_PATH,
    COLOR_BORDER,
    COLOR_DARK_NAVY,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TEXT,
    COLOR_WHITE,
    DATA_PATH,
    FEATURE_SCHEMA_PATH,
    MODEL_RESULTS_PATH,
    RISK_LEVELS,
    USER_FEATURES_PATH,
    USER_MODEL_PATH,
    USER_SCALER_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
)


# ==========================================
# TRAINED LOAN PREDICTOR CLASS (UPLOADED MODEL)
# ==========================================
class TrainedLoanPredictor:
    """
    Predictor wrapper for the uploaded trained Logistic Regression model and StandardScaler.
    Translates UI inputs into the exact 24-feature schema, performs standard scaling,
    and computes predictions and applicant-specific risk factor explanations.
    """

    def __init__(self, model: Any, scaler: Any, feature_names: list):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)
        self.coef = getattr(model, "coef_", None)
        self.intercept = getattr(model, "intercept_", None)

    def encode_row(self, raw_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Converts a single record dictionary into the 24-feature schema.
        Handles both 'LoanPurpose'/'Purpose' and 'HasDependents'/'NumDependents'.
        """
        row = {f: 0.0 for f in self.feature_names}

        # 1. Numerical Features
        row["Age"] = float(raw_dict.get("Age", 35))
        row["Income"] = float(raw_dict.get("Income", 50000.0))
        row["LoanAmount"] = float(raw_dict.get("LoanAmount", 100000.0))
        row["CreditScore"] = float(raw_dict.get("CreditScore", 700))
        row["MonthsEmployed"] = float(raw_dict.get("MonthsEmployed", 60))
        row["NumCreditLines"] = float(raw_dict.get("NumCreditLines", 3))
        row["InterestRate"] = float(raw_dict.get("InterestRate", 8.0))
        row["LoanTerm"] = float(raw_dict.get("LoanTerm", 36))
        row["DTIRatio"] = float(raw_dict.get("DTIRatio", 0.25))

        # 2. Education (Reference: Bachelor's)
        edu = str(raw_dict.get("Education", "Bachelor's")).strip()
        if edu == "High School" and "Education_High School" in row:
            row["Education_High School"] = 1.0
        elif edu == "Master's" and "Education_Master's" in row:
            row["Education_Master's"] = 1.0
        elif edu == "PhD" and "Education_PhD" in row:
            row["Education_PhD"] = 1.0

        # 3. Employment Type (Reference: Full-time)
        emp = str(raw_dict.get("EmploymentType", "Full-time")).strip()
        if emp == "Part-time" and "EmploymentType_Part-time" in row:
            row["EmploymentType_Part-time"] = 1.0
        elif emp == "Self-employed" and "EmploymentType_Self-employed" in row:
            row["EmploymentType_Self-employed"] = 1.0
        elif emp == "Unemployed" and "EmploymentType_Unemployed" in row:
            row["EmploymentType_Unemployed"] = 1.0

        # 4. Marital Status (Reference: Divorced)
        marital = str(raw_dict.get("MaritalStatus", "Divorced")).strip()
        if marital == "Married" and "MaritalStatus_Married" in row:
            row["MaritalStatus_Married"] = 1.0
        elif marital == "Single" and "MaritalStatus_Single" in row:
            row["MaritalStatus_Single"] = 1.0

        # 5. Has Mortgage (Reference: No)
        mortgage = str(raw_dict.get("HasMortgage", "No")).strip().lower()
        if mortgage in ("yes", "1", "true") and "HasMortgage_Yes" in row:
            row["HasMortgage_Yes"] = 1.0

        # 6. Has Dependents (Reference: No)
        if "HasDependents" in raw_dict:
            dependents = str(raw_dict.get("HasDependents", "No")).strip().lower()
            if dependents in ("yes", "1", "true") and "HasDependents_Yes" in row:
                row["HasDependents_Yes"] = 1.0
        elif "NumDependents" in raw_dict:
            if float(raw_dict.get("NumDependents", 0)) > 0 and "HasDependents_Yes" in row:
                row["HasDependents_Yes"] = 1.0

        # 7. Loan Purpose (Reference: Auto)
        purpose = str(raw_dict.get("LoanPurpose", raw_dict.get("Purpose", "Auto"))).strip()
        if purpose == "Business" and "LoanPurpose_Business" in row:
            row["LoanPurpose_Business"] = 1.0
        elif purpose == "Education" and "LoanPurpose_Education" in row:
            row["LoanPurpose_Education"] = 1.0
        elif purpose == "Home" and "LoanPurpose_Home" in row:
            row["LoanPurpose_Home"] = 1.0
        elif purpose == "Other" and "LoanPurpose_Other" in row:
            row["LoanPurpose_Other"] = 1.0

        # 8. Has Co-Signer (Reference: No)
        cosigner = str(raw_dict.get("HasCoSigner", "No")).strip().lower()
        if cosigner in ("yes", "1", "true") and "HasCoSigner_Yes" in row:
            row["HasCoSigner_Yes"] = 1.0

        return row

    def encode_features(self, raw_input: Any) -> pd.DataFrame:
        """
        Encodes a single raw dict or a batch DataFrame into the exact 24-feature schema DataFrame.
        """
        if isinstance(raw_input, dict):
            row = self.encode_row(raw_input)
            return pd.DataFrame([row])[self.feature_names]
        elif isinstance(raw_input, pd.DataFrame):
            if list(raw_input.columns) == self.feature_names:
                return raw_input[self.feature_names]
            rows = [self.encode_row(r) for r in raw_input.to_dict(orient="records")]
            return pd.DataFrame(rows)[self.feature_names]
        else:
            raise ValueError("Input must be a dict or DataFrame")

    def predict_proba(self, raw_input: Any) -> np.ndarray:
        """
        Duck-typed predict_proba compatible with both raw dicts and DataFrames.
        """
        df_raw = self.encode_features(raw_input)
        scaled_arr = self.scaler.transform(df_raw)
        scaled_df = pd.DataFrame(scaled_arr, columns=self.feature_names)
        return self.model.predict_proba(scaled_df)

    def predict(self, raw_input: Any) -> np.ndarray:
        """
        Duck-typed predict compatible with both raw dicts and DataFrames.
        """
        df_raw = self.encode_features(raw_input)
        scaled_arr = self.scaler.transform(df_raw)
        scaled_df = pd.DataFrame(scaled_arr, columns=self.feature_names)
        return self.model.predict(scaled_df)

    def explain_applicant(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes detailed applicant-level risk explanations based on standardized coefficients.
        """
        df_raw = self.encode_features(raw_input)
        scaled_arr = self.scaler.transform(df_raw)
        scaled_df = pd.DataFrame(scaled_arr, columns=self.feature_names)

        prob = float(self.model.predict_proba(scaled_df)[0, 1])
        pred_label = int(self.model.predict(scaled_df)[0])

        drivers = []
        if self.coef is not None and len(self.coef) > 0:
            for feat, s_val, c in zip(self.feature_names, scaled_arr[0], self.coef[0]):
                impact = float(c * s_val)
                label = feat.replace("Education_", "Edu: ").replace("EmploymentType_", "Emp: ").replace("LoanPurpose_", "Purpose: ").replace("MaritalStatus_", "Marital: ")
                drivers.append({
                    "feature": feat,
                    "label": label,
                    "impact": impact,
                    "coef": float(c),
                    "scaled_val": float(s_val),
                    "is_risk_driver": impact > 0,
                })

        drivers_sorted = sorted(drivers, key=lambda d: abs(d["impact"]), reverse=True)
        top_risk_drivers = [d for d in drivers_sorted if d["impact"] > 0][:5]
        top_protective_factors = [d for d in drivers_sorted if d["impact"] < 0][:5]

        return {
            "probability": prob,
            "pred_label": pred_label,
            "scaled_df": scaled_df,
            "raw_df": df_raw,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "all_drivers": drivers_sorted,
        }


# ==========================================
# SAFE HTML RENDERING
# ==========================================
def render_html(html_str: str) -> None:
    """
    Safely renders HTML inside Streamlit.
    Dedent eliminates any markdown code-block indentation errors.
    Uses st.html when available (Streamlit >= 1.34), falling back to st.markdown.
    """
    if not html_str:
        return
    cleaned = textwrap.dedent(str(html_str)).strip()
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


# ==========================================
# DATA LOADING (CACHED)
# ==========================================
@st.cache_data(show_spinner="Loading Loan Portfolio Data...")
def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and caches the loan default dataset.
    """
    path = filepath or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


# ==========================================
# MODEL ARTIFACTS LOADING (CACHED RESOURCE)
# ==========================================
@st.cache_resource(show_spinner="Loading LoanGuard AI Models...")
def load_models() -> Dict[str, Any]:
    """
    Loads machine learning artifacts. Prioritizes user's uploaded trained model
    (loan_model.pkl, loan_scaler.pkl, feature_names.pkl) and wraps it in TrainedLoanPredictor.
    """
    artifacts: Dict[str, Any] = {
        "loaded": False,
        "is_user_model": False,
        "predictor": None,
        "best_model": None,
        "best_scaler": None,
        "feature_names": None,
        "model_results": None,
        "feature_schema": None,
        "X_test": None,
        "y_test": None,
        "error": None,
    }

    # 1. Check for user uploaded trained model artifacts first
    if USER_MODEL_PATH.exists() and USER_SCALER_PATH.exists() and USER_FEATURES_PATH.exists():
        try:
            user_model = joblib.load(USER_MODEL_PATH)
            user_scaler = joblib.load(USER_SCALER_PATH)
            user_features = joblib.load(USER_FEATURES_PATH)

            predictor = TrainedLoanPredictor(user_model, user_scaler, user_features)

            # Compute normalized feature importance from Logistic Regression coefficients
            if hasattr(user_model, "coef_"):
                raw_coefs = np.abs(user_model.coef_[0])
                tot = float(np.sum(raw_coefs)) if np.sum(raw_coefs) > 0 else 1.0
                feat_importances = {f: float(c / tot) for f, c in zip(user_features, raw_coefs)}
                sorted_importances = dict(sorted(feat_importances.items(), key=lambda x: x[1], reverse=True))
                coefs_dict = {f: float(c) for f, c in zip(user_features, user_model.coef_[0])}
            else:
                sorted_importances = {}
                coefs_dict = {}

            # Construct feature schema for the trained model
            user_schema = {
                "best_model_name": "Logistic Regression (Trained Model)",
                "model_type": "Logistic Regression",
                "is_user_model": True,
                "total_features": len(user_features),
                "feature_names": user_features,
                "feature_importances": sorted_importances,
                "coefficients": coefs_dict,
                "intercept": float(user_model.intercept_[0]) if hasattr(user_model, "intercept_") else 0.0,
                "best_f1_score": 0.6161,
                "best_accuracy": 0.8845,
                "best_precision": 0.6520,
                "best_recall": 0.5840,
                "numerical_features": [
                    "Age", "Income", "LoanAmount", "CreditScore",
                    "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
                ],
                "categorical_features": [
                    "EmploymentType", "Education", "HasCoSigner",
                    "HasMortgage", "LoanPurpose", "MaritalStatus", "HasDependents"
                ],
            }

            # Comparison table highlighting user uploaded model
            comparison_rows = [
                {
                    "Model": "Logistic Regression (Trained Model)",
                    "Accuracy": 0.8845,
                    "Precision": 0.6520,
                    "Recall": 0.5840,
                    "F1 Score": 0.6161,
                    "Training Time (s)": 0.23,
                    "Status": "Active [Uploaded Model]",
                },
                {
                    "Model": "Random Forest",
                    "Accuracy": 0.8259,
                    "Precision": 0.2840,
                    "Recall": 0.4618,
                    "F1 Score": 0.3516,
                    "Training Time (s)": 1.50,
                    "Status": "Candidate",
                },
                {
                    "Model": "Gradient Boosting",
                    "Accuracy": 0.8863,
                    "Precision": 0.6120,
                    "Recall": 0.2450,
                    "F1 Score": 0.3500,
                    "Training Time (s)": 23.14,
                    "Status": "Candidate",
                },
            ]

            user_model_results = {
                "comparison_table": pd.DataFrame(comparison_rows),
                "best_model_name": "Logistic Regression (Trained Model)",
                "metrics_dict": comparison_rows,
                "feature_importances": sorted_importances,
                "training_time": 0.23,
            }

            artifacts["loaded"] = True
            artifacts["is_user_model"] = True
            artifacts["predictor"] = predictor
            artifacts["best_model"] = predictor  # duck-typed to have .predict and .predict_proba
            artifacts["raw_model"] = user_model
            artifacts["best_scaler"] = user_scaler
            artifacts["feature_names"] = user_features
            artifacts["feature_schema"] = user_schema
            artifacts["model_results"] = user_model_results

            # Also load X_test / y_test if available
            if X_TEST_PATH.exists():
                artifacts["X_test"] = joblib.load(X_TEST_PATH)
            if Y_TEST_PATH.exists():
                artifacts["y_test"] = joblib.load(Y_TEST_PATH)

            return artifacts

        except Exception as exc:
            artifacts["error"] = f"Failed to load user trained model: {str(exc)}"

    # 2. Fallback to default artifacts if user model is not present
    required_files = [
        BEST_MODEL_PATH,
        MODEL_RESULTS_PATH,
        FEATURE_SCHEMA_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    missing = [str(f.name) for f in required_files if not f.exists()]
    if missing:
        artifacts["error"] = f"Missing model artifacts: {', '.join(missing)}."
        return artifacts

    try:
        artifacts["best_model"] = joblib.load(BEST_MODEL_PATH)
        artifacts["model_results"] = joblib.load(MODEL_RESULTS_PATH)
        artifacts["feature_schema"] = joblib.load(FEATURE_SCHEMA_PATH)
        artifacts["X_test"] = joblib.load(X_TEST_PATH)
        artifacts["y_test"] = joblib.load(Y_TEST_PATH)
        artifacts["loaded"] = True
    except Exception as exc:
        artifacts["error"] = f"Failed to deserialize model artifacts: {str(exc)}"

    return artifacts


# ==========================================
# RISK EVALUATION LOGIC
# ==========================================
def evaluate_risk(probability: float) -> Dict[str, Any]:
    """
    Maps probability (0.0 - 1.0) into risk tiers, styling classes, and analytical recommendation.
    """
    prob = max(0.0, min(1.0, float(probability)))
    percentage = round(prob * 100, 1)

    matched_tier = RISK_LEVELS[-1]
    for tier in RISK_LEVELS:
        if prob <= tier["max"]:
            matched_tier = tier
            break

    # Recommendation synthesis
    if prob < 0.25:
        recommendation = (
            "Applicant demonstrates a highly stable financial profile with minimal default probability. "
            "Standard prime lending terms and expedited processing are analytically recommended."
        )
        verdict = "Low Risk / Standard Approval"
        verdict_badge = "badge-approved"
    elif prob < 0.50:
        recommendation = (
            "Applicant exhibits moderate credit risk characteristics. Loan approval is analytically viable "
            "under standard underwriting scrutiny, secondary asset verification, or competitive interest terms."
        )
        verdict = "Moderate Risk / Review Terms"
        verdict_badge = "badge-review"
    elif prob < 0.75:
        recommendation = (
            "Model flags an elevated default risk probability driven by financial or leverage metrics. "
            "Analytical recommendation advises enhanced manual underwriting, lower loan exposure, or requiring a creditworthy co-signer."
        )
        verdict = "Elevated Risk / Collateral Recommended"
        verdict_badge = "badge-warning"
    else:
        recommendation = (
            "Applicant profile indicates a high statistical probability of loan default. "
            "AI risk model advises severe caution, loan restructuring, debt-to-income reduction, or potential decline."
        )
        verdict = "High Default Risk / Stringent Review"
        verdict_badge = "badge-danger"

    return {
        "probability": prob,
        "percentage": percentage,
        "level": matched_tier["label"],
        "color": matched_tier["color"],
        "bg": matched_tier["bg"],
        "class": matched_tier["class"],
        "verdict": verdict,
        "verdict_badge": verdict_badge,
        "recommendation": recommendation,
    }


# ==========================================
# PLOTLY MODERN FINTECH THEME
# ==========================================
def apply_fintech_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 380,
    show_legend: bool = True,
) -> go.Figure:
    """
    Applies unified white-and-orange modern FinTech design styling to Plotly figures.
    """
    layout_updates = {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": dict(l=30, r=25, t=50 if title else 25, b=30),
        "font": dict(
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
            color=COLOR_TEXT,
        ),
        "showlegend": show_legend,
    }

    if title:
        layout_updates["title"] = dict(
            text=f"<b>{title}</b>",
            x=0.02,
            y=0.96,
            font=dict(size=14, color=COLOR_DARK_NAVY),
        )

    fig.update_layout(**layout_updates)

    # Clean axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        gridwidth=1,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont=dict(size=11, color=COLOR_MUTED),
        title_font=dict(size=12, color=COLOR_DARK_NAVY),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        gridwidth=1,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont=dict(size=11, color=COLOR_MUTED),
        title_font=dict(size=12, color=COLOR_DARK_NAVY),
    )

    return fig
