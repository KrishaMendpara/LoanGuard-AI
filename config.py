"""
LoanGuard AI - Configuration & Constants
Intelligent Credit Risk & Loan Default Prediction
"""

from pathlib import Path

# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "loan_default.csv"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_RESULTS_PATH = MODELS_DIR / "model_results.pkl"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.pkl"
X_TEST_PATH = MODELS_DIR / "X_test.pkl"
Y_TEST_PATH = MODELS_DIR / "y_test.pkl"

# Uploaded Model Artifacts (Root workspace or models dir)
WORKSPACE_DIR = BASE_DIR.parent
USER_MODEL_PATH = (MODELS_DIR / "loan_model.pkl") if (MODELS_DIR / "loan_model.pkl").exists() else (WORKSPACE_DIR / "loan_model.pkl")
USER_SCALER_PATH = (MODELS_DIR / "loan_scaler.pkl") if (MODELS_DIR / "loan_scaler.pkl").exists() else (WORKSPACE_DIR / "loan_scaler.pkl")
USER_FEATURES_PATH = (MODELS_DIR / "feature_names.pkl") if (MODELS_DIR / "feature_names.pkl").exists() else (WORKSPACE_DIR / "feature_names.pkl")

ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "style.css"
JS_PATH = ASSETS_DIR / "app.js"
IMAGES_DIR = ASSETS_DIR / "images"

# ==========================================
# BRAND & APPLICATION METADATA
# ==========================================
PROJECT_NAME = "LoanGuard AI"
PROJECT_TAGLINE = "Intelligent Credit Risk & Loan Default Prediction"
APP_ICON = "🛡️"
VERSION = "2.4.0"

# ==========================================
# DESIGN THEME (ORANGE + WHITE FINTECH)
# ==========================================
COLOR_PRIMARY = "#FF6B00"
COLOR_SECONDARY = "#FF8A3D"
COLOR_LIGHT_ORANGE = "#FFF3E8"
COLOR_DARK = "#111827"
COLOR_DARK_NAVY = "#172033"
COLOR_WHITE = "#FFFFFF"
COLOR_BACKGROUND = "#F8FAFC"
COLOR_BORDER = "#E5E7EB"
COLOR_TEXT = "#1F2937"
COLOR_MUTED = "#6B7280"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"

# ==========================================
# MACHINE LEARNING SCHEMA
# ==========================================
TARGET_COLUMN = "loan_default"

NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumDependents",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "EmploymentType",
    "Education",
    "HasCoSigner",
    "HasMortgage",
    "Purpose",
    "MaritalStatus",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Form option lists
EDUCATION_CHOICES = ["Bachelor's", "Master's", "High School", "PhD"]
EMPLOYMENT_CHOICES = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
MARITAL_CHOICES = ["Single", "Married", "Divorced"]
PURPOSE_CHOICES = ["Auto", "Business", "Education", "Home", "Other"]
YES_NO_CHOICES = ["Yes", "No"]

# Risk level boundaries
RISK_LEVELS = [
    {"max": 0.25, "label": "LOW RISK", "color": "#10B981", "bg": "#ECFDF5", "class": "risk-low"},
    {"max": 0.50, "label": "MODERATE RISK", "color": "#F59E0B", "bg": "#FFFBEB", "class": "risk-moderate"},
    {"max": 0.75, "label": "HIGH RISK", "color": "#FF6B00", "bg": "#FFF3E8", "class": "risk-high"},
    {"max": 1.00, "label": "VERY HIGH RISK", "color": "#EF4444", "bg": "#FEF2F2", "class": "risk-very-high"},
]
