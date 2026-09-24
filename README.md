# 🛡️ LoanGuard AI
### Intelligent Credit Risk & Loan Default Prediction Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-orange.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.34+-FF6B00.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-172033.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/plotly-5.18+-blue.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

**LoanGuard AI** is a state-of-the-art FinTech web application and credit analytics platform designed for commercial banks, credit unions, and digital lenders. It leverages supervised machine learning pipelines to analyze borrower demographic, employment, and financial metrics, predicting loan default probabilities and delivering transparent risk categorization.

Built with an **Orange & White FinTech design system**, modern typography (`Inter`), responsive Bootstrap layouts, interactive Plotly visualizations, and zero raw HTML artifacts, LoanGuard AI bridges the gap between complex ML modeling and executive decision-making.

---

## ✨ Key Features

1. **AI Risk Assessment Engine**:
   - Clean 4-card applicant evaluation form (Personal, Employment, Financial, Loan Profile).
   - Real-time default probability scoring (0.0% – 100.0%).
   - Dynamic Risk Tiers (`LOW RISK`, `MODERATE RISK`, `HIGH RISK`, `VERY HIGH RISK`).
   - Animated risk meter & actionable underwriting recommendations.
   - Genuine model feature importance calculation without fabricated data.

2. **Portfolio Analytics Dashboard**:
   - Real-time dynamic sidebar filtering by Education, Employment, Loan Purpose, Marital Status, Credit Score, and Income.
   - 6 Top KPI metric cards dynamically updating with cohort selections.
   - Data Quality & Integrity Benchmark cards.
   - **8 Interactive Plotly Visualizations**:
     - Loan Default Distribution (Donut chart with custom center badge)
     - Default Rate by Education Level (Bar chart)
     - Default Rate by Employment Type (Bar chart)
     - Credit Score Distribution (Overlay histogram)
     - Income vs. Loan Amount (High-performance sampled scatter plot)
     - Interest Rate vs. Default Probability (Trend curve)
     - Borrower Age Cohort vs. Default Risk (Distribution)
     - Financial Metrics Correlation Heatmap (Normalized matrix)

3. **Model Intelligence & Benchmarking**:
   - Head-to-head comparison of **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
   - Metric cards for Accuracy, Precision, Recall, and F1 Score.
   - Grouped interactive benchmark bar chart.
   - Architectural deep-dive cards explaining linear vs. ensemble mechanics.
   - Complete deployment and training metadata specifications.

4. **FinTech UI & Design System**:
   - Custom top sticky navbar with shield brand mark and quick CTA.
   - Synced sidebar navigation with live system status indicator (`● AI Models Ready`).
   - Unified CSS design tokens (`--primary-orange: #FF6B00`, `--dark-navy: #172033`).
   - Dedicated `render_html()` helper preventing raw HTML rendering issues.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Core Framework** | Python 3.10+ / Streamlit | Reactive web dashboard and server execution |
| **Machine Learning** | Scikit-Learn (Pipelines, Ensembles) | Preprocessing, classification, and metrics |
| **Data Processing** | Pandas, NumPy | High-performance tabular transformation |
| **Visualization** | Plotly Graph Objects & Express | Interactive FinTech charting engine |
| **Artifact Persistence**| Joblib | Fast serialization of models & schemas |
| **Styling & Icons** | Vanilla CSS3, Bootstrap Icons | Custom theme, typography, cards & shadows |
| **Micro-Interactions**| Vanilla JavaScript | Progress animations & smooth scroll behaviors |

---

## 📂 Project Structure

```
LoanGuardAI/
│
├── app.py                      # Main entry point & page router
├── config.py                   # Global constants, paths, colors & schemas
├── utils.py                    # Safe render_html, model/data loaders, theming
├── train_model.py              # ML training, benchmarking & artifact export
├── requirements.txt            # Production dependencies
├── README.md                   # System documentation
│
├── data/
│   └── loan_default.csv        # Standardized portfolio dataset (255k records)
│
├── models/
│   ├── best_model.pkl          # Exported Scikit-Learn pipeline
│   ├── model_results.pkl       # Benchmark results & metrics dictionary
│   ├── feature_schema.pkl      # Feature metadata & valid categories
│   ├── X_test.pkl              # Evaluation split features
│   └── y_test.pkl              # Evaluation split targets
│
├── assets/
│   ├── style.css               # Complete FinTech UI design system
│   ├── app.js                  # Micro-interactions & animations
│   └── images/                 # Brand assets & placeholders
│
└── views/
    ├── home.py                 # Landing page (Hero, Stats, Features, How it works, CTA)
    ├── analytics.py            # Portfolio analytics (In-page filters, 6 KPIs, 8 Charts)
    ├── models.py               # Model benchmarking & comparisons
    └── prediction.py           # 4-card risk assessment engine
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Repository
```bash
cd LoanGuardAI
```

### 2. Install Dependencies
Ensure you have Python 3.10 or higher installed. Install the certified requirements:
```bash
pip install -r requirements.txt
```

---

## 🧠 Training & Benchmarking Models

To train all three candidate algorithms (Logistic Regression, Random Forest, Gradient Boosting), evaluate them on out-of-sample data, and export the best pipeline:

```bash
python train_model.py
```

### Training Output Summary:
- Preprocesses 15 features with median imputation, standard scaling, and one-hot encoding.
- Evaluates models on a stratified 80/20 train/test split.
- Automatically selects the winner based on **F1 Score** to properly address default class imbalance.
- Exports `best_model.pkl`, `model_results.pkl`, `feature_schema.pkl`, `X_test.pkl`, and `y_test.pkl` to `models/`.

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will start locally at:
```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

---

## 📊 How the Prediction Pipeline Works

1. **User Input Collection**:
   The user inputs 15 attributes across 4 structured cards in `pages/prediction.py`:
   - **Numerical (9)**: `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumDependents`, `InterestRate`, `LoanTerm`, `DTIRatio`.
   - **Categorical (6)**: `EmploymentType`, `Education`, `HasCoSigner`, `HasMortgage`, `Purpose`, `MaritalStatus`.

2. **Pipeline Preprocessing**:
   The raw DataFrame is passed directly into `best_model.pkl`:
   ```python
   # Inside best_model.pkl (Scikit-Learn Pipeline)
   ColumnTransformer(
       transformers=[
           ('num', Pipeline([SimpleImputer(strategy='median'), StandardScaler()]), num_cols),
           ('cat', Pipeline([SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')]), cat_cols)
       ]
   )
   ```

3. **Inference & Probability Calibration**:
   The classifier outputs both binary class prediction (`0` for non-default, `1` for default) and predicted probabilities:
   ```python
   raw_prob = best_pipeline.predict_proba(input_df)[0, 1]
   ```

4. **Risk Tier Assignment & Recommendations**:
   - **0.0% – 25.0%**: `LOW RISK` (Standard prime approval terms)
   - **25.0% – 50.0%**: `MODERATE RISK` (Review terms / secondary checks)
   - **50.0% – 75.0%**: `HIGH RISK` (Elevated risk / collateral recommended)
   - **75.0% – 100.0%**: `VERY HIGH RISK` (Stringent review / potential decline)

5. **Explainable Risk Insights**:
   Feature importances are retrieved directly from the model and plotted dynamically to indicate the primary drivers behind the prediction.

---

## 🤖 Machine Learning Models Comparison

| Algorithm | Accuracy | Precision | Recall | F1 Score | Status |
|---|---|---|---|---|---|
| **Logistic Regression** | 67.22% | 21.45% | 68.48% | 0.3267 | Candidate (Linear Baseline) |
| **Random Forest** | **82.59%** | **30.63%** | **39.44%** | **0.3448** | 🏆 **Best Model (Selected)** |
| **Gradient Boosting** | 88.63% | 58.57% | 7.06% | 0.1260 | Candidate (High Accuracy) |

*Why Random Forest is selected*: In credit risk modeling where default classes are imbalanced (~11.6% defaults), raw accuracy is misleading. Random Forest delivers the highest harmonic balance (**F1 Score of 0.3448**) while maintaining an 82.6% accuracy.

---

## 📁 Dataset Details

- **Dataset File**: `data/loan_default.csv`
- **Total Records**: 255,347 borrower applications
- **Target Variable**: `loan_default` (0 = Non-Default, 1 = Default)
- **Baseline Default Rate**: 11.61%
- **Data Integrity**: 0 missing values, 0 duplicate records.

---

## 🔮 Future Improvements

- [ ] SHAP (SHapley Additive exPlanations) force plots for individual borrower attribution.
- [ ] Integration with credit bureau REST APIs (Experian, TransUnion).
- [ ] Automated PDF Credit Underwriting Memo generation.
- [ ] Multi-tenant authentication with role-based access control (Auditor, Underwriter, Executive).

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
