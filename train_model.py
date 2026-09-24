"""
LoanGuard AI - Machine Learning Pipeline Training Script
Trains, benchmarks, and exports models for credit risk & loan default prediction.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add parent to path for config imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_SCHEMA_PATH,
    MODEL_RESULTS_PATH,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    X_TEST_PATH,
    Y_TEST_PATH,
)


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def train_pipeline(max_sample_size: int = 75000) -> None:
    print("=" * 65)
    print(" [*] LOANGUARD AI - MODEL TRAINING & BENCHMARKING ENGINE")
    print("=" * 65)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    print(f"\n[1/5] Loading loan portfolio dataset from: {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Total records loaded: {len(df):,} rows x {df.shape[1]} columns")

    # Feature segregation
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    # Use stratified sampling if dataset exceeds max_sample_size for optimal training speed & reproducibility
    if len(df) > max_sample_size:
        print(f"      Selecting stratified benchmark sample of {max_sample_size:,} records...")
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=max_sample_size, random_state=42, stratify=y
        )
    else:
        X_sample, y_sample = X, y

    # Stratified 80/20 train/test split
    print(f"[2/5] Performing stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.20, random_state=42, stratify=y_sample
    )
    print(f"      Training set: {len(X_train):,} samples")
    print(f"      Test set:     {len(X_test):,} samples")
    print(f"      Default rate in train: {y_train.mean():.2%}")

    # Build preprocessing pipeline
    print(f"[3/5] Constructing Scikit-Learn ColumnTransformer pipeline...")
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # Models to benchmark
    models_to_train = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
    }

    print(f"\n[4/5] Training and benchmarking candidate algorithms...")
    benchmark_results = []
    trained_pipelines = {}

    for name, clf in models_to_train.items():
        t0 = time.time()
        print(f"      Training {name}...", end="", flush=True)

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        full_pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Evaluate on test set
        y_pred = full_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        trained_pipelines[name] = full_pipeline
        benchmark_results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
            "Training Time (s)": round(fit_time, 2),
        })

        print(f" Done ({fit_time:.2f}s) | Acc: {acc:.2%} | F1: {f1:.4f}")

    results_df = pd.DataFrame(benchmark_results)

    # Select best model based on F1 Score
    best_row = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n[+] Model Selection: '{best_model_name}' achieved highest F1 Score ({best_row['F1 Score']:.4f})")

    # Add Status column
    results_df["Status"] = results_df["Model"].apply(
        lambda m: "Best Model [Selected]" if m == best_model_name else "Candidate"
    )

    # Extract feature names & feature importance
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_classifier = best_pipeline.named_steps["classifier"]

    # Retrieve transformed feature names
    cat_ohe = fitted_preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_feature_names = list(cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_transformed_features = NUMERICAL_FEATURES + cat_feature_names

    feature_importances = {}
    if hasattr(fitted_classifier, "feature_importances_"):
        raw_importances = fitted_classifier.feature_importances_
        feature_importances = dict(zip(all_transformed_features, [float(v) for v in raw_importances]))
    elif hasattr(fitted_classifier, "coef_"):
        raw_coefs = np.abs(fitted_classifier.coef_[0])
        feature_importances = dict(zip(all_transformed_features, [float(v) for v in raw_coefs]))

    # Sort feature importance
    sorted_importances = dict(
        sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    )

    # Build feature schema metadata for prediction validation
    feature_schema = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "total_train_records": len(X_train),
        "total_test_records": len(X_test),
        "best_model_name": best_model_name,
        "best_f1_score": float(best_row["F1 Score"]),
        "best_accuracy": float(best_row["Accuracy"]),
        "best_precision": float(best_row["Precision"]),
        "best_recall": float(best_row["Recall"]),
        "feature_importances": sorted_importances,
        "transformed_feature_names": all_transformed_features,
        "numerical_medians": {col: float(df[col].median()) for col in NUMERICAL_FEATURES},
        "category_choices": {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES},
    }

    # Export artifacts
    print(f"\n[5/5] Exporting model artifacts to: {MODELS_DIR}...")
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"      -> {BEST_MODEL_PATH.name} (Complete Pipeline)")

    joblib.dump({
        "comparison_table": results_df,
        "best_model_name": best_model_name,
        "metrics_dict": benchmark_results,
        "feature_importances": sorted_importances,
        "training_time": sum(r["Training Time (s)"] for r in benchmark_results),
    }, MODEL_RESULTS_PATH)
    print(f"      -> {MODEL_RESULTS_PATH.name} (Benchmark metrics & comparison)")

    joblib.dump(feature_schema, FEATURE_SCHEMA_PATH)
    print(f"      -> {FEATURE_SCHEMA_PATH.name} (Feature schema & medians)")

    joblib.dump(X_test.head(1000), X_TEST_PATH)
    print(f"      -> {X_TEST_PATH.name} (Evaluation sample)")

    joblib.dump(y_test.head(1000), Y_TEST_PATH)
    print(f"      -> {Y_TEST_PATH.name} (Evaluation targets)")

    print("\n" + "=" * 65)
    print(" [OK] ALL MODEL ARTIFACTS SUCCESSFULLY GENERATED & VALIDATED")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train_pipeline()
