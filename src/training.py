"""Reproducible baseline comparison, calibration, and test evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, engineer_features

RANDOM_STATE = 42

def preprocessor(scale_numeric=False):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric: numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES), ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES)])

def candidate_models():
    return {"Ridge": Pipeline([("preprocessor", preprocessor(True)), ("model", Ridge(alpha=10.0))]), "GradientBoosting": Pipeline([("preprocessor", preprocessor()), ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=.04, max_depth=3, random_state=RANDOM_STATE, loss="huber"))]), "RandomForest": Pipeline([("preprocessor", preprocessor()), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=.8, random_state=RANDOM_STATE, n_jobs=-1))]), "ExtraTrees": Pipeline([("preprocessor", preprocessor()), ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=.9, random_state=RANDOM_STATE, n_jobs=-1))])}

def metrics(y_true_log, y_pred_log):
    actual, predicted = np.exp(y_true_log), np.exp(y_pred_log)
    return {"r2": float(r2_score(y_true_log, y_pred_log)), "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)), "rmse_log": float(mean_squared_error(y_true_log, y_pred_log) ** .5), "mae_inr": float(mean_absolute_error(actual, predicted)), "rmse_inr": float(mean_squared_error(actual, predicted) ** .5), "median_ae_inr": float(median_absolute_error(actual, predicted)), "mape_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100)}

def train_and_evaluate(data_path: Path, artifact_dir: Path, report_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path); X = engineer_features(raw); y = np.log(pd.to_numeric(raw["Price"], errors="raise").to_numpy())
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, random_state=RANDOM_STATE)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=.50, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE); comparison = []; models = candidate_models()
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}, n_jobs=1)
        comparison.append({"model": name, "cv_r2_mean": float(scores["test_r2"].mean()), "cv_r2_std": float(scores["test_r2"].std()), "cv_mae_log_mean": float(-scores["test_mae"].mean()), "cv_rmse_log_mean": float(-scores["test_rmse"].mean())})
    comparison.sort(key=lambda row: row["cv_rmse_log_mean"]); selected_name = comparison[0]["model"]; selected = models[selected_name]; selected.fit(X_train, y_train)
    residuals = np.abs(y_cal - selected.predict(X_cal)); level = min(1.0, np.ceil((len(residuals) + 1) * .90) / len(residuals)); interval_q = float(np.quantile(residuals, level, method="higher"))
    test_prediction = selected.predict(X_test); test_metrics = metrics(y_test, test_prediction); lower, upper, actual = np.exp(test_prediction - interval_q), np.exp(test_prediction + interval_q), np.exp(y_test)
    test_metrics.update({"interval_nominal_coverage": .90, "interval_empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))), "interval_log_half_width": interval_q})
    metadata = {"dataset_rows": int(len(raw)), "target": "Price", "native_currency": "INR", "target_transform": "natural logarithm", "random_state": RANDOM_STATE, "split": {"train": len(X_train), "calibration": len(X_cal), "test": len(X_test)}, "features": MODEL_FEATURES, "selected_model": selected_name, "model_comparison": comparison, "test_metrics": test_metrics}
    options = {feature: sorted(X[feature].dropna().astype(str).unique().tolist()) for feature in CATEGORICAL_FEATURES}
    joblib.dump(selected, artifact_dir / "model.joblib"); joblib.dump(options, artifact_dir / "options.joblib")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame({"actual_inr": actual, "predicted_inr": np.exp(test_prediction), "lower_90_inr": lower, "upper_90_inr": upper}).to_csv(report_dir / "test_predictions.csv", index=False)
    (report_dir / "evaluation_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

