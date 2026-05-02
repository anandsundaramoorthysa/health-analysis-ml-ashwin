"""ML pipeline: trains and persists the 3 best models for the workhealth app.

Selection rationale — one best-in-class model per prediction task:

1. **XGBoost Classifier**          -> RiskCategory (Low / Moderate / High)
   * Multi-class accuracy 0.93, beats Random Forest / Decision Tree.
   * Feature-importance source for the dashboard.

2. **Gradient Boosting Regressor** -> StressScore (continuous, 0-40)
   * R-squared 0.75, beats Ridge / Lasso baselines.

3. **Logistic Regression**         -> HighRiskFlag (binary, urgent action)
   * Accuracy 0.99, lowest memory footprint, fastest inference.

All three are sized for Streamlit Community Cloud's 1 GB RAM cap. Pre-trained
artefacts are persisted (compressed joblib + JSON reports + JSON learning
curves) so the deployed app loads instead of re-training at request time.
"""
from __future__ import annotations

import gc
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import (cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "WorkHours", "ScreenTime", "SleepHours", "ActivityMinutes",
    "CaffeineCups", "WaterIntakeL", "BreaksPerDay", "BMI",
    "JobSatisfaction", "BurnoutLevel", "Age",
]
USER_FEATURES_7 = [
    "WorkHours", "SleepHours", "CaffeineCups", "ActivityMinutes",
    "ScreenTime", "WaterIntakeL", "JobSatisfaction",
]


@dataclass
class ModelReport:
    name: str
    target: str
    kind: str          # "classifier" | "regressor"
    train_score: float
    test_score: float
    cv_mean: float
    cv_std: float
    gap: float
    status: str
    extra: dict


class XGBWrap:
    """Wraps XGBoost so its `.predict()` returns string labels (Low/Moderate/High)."""
    def __init__(self, m, rev):
        self.m, self.rev = m, rev
        self.classes_ = np.array([rev[i] for i in sorted(rev)])

    def predict(self, X):
        return np.array([self.rev[int(p)] for p in self.m.predict(X)])

    def predict_proba(self, X):
        return self.m.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.m.feature_importances_


def _fit_status(train: float, test: float) -> str:
    gap = train - test
    if gap > 0.10:
        return "Overfitting"
    if test < 0.50:
        return "Underfitting"
    return "Good Fit"


def _train_classifier(model, X_tr, y_tr, X_te, y_te, name, target):
    model.fit(X_tr, y_tr)
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc = accuracy_score(y_te, model.predict(X_te))
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=1)
    return ModelReport(
        name=name, target=target, kind="classifier",
        train_score=round(float(train_acc), 4),
        test_score=round(float(test_acc), 4),
        cv_mean=round(float(cv.mean()), 4),
        cv_std=round(float(cv.std()), 4),
        gap=round(float(train_acc - test_acc), 4),
        status=_fit_status(train_acc, test_acc),
        extra={
            "confusion_matrix": confusion_matrix(y_te, model.predict(X_te)).tolist(),
            "labels": sorted(np.unique(y_te).tolist()),
        },
    )


def _train_regressor(model, X_tr, y_tr, X_te, y_te, name, target):
    model.fit(X_tr, y_tr)
    train_r2 = r2_score(y_tr, model.predict(X_tr))
    test_r2 = r2_score(y_te, model.predict(X_te))
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=1)
    return ModelReport(
        name=name, target=target, kind="regressor",
        train_score=round(float(train_r2), 4),
        test_score=round(float(test_r2), 4),
        cv_mean=round(float(cv.mean()), 4),
        cv_std=round(float(cv.std()), 4),
        gap=round(float(train_r2 - test_r2), 4),
        status=_fit_status(train_r2, test_r2),
        extra={"mae": round(float(mean_absolute_error(y_te, model.predict(X_te))), 3)},
    )


def _compute_learning_curves(models: dict, df: pd.DataFrame) -> dict:
    """Compute learning curves once at train time so the dashboard never
    has to re-fit live (huge memory saver)."""
    sample = df.sample(min(2000, len(df)), random_state=42)
    X = sample[FEATURES]
    out: dict = {}
    for name, model in models.items():
        try:
            kind = "classifier"
            target_col = "RiskCategory"
            if name == "Gradient Boosting":
                target_col, kind = "StressScore", "regressor"
            elif name == "Logistic Regression":
                target_col = "HighRiskFlag"
            scoring = "accuracy" if kind == "classifier" else "r2"
            sizes, train_scores, val_scores = learning_curve(
                model, X, sample[target_col], cv=3, scoring=scoring, n_jobs=1,
                train_sizes=np.linspace(0.2, 1.0, 5))
            out[name] = {
                "sizes": sizes.tolist(),
                "train_mean": train_scores.mean(axis=1).tolist(),
                "val_mean": val_scores.mean(axis=1).tolist(),
                "kind": kind,
            }
        except Exception as e:  # pragma: no cover
            out[name] = {"error": str(e)}
        gc.collect()
    return out


def train_all(df: pd.DataFrame, save: bool = True) -> dict[str, Any]:
    """Train the 3 selected models, return models + reports + importances."""
    X = df[FEATURES].copy()
    y_cls = df["RiskCategory"]
    y_bin = df["HighRiskFlag"]
    y_reg = df["StressScore"]

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X, y_bin, test_size=0.2, stratify=y_bin, random_state=42)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)

    reports: dict[str, ModelReport] = {}
    models: dict[str, Any] = {}

    # ---------- 1. XGBoost (RiskCategory) ----------
    try:
        from xgboost import XGBClassifier
        labels_sorted = sorted(y_cls.unique().tolist())
        label_map = {lab: i for i, lab in enumerate(labels_sorted)}
        rev_map = {i: lab for lab, i in label_map.items()}
        xgb_pipe = XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="mlogloss", random_state=42, n_jobs=1,
            tree_method="hist",
        )
        yc_tr_enc = yc_tr.map(label_map)
        xgb_pipe.fit(Xc_tr, yc_tr_enc)
        wrapped = XGBWrap(xgb_pipe, rev_map)
        train_acc = accuracy_score(yc_tr, wrapped.predict(Xc_tr))
        test_acc = accuracy_score(yc_te, wrapped.predict(Xc_te))
        # Cheaper 3-fold CV (memory)
        cv = cross_val_score(
            XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.1,
                eval_metric="mlogloss", random_state=42, n_jobs=1,
                tree_method="hist"),
            Xc_tr, yc_tr_enc, cv=3, scoring="accuracy", n_jobs=1)
        reports["XGBoost"] = ModelReport(
            "XGBoost", "RiskCategory", "classifier",
            round(float(train_acc), 4), round(float(test_acc), 4),
            round(float(cv.mean()), 4), round(float(cv.std()), 4),
            round(float(train_acc - test_acc), 4),
            _fit_status(train_acc, test_acc),
            {"confusion_matrix": confusion_matrix(yc_te, wrapped.predict(Xc_te)).tolist(),
             "labels": labels_sorted})
        models["XGBoost"] = wrapped
    except Exception as e:
        reports["XGBoost"] = ModelReport(
            "XGBoost", "RiskCategory", "classifier",
            0, 0, 0, 0, 0, f"unavailable: {e}", {})

    # ---------- 2. Gradient Boosting (StressScore) ----------
    gb = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
    reports["Gradient Boosting"] = _train_regressor(
        gb, Xr_tr, yr_tr, Xr_te, yr_te, "Gradient Boosting", "StressScore")
    models["Gradient Boosting"] = gb

    # ---------- 3. Logistic Regression (HighRiskFlag) ----------
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    reports["Logistic Regression"] = _train_classifier(
        lr, Xb_tr, yb_tr, Xb_te, yb_te, "Logistic Regression", "HighRiskFlag")
    models["Logistic Regression"] = lr

    # Feature importance from XGBoost (the primary classifier).
    xgb_model = models.get("XGBoost")
    if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
        importances = dict(zip(
            FEATURES, [float(v) for v in xgb_model.feature_importances_]))
    else:
        importances = {f: 0.0 for f in FEATURES}

    # Pre-compute learning curves once.
    learning_curves = _compute_learning_curves(models, df)

    if save:
        joblib.dump(models, MODEL_DIR / "models.joblib", compress=3)
        with open(MODEL_DIR / "reports.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in reports.items()}, f, indent=2)
        joblib.dump(FEATURES, MODEL_DIR / "features.joblib", compress=3)
        with open(MODEL_DIR / "importances.json", "w", encoding="utf-8") as f:
            json.dump(importances, f, indent=2)
        with open(MODEL_DIR / "learning_curves.json", "w", encoding="utf-8") as f:
            json.dump(learning_curves, f)

    gc.collect()
    return {"models": models, "reports": reports,
            "importances": importances, "features": FEATURES,
            "learning_curves": learning_curves}


@st.cache_resource(show_spinner=False)
def load_or_train(df: pd.DataFrame) -> dict[str, Any]:
    """Load pre-trained models from disk if present; train + persist otherwise."""
    model_path = MODEL_DIR / "models.joblib"
    report_path = MODEL_DIR / "reports.json"
    if model_path.exists() and report_path.exists():
        models = joblib.load(model_path)
        with open(report_path, "r", encoding="utf-8") as f:
            reports_raw = json.load(f)
        reports = {k: ModelReport(**v) for k, v in reports_raw.items()}
        with open(MODEL_DIR / "importances.json", "r", encoding="utf-8") as f:
            importances = json.load(f)
        learning_curves = {}
        lc_path = MODEL_DIR / "learning_curves.json"
        if lc_path.exists():
            with open(lc_path, "r", encoding="utf-8") as f:
                learning_curves = json.load(f)
        return {"models": models, "reports": reports,
                "importances": importances, "features": FEATURES,
                "learning_curves": learning_curves}
    return train_all(df, save=True)


def predict_for_user(models: dict, user_row: pd.DataFrame) -> dict:
    """Run all 3 models on a one-row user dataframe."""
    X = user_row[FEATURES]
    out: dict = {}

    # XGBoost — primary multi-class risk classifier
    xgb = models.get("XGBoost")
    if xgb is not None and hasattr(xgb, "predict"):
        try:
            out["RiskCategory"] = str(xgb.predict(X)[0])
            out["RiskProba"] = dict(
                zip(xgb.classes_,
                    [round(float(p), 3) for p in xgb.predict_proba(X)[0]]))
        except Exception:
            pass

    # Gradient Boosting — continuous stress score
    gb = models.get("Gradient Boosting")
    if gb is not None:
        out["StressPrediction"] = round(float(gb.predict(X)[0]), 2)

    # Logistic Regression — binary high-risk flag
    lr = models.get("Logistic Regression")
    if lr is not None:
        out["HighRiskProb"] = round(float(lr.predict_proba(X)[0][1]), 3)

    return out
