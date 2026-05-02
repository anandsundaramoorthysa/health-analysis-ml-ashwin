"""ML pipeline: trains and persists 8 supervised models on the workhealth dataset.

Models
------
1. Random Forest Classifier   -> RiskCategory (Low/Moderate/High)
2. XGBoost Classifier         -> RiskCategory
3. Gradient Boosting Regressor-> StressScore
4. Logistic Regression        -> HighRiskFlag (binary)
5. Ridge Regression           -> StressScore
6. Lasso Regression           -> StressScore
7. KNN Classifier             -> RiskCategory (similarity baseline)
8. Decision Tree Classifier   -> RiskCategory (interpretable baseline)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             r2_score)
from sklearn.model_selection import (KFold, cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def _fit_status(train: float, test: float) -> str:
    gap = train - test
    if gap > 0.10:
        return "Overfitting"
    if test < 0.50:
        return "Underfitting"
    return "Good Fit"


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


def _train_classifier(model, X_tr, y_tr, X_te, y_te, name, target):
    model.fit(X_tr, y_tr)
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc = accuracy_score(y_te, model.predict(X_te))
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1)
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
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
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


def train_all(df: pd.DataFrame, save: bool = True) -> dict[str, Any]:
    """Train all 8 models, return models + reports + feature importances."""
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

    # 1. Random Forest (regularised to avoid overfitting per spec)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1)
    reports["Random Forest"] = _train_classifier(
        rf, Xc_tr, yc_tr, Xc_te, yc_te, "Random Forest", "RiskCategory")
    models["Random Forest"] = rf

    # 2. XGBoost
    try:
        from xgboost import XGBClassifier
        labels_sorted = sorted(y_cls.unique().tolist())
        label_map = {lab: i for i, lab in enumerate(labels_sorted)}
        rev_map = {i: lab for lab, i in label_map.items()}
        xgb_pipe = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
            tree_method="hist",
        )
        yc_tr_enc = yc_tr.map(label_map)
        yc_te_enc = yc_te.map(label_map)
        xgb_pipe.fit(Xc_tr, yc_tr_enc)
        wrapped = XGBWrap(xgb_pipe, rev_map)
        train_acc = accuracy_score(yc_tr, wrapped.predict(Xc_tr))
        test_acc = accuracy_score(yc_te, wrapped.predict(Xc_te))
        cv = cross_val_score(XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
                tree_method="hist"),
            Xc_tr, yc_tr_enc, cv=5, scoring="accuracy", n_jobs=-1)
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
        # XGBoost optional fallback
        reports["XGBoost"] = ModelReport(
            "XGBoost", "RiskCategory", "classifier",
            0, 0, 0, 0, 0, f"unavailable: {e}", {})

    # 3. Gradient Boosting Regressor (Stress)
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    reports["Gradient Boosting"] = _train_regressor(
        gb, Xr_tr, yr_tr, Xr_te, yr_te, "Gradient Boosting", "StressScore")
    models["Gradient Boosting"] = gb

    # 4. Logistic Regression (binary high risk flag)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    reports["Logistic Regression"] = _train_classifier(
        lr, Xb_tr, yb_tr, Xb_te, yb_te, "Logistic Regression", "HighRiskFlag")
    models["Logistic Regression"] = lr

    # 5. Ridge / 6. Lasso (Stress)
    ridge = Pipeline([("scaler", StandardScaler()),
                      ("reg", Ridge(alpha=1.0, random_state=42))])
    reports["Ridge Regression"] = _train_regressor(
        ridge, Xr_tr, yr_tr, Xr_te, yr_te, "Ridge Regression", "StressScore")
    models["Ridge Regression"] = ridge

    lasso = Pipeline([("scaler", StandardScaler()),
                      ("reg", Lasso(alpha=0.1, random_state=42, max_iter=5000))])
    reports["Lasso Regression"] = _train_regressor(
        lasso, Xr_tr, yr_tr, Xr_te, yr_te, "Lasso Regression", "StressScore")
    models["Lasso Regression"] = lasso

    # 7. KNN
    knn = Pipeline([("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15))])
    reports["KNN"] = _train_classifier(
        knn, Xc_tr, yc_tr, Xc_te, yc_te, "KNN", "RiskCategory")
    models["KNN"] = knn

    # 8. Decision Tree (interpretable)
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
    reports["Decision Tree"] = _train_classifier(
        dt, Xc_tr, yc_tr, Xc_te, yc_te, "Decision Tree", "RiskCategory")
    models["Decision Tree"] = dt

    # Feature importances from the RF (interpretable, regularised)
    importances = dict(zip(FEATURES, rf.feature_importances_.tolist()))

    if save:
        joblib.dump(models, MODEL_DIR / "models.joblib")
        with open(MODEL_DIR / "reports.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in reports.items()}, f, indent=2)
        joblib.dump(FEATURES, MODEL_DIR / "features.joblib")
        with open(MODEL_DIR / "importances.json", "w", encoding="utf-8") as f:
            json.dump(importances, f, indent=2)

    return {"models": models, "reports": reports,
            "importances": importances, "features": FEATURES,
            "splits": {"X_test_cls": Xc_te, "y_test_cls": yc_te,
                       "X_test_reg": Xr_te, "y_test_reg": yr_te}}


@st.cache_resource(show_spinner=False)
def load_or_train(df: pd.DataFrame) -> dict[str, Any]:
    model_path = MODEL_DIR / "models.joblib"
    report_path = MODEL_DIR / "reports.json"
    if model_path.exists() and report_path.exists():
        models = joblib.load(model_path)
        with open(report_path, "r", encoding="utf-8") as f:
            reports_raw = json.load(f)
        reports = {k: ModelReport(**v) for k, v in reports_raw.items()}
        with open(MODEL_DIR / "importances.json", "r", encoding="utf-8") as f:
            importances = json.load(f)
        return {"models": models, "reports": reports,
                "importances": importances, "features": FEATURES}
    return train_all(df, save=True)


def predict_for_user(models: dict, user_row: pd.DataFrame) -> dict:
    """Run all relevant models on a one-row user dataframe."""
    X = user_row[FEATURES]
    out = {}
    rf = models.get("Random Forest")
    if rf is not None:
        proba = rf.predict_proba(X)[0]
        out["RiskCategory"] = rf.predict(X)[0]
        out["RiskProba"] = dict(zip(rf.classes_, [round(float(p), 3) for p in proba]))

    xgb = models.get("XGBoost")
    if xgb is not None and hasattr(xgb, "predict"):
        try:
            out["RiskCategory_XGB"] = xgb.predict(X)[0]
            out["RiskProba_XGB"] = dict(
                zip(xgb.classes_, [round(float(p), 3) for p in xgb.predict_proba(X)[0]]))
        except Exception:
            pass

    gb = models.get("Gradient Boosting")
    if gb is not None:
        out["StressPrediction"] = round(float(gb.predict(X)[0]), 2)

    lr = models.get("Logistic Regression")
    if lr is not None:
        out["HighRiskProb"] = round(float(lr.predict_proba(X)[0][1]), 3)

    ridge = models.get("Ridge Regression")
    if ridge is not None:
        out["StressPrediction_Ridge"] = round(float(ridge.predict(X)[0]), 2)
    return out


def learning_curve_data(model, X, y, kind: str = "classifier"):
    """Return train/val curves for the Model Fitness page."""
    scoring = "accuracy" if kind == "classifier" else "r2"
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6))
    return {
        "sizes": sizes.tolist(),
        "train_mean": train_scores.mean(axis=1).tolist(),
        "val_mean": val_scores.mean(axis=1).tolist(),
        "train_std": train_scores.std(axis=1).tolist(),
        "val_std": val_scores.std(axis=1).tolist(),
    }
