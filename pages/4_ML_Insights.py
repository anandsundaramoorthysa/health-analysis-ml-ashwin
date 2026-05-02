"""ML Insights - feature importances, predictions, model comparison."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, auc

from utils.data_loader import load_data
from utils.ml_models import FEATURES, load_or_train
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="ML Insights",
                   page_icon=":material/model_training:", layout="wide")
inject_css()

st.title(":material/model_training: ML Insights")
st.caption("8 models | feature importance | actual vs predicted | ROC / confusion matrix.")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

artefacts = load_or_train(df)
models = artefacts["models"]
reports = artefacts["reports"]
imps = artefacts["importances"]

# ---------- Reports table ----------
section("Model Performance Summary", icon="leaderboard")
report_rows = []
for name, rep in reports.items():
    report_rows.append({
        "Model": name, "Target": rep.target, "Type": rep.kind,
        "Train": rep.train_score, "Test": rep.test_score,
        "CV mean": rep.cv_mean, "CV std": rep.cv_std,
        "Gap": rep.gap, "Status": rep.status,
    })
rep_df = pd.DataFrame(report_rows)

def _color(v):
    if v == "Good Fit":
        return "color: #22c55e"
    if v == "Overfitting":
        return "color: #ef4444"
    return "color: #f59e0b"

try:
    styled = rep_df.style.map(_color, subset=["Status"])
except AttributeError:
    styled = rep_df.style.applymap(_color, subset=["Status"])
st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------- Feature Importance ----------
section("Feature Importance - Random Forest", icon="bar_chart")
imp_df = (pd.DataFrame({"Feature": list(imps.keys()),
                         "Importance": list(imps.values())})
          .sort_values("Importance", ascending=True))
fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
             color="Importance", color_continuous_scale="Tealrose")
fig.update_layout(height=480)
st.plotly_chart(fig, use_container_width=True)

# ---------- Confusion Matrix ----------
section("Confusion Matrix", icon="grid_view")
clf_models = [n for n, r in reports.items()
              if r.kind == "classifier" and r.status != "unavailable"]
chosen = st.selectbox("Choose classifier", clf_models, index=0)
rep = reports[chosen]
if rep.extra and "confusion_matrix" in rep.extra:
    cm = np.array(rep.extra["confusion_matrix"])
    labels = rep.extra["labels"]
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto",
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "True"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{chosen} | Test accuracy = {rep.test_score:.3f} | "
               f"CV = {rep.cv_mean:.3f} +/- {rep.cv_std:.3f}")

# ---------- Predicted vs Actual (regression) ----------
section("Stress Score - Predicted vs Actual (Gradient Boosting)", icon="timeline")
gb = models.get("Gradient Boosting")
if gb is not None:
    sample = filt.sample(min(1500, len(filt)), random_state=3)
    preds = gb.predict(sample[FEATURES])
    plot_df = pd.DataFrame({"Actual": sample["StressScore"].values,
                             "Predicted": preds,
                             "RiskCategory": sample["RiskCategory"].values})
    fig = px.scatter(plot_df, x="Actual", y="Predicted", color="RiskCategory",
                     opacity=0.6,
                     color_discrete_map={"Low": "#22c55e",
                                         "Moderate": "#f59e0b",
                                         "High": "#ef4444"})
    lo, hi = plot_df["Actual"].min(), plot_df["Actual"].max()
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color="cyan", dash="dash"))
    fig.update_layout(height=480, xaxis_title="Actual stress",
                      yaxis_title="Predicted stress")
    st.plotly_chart(fig, use_container_width=True)

# ---------- ROC for Logistic Regression ----------
section("ROC Curve - Logistic Regression (HighRisk vs Not)", icon="show_chart")
lr = models.get("Logistic Regression")
if lr is not None:
    sample = filt.sample(min(2500, len(filt)), random_state=4)
    proba = lr.predict_proba(sample[FEATURES])[:, 1]
    fpr, tpr, _ = roc_curve(sample["HighRiskFlag"], proba)
    auc_val = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                              name=f"ROC (AUC = {auc_val:.3f})",
                              line=dict(color="#2E86AB", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="Random",
                              line=dict(color="grey", dash="dash")))
    fig.update_layout(height=440, xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Coefficient comparison ----------
section("Ridge vs Lasso - Coefficient Comparison (Stress regression)",
        icon="compare_arrows")
ridge = models.get("Ridge Regression")
lasso = models.get("Lasso Regression")
if ridge is not None and lasso is not None:
    ridge_coef = ridge.named_steps["reg"].coef_
    lasso_coef = lasso.named_steps["reg"].coef_
    coef_df = pd.DataFrame({
        "Feature": FEATURES, "Ridge": ridge_coef, "Lasso": lasso_coef
    }).melt(id_vars="Feature", var_name="Model", value_name="Coefficient")
    fig = px.bar(coef_df, x="Coefficient", y="Feature", color="Model",
                 orientation="h", barmode="group")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
