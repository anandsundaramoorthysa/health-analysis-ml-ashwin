"""ML Insights - 3-model summary, feature importances, predictions, ROC."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, roc_curve

from utils.data_loader import load_data
from utils.ml_models import FEATURES, load_or_train
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="ML Insights",
                   page_icon=":material/model_training:", layout="wide")
inject_css()

st.title(":material/model_training: ML Insights")
st.caption("3 best-in-class models | feature importance | confusion matrix | "
           "predicted vs actual | ROC curve.")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

artefacts = load_or_train(df)
models = artefacts["models"]
reports = artefacts["reports"]
imps = artefacts["importances"]

# ---------- Model line-up ----------
section("The 3 Models in Use", icon="hub")
st.markdown("""
| Role | Model | Target | Why this one |
|---|---|---|---|
| Multi-class risk classifier | **XGBoost** | RiskCategory (Low/Mod/High) | Highest test accuracy across all candidates |
| Stress regressor            | **Gradient Boosting** | StressScore (0-40) | Best R-squared on the regression task |
| Urgent-action binary flag   | **Logistic Regression** | HighRiskFlag | Saturated accuracy, smallest memory footprint |
""")

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

# ---------- Feature Importance (XGBoost) ----------
section("Feature Importance - XGBoost", icon="bar_chart")
imp_df = (pd.DataFrame({"Feature": list(imps.keys()),
                         "Importance": list(imps.values())})
          .sort_values("Importance", ascending=True))
fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
             color="Importance", color_continuous_scale="Tealrose")
fig.update_layout(height=480)
st.plotly_chart(fig, use_container_width=True)
st.caption("Higher = the feature contributes more to the XGBoost decision "
           "trees when classifying risk category.")

# ---------- Confusion matrices for both classifiers ----------
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

# ---------- Predicted vs Actual (Gradient Boosting) ----------
section("Stress Score - Predicted vs Actual (Gradient Boosting)",
        icon="timeline")
gb = models.get("Gradient Boosting")
if gb is not None:
    sample = filt.sample(min(600, len(filt)), random_state=3)
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
    st.caption("Closer to the dashed cyan line = better prediction. "
               "MAE: " + str(reports['Gradient Boosting'].extra.get('mae', '-')))

# ---------- ROC for Logistic Regression ----------
section("ROC Curve - Logistic Regression (HighRisk vs Not)",
        icon="show_chart")
lr = models.get("Logistic Regression")
if lr is not None:
    sample = filt.sample(min(1000, len(filt)), random_state=4)
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
    st.caption("AUC close to 1.0 = the model perfectly separates "
               "high-risk from non-high-risk employees.")
