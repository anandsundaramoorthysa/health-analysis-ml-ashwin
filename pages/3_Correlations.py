"""Correlations - heatmap, top-N, scatter explorer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_data
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="Correlations",
                   page_icon=":material/scatter_plot:", layout="wide")
inject_css()

st.title(":material/scatter_plot: Correlation Analysis")
st.caption("How do lifestyle inputs relate to stress, burnout, and risk?")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

NUMERIC = ["Age", "WorkHours", "ScreenTime", "SleepHours", "ActivityMinutes",
           "CaffeineCups", "WaterIntakeL", "BreaksPerDay", "BMI",
           "JobSatisfaction", "StressScore", "BurnoutLevel",
           "WellnessScore", "RiskScore"]

# ---------- Method selector ----------
method = st.radio("Correlation method", ["pearson", "spearman", "kendall"],
                  horizontal=True)
corr = filt[NUMERIC].corr(method=method)

# ---------- Heatmap ----------
section(f"Correlation Heatmap - {method.title()}", icon="grid_on")
fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
fig.update_layout(height=620)
st.plotly_chart(fig, use_container_width=True)

# ---------- Top correlations ----------
section("Top 15 Strongest Correlations", icon="leaderboard")
pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((cols[i], cols[j], corr.iloc[i, j]))
top = (pd.DataFrame(pairs, columns=["Feature A", "Feature B", "Correlation"])
       .assign(AbsCorr=lambda d: d["Correlation"].abs())
       .sort_values("AbsCorr", ascending=False)
       .drop(columns="AbsCorr").head(15)
       .round(3))
top["Direction"] = np.where(top["Correlation"] > 0, "Positive", "Negative")
top["Strength"] = pd.cut(
    top["Correlation"].abs(),
    bins=[0, 0.2, 0.4, 0.6, 1.0],
    labels=["Weak", "Moderate", "Strong", "Very Strong"]).astype(str)
st.dataframe(top, use_container_width=True, hide_index=True)

# ---------- Interactive scatter ----------
section("Scatter Explorer - pick any X vs Y", icon="bubble_chart")
c1, c2, c3 = st.columns(3)
with c1:
    xa = st.selectbox("X", NUMERIC, index=NUMERIC.index("SleepHours"))
with c2:
    ya = st.selectbox("Y", NUMERIC, index=NUMERIC.index("StressScore"))
with c3:
    color = st.selectbox("Color", ["RiskCategory", "WorkMode", "Gender",
                                    "AgeGroup", "BMICategory"])
scatter_sample = filt.sample(min(800, len(filt)), random_state=1)
try:
    fig = px.scatter(
        scatter_sample, x=xa, y=ya, color=color, opacity=0.7, trendline="ols",
        color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b",
                            "High": "#ef4444"})
except (ImportError, ModuleNotFoundError):
    fig = px.scatter(
        scatter_sample, x=xa, y=ya, color=color, opacity=0.7,
        color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b",
                            "High": "#ef4444"})
fig.update_layout(height=520)
st.plotly_chart(fig, use_container_width=True)

# ---------- Pearson r summary ----------
r = filt[xa].corr(filt[ya])
direction = "Positive" if r > 0 else "Negative"
strength = ("Weak" if abs(r) < 0.2 else "Moderate" if abs(r) < 0.4
             else "Strong" if abs(r) < 0.6 else "Very Strong")
st.info(f"**Pearson r = {r:.3f}** | {direction} | {strength}  |  "
        f"n = {len(filt):,}")
