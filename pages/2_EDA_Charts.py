"""EDA Charts - every major chart type for univariate / bivariate exploration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_data
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="EDA Charts",
                   page_icon=":material/insights:", layout="wide")
inject_css()

st.title(":material/insights: EDA Charts")
st.caption("Histograms | Box | Violin | Scatter | Sunburst | Treemap | Parallel Coords")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

NUMERIC = ["WorkHours", "ScreenTime", "SleepHours", "ActivityMinutes",
           "CaffeineCups", "WaterIntakeL", "BreaksPerDay", "BMI",
           "JobSatisfaction", "StressScore", "BurnoutLevel",
           "WellnessScore", "RiskScore", "Age"]

# ---------- 1. Univariate ----------
section("1) Univariate - pick a metric, see its distribution", icon="show_chart")
ucol1, ucol2 = st.columns([1, 3])
with ucol1:
    metric = st.selectbox("Metric", NUMERIC, index=NUMERIC.index("StressScore"))
    chart_kind = st.radio(
        "Chart type",
        ["Histogram", "Box", "Violin", "ECDF", "Density"], horizontal=True)
    color_by = st.selectbox(
        "Group by", ["WorkMode", "Gender", "AgeGroup", "RiskCategory",
                     "BMICategory", "SleepCategory"])
with ucol2:
    if chart_kind == "Histogram":
        fig = px.histogram(filt, x=metric, color=color_by,
                           barmode="overlay", nbins=40, opacity=0.7)
    elif chart_kind == "Box":
        fig = px.box(filt, x=color_by, y=metric, color=color_by, points="outliers")
    elif chart_kind == "Violin":
        fig = px.violin(filt, x=color_by, y=metric, color=color_by,
                        box=True, points=False)
    elif chart_kind == "ECDF":
        fig = px.ecdf(filt, x=metric, color=color_by)
    else:
        fig = px.histogram(filt, x=metric, color=color_by, nbins=50,
                           histnorm="density", barmode="overlay", opacity=0.6)
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2. Bivariate scatter ----------
section("2) Bivariate Scatter Explorer", icon="scatter_plot")
b1, b2, b3, b4 = st.columns(4)
with b1:
    x_axis = st.selectbox("X", NUMERIC, index=NUMERIC.index("WorkHours"), key="x")
with b2:
    y_axis = st.selectbox("Y", NUMERIC, index=NUMERIC.index("StressScore"), key="y")
with b3:
    color_axis = st.selectbox(
        "Color", ["RiskCategory", "WorkMode", "Gender", "AgeGroup"], key="c")
with b4:
    size_axis = st.selectbox(
        "Size", ["BurnoutLevel", "BMI", "RiskScore"], key="s")
fig = px.scatter(
    filt.sample(min(2500, len(filt)), random_state=1),
    x=x_axis, y=y_axis, color=color_axis, size=size_axis,
    opacity=0.7, hover_data=["EmployeeID", "Department"])
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------- 3. Sunburst & Treemap ----------
section("3) Hierarchical Views - Sunburst & Treemap", icon="account_tree")
c1, c2 = st.columns(2)
agg = filt.groupby(["WorkMode", "RiskCategory"]).size().reset_index(name="n")
with c1:
    fig = px.sunburst(agg, path=["WorkMode", "RiskCategory"], values="n",
                      color="RiskCategory",
                      color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b",
                                          "High": "#ef4444"},
                      title="Work Mode -> Risk")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    agg2 = filt.groupby(["Department", "RiskCategory"]).size().reset_index(name="n")
    fig = px.treemap(agg2, path=["Department", "RiskCategory"], values="n",
                     color="RiskCategory",
                     color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b",
                                         "High": "#ef4444"},
                     title="Department -> Risk")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 4. Stacked bars by Age Group ----------
section("4) Risk by Age Group (Stacked Bar)", icon="stacked_bar_chart")
age_mix = (filt.groupby(["AgeGroup", "RiskCategory"]).size()
           .reset_index(name="n"))
fig = px.bar(age_mix, x="AgeGroup", y="n", color="RiskCategory", barmode="stack",
             color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b",
                                 "High": "#ef4444"})
fig.update_layout(height=420, yaxis_title="Employees")
st.plotly_chart(fig, use_container_width=True)

# ---------- 5. Parallel coordinates ----------
section("5) Parallel Coordinates - multivariate fingerprints", icon="ssid_chart")
sample = filt.sample(min(800, len(filt)), random_state=2).copy()
sample["RiskN"] = sample["RiskCategory"].map({"Low": 0, "Moderate": 1, "High": 2})
fig = go.Figure(data=go.Parcoords(
    line=dict(color=sample["RiskN"],
              colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1.0, "#ef4444"]],
              showscale=True, cmin=0, cmax=2),
    dimensions=[
        dict(label="Sleep", values=sample["SleepHours"]),
        dict(label="Work hrs", values=sample["WorkHours"]),
        dict(label="Activity", values=sample["ActivityMinutes"]),
        dict(label="Stress", values=sample["StressScore"]),
        dict(label="Burnout", values=sample["BurnoutLevel"]),
        dict(label="Caffeine", values=sample["CaffeineCups"]),
        dict(label="JobSat", values=sample["JobSatisfaction"]),
    ],
))
fig.update_layout(height=480)
st.plotly_chart(fig, use_container_width=True)

# ---------- 6. Trend by Age ----------
section("6) Trend Lines - metric vs Age", icon="trending_up")
trend_metric = st.selectbox("Trend metric", NUMERIC,
                             index=NUMERIC.index("BurnoutLevel"))
agg = filt.groupby("Age")[trend_metric].mean().reset_index()
fig = px.line(agg, x="Age", y=trend_metric, markers=True)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)
