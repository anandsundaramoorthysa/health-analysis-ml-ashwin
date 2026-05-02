"""Compare Peers - drill-down by department, role, age band."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from utils.data_loader import load_data
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="Compare Peers",
                   page_icon=":material/groups:", layout="wide")
inject_css()

st.title(":material/groups: Compare Peers")
st.caption("Group-level analysis - how does each cohort compare?")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

# ---------- Group-by metric ----------
section("Cohort Means", icon="bar_chart")
group_dim = st.selectbox(
    "Group by", ["WorkMode", "Department", "Gender", "AgeGroup",
                 "BMICategory", "SleepCategory", "RiskCategory"], index=0)
metric = st.selectbox(
    "Metric", ["StressScore", "BurnoutLevel", "WellnessScore", "RiskScore",
               "WorkHours", "SleepHours", "ActivityMinutes",
               "JobSatisfaction", "CaffeineCups", "ScreenTime"], index=0)

agg = (filt.groupby(group_dim)[metric]
       .agg(["mean", "median", "std", "count"])
       .round(2).reset_index().sort_values("mean", ascending=False))
fig = px.bar(agg, x=group_dim, y="mean", color=group_dim,
             text=agg["mean"], hover_data=["median", "std", "count"])
fig.update_layout(height=460, showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(agg, use_container_width=True, hide_index=True)

# ---------- Heatmap by 2 dimensions ----------
section("Two-Way Heatmap (avg metric per cell)", icon="grid_on")
c1, c2, c3 = st.columns(3)
with c1:
    rows = st.selectbox("Rows", ["WorkMode", "Department", "AgeGroup",
                                  "Gender"], index=0)
with c2:
    cols = st.selectbox("Cols", ["RiskCategory", "BMICategory",
                                  "SleepCategory"], index=0)
with c3:
    cell_metric = st.selectbox(
        "Cell value",
        ["StressScore", "BurnoutLevel", "WellnessScore", "WorkHours",
         "SleepHours", "ActivityMinutes"], index=0)
pivot = (filt.groupby([rows, cols])[cell_metric].mean().round(2)
         .unstack(fill_value=0))
fig = px.imshow(pivot, text_auto=True, aspect="auto",
                color_continuous_scale="Tealrose")
fig.update_layout(height=480, xaxis_title=cols, yaxis_title=rows)
st.plotly_chart(fig, use_container_width=True)

# ---------- Box plot per cohort ----------
section("Distribution per Cohort", icon="candlestick_chart")
fig = px.box(filt, x=group_dim, y=metric, color=group_dim, points=False)
fig.update_layout(height=480, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---------- High-risk rate per cohort ----------
section("High-Risk Rate (%) per Cohort", icon="warning")
hr = (filt.assign(High=(filt["RiskCategory"] == "High").astype(int))
      .groupby(group_dim)["High"].mean().mul(100).round(1)
      .reset_index().sort_values("High", ascending=False))
fig = px.bar(hr, x=group_dim, y="High", color="High",
             color_continuous_scale="Reds", text="High")
fig.update_layout(height=440, yaxis_title="% high-risk")
st.plotly_chart(fig, use_container_width=True)
