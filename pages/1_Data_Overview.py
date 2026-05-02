"""Data Overview - schema, descriptive stats, demographic breakdowns."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_data
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="Data Overview",
                   page_icon=":material/table_chart:", layout="wide")
inject_css()

st.title(":material/table_chart: Data Overview")
st.caption("Schema, descriptive statistics, and demographic distributions.")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

# ---------- Schema ----------
section("Dataset Schema", icon="schema")
schema = pd.DataFrame({
    "Column": df.columns,
    "DType": df.dtypes.astype(str).values,
    "Unique": [df[c].nunique() for c in df.columns],
    "Sample": [str(df[c].iloc[0])[:40] for c in df.columns],
})
st.dataframe(schema, use_container_width=True, hide_index=True)

# ---------- Descriptive statistics ----------
section("Descriptive Statistics - Numeric Features", icon="functions")
num_cols = filt.select_dtypes(include="number").columns.tolist()
st.dataframe(filt[num_cols].describe().round(2).T,
             use_container_width=True)

# ---------- Demographics ----------
section("Demographic Mix", icon="diversity_3")
c1, c2, c3 = st.columns(3)
with c1:
    fig = px.pie(filt, names="Gender", hole=0.45, title="Gender")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.pie(filt, names="WorkMode", hole=0.45, title="Work Mode")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
with c3:
    fig = px.pie(filt, names="AgeGroup", hole=0.45, title="Age Group")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    section("Department Headcount", icon="apartment")
    dept = filt["Department"].value_counts().reset_index()
    dept.columns = ["Department", "Count"]
    fig = px.bar(dept, x="Count", y="Department", orientation="h",
                 color="Count", color_continuous_scale="Tealrose")
    fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
with c2:
    section("BMI Category Mix", icon="monitor_weight")
    bmi_counts = filt["BMICategory"].value_counts().reset_index()
    bmi_counts.columns = ["BMICategory", "Count"]
    fig = px.bar(bmi_counts, x="BMICategory", y="Count",
                 color="BMICategory", text="Count",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Group means ----------
section("Group Means - Work Mode / Gender / Age", icon="bar_chart")
metric = st.selectbox(
    "Metric",
    ["StressScore", "BurnoutLevel", "SleepHours", "WorkHours",
     "ActivityMinutes", "WellnessScore", "RiskScore", "BMI"],
    index=0,
)
g1, g2, g3 = st.columns(3)
with g1:
    grp = filt.groupby("WorkMode")[metric].mean().reset_index()
    fig = px.bar(grp, x="WorkMode", y=metric, color="WorkMode",
                 text=grp[metric].round(2), title=f"{metric} by Work Mode")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with g2:
    grp = filt.groupby("Gender")[metric].mean().reset_index()
    fig = px.bar(grp, x="Gender", y=metric, color="Gender",
                 text=grp[metric].round(2), title=f"{metric} by Gender")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with g3:
    grp = filt.groupby("AgeGroup")[metric].mean().reset_index()
    fig = px.line(grp, x="AgeGroup", y=metric, markers=True,
                  title=f"{metric} by Age Group")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Raw sample ----------
section("Raw Data Sample (first 50)", icon="dataset")
st.dataframe(filt.head(50), use_container_width=True, hide_index=True)
