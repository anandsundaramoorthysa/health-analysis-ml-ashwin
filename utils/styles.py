"""Shared CSS, KPI cards, and sidebar filter helpers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

CUSTOM_CSS = """
<style>
.kpi-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border-radius: 14px;
    padding: 18px 16px;
    border: 1px solid rgba(46, 134, 171, 0.35);
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.kpi-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.kpi-value {
    color: #f8fafc;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    color: #64748b;
    font-size: 0.72rem;
    margin-top: 4px;
}
.section-title {
    color: #2E86AB;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    border-bottom: 2px solid rgba(46,134,171,0.35);
    padding-bottom: 4px;
}
.badge { padding: 4px 10px; border-radius: 999px; font-size: 0.78rem;
         font-weight: 600; display: inline-block; }
.badge-low  { background: rgba(34,197,94,0.18); color: #22c55e;  border: 1px solid #22c55e; }
.badge-mod  { background: rgba(245,158,11,0.18); color: #f59e0b; border: 1px solid #f59e0b; }
.badge-high { background: rgba(239,68,68,0.18); color: #ef4444;  border: 1px solid #ef4444; }
.tip-card {
    background: rgba(46,134,171,0.08);
    border-left: 4px solid #2E86AB;
    padding: 10px 14px; border-radius: 8px; margin: 6px 0;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section(title: str, icon: str | None = None) -> None:
    """Render a section heading. Pass `icon` as a Material Symbols name
    (e.g. 'insights', 'monitor_heart') — never a unicode emoji."""
    if icon:
        st.markdown(f"#### :material/{icon}: {title}")
    else:
        st.markdown(f"#### {title}")


def kpi_card(label: str, value: str, sub: str = "") -> None:
    html = f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def kpi_row(df: pd.DataFrame) -> None:
    """Render the standard 8-KPI strip."""
    high = (df["RiskCategory"] == "High").mean() * 100
    moderate = (df["RiskCategory"] == "Moderate").mean() * 100
    avg_stress = df["StressScore"].mean()
    avg_sleep = df["SleepHours"].mean()
    avg_burnout = df["BurnoutLevel"].mean()
    avg_work = df["WorkHours"].mean()
    avg_wellness = df["WellnessScore"].mean()
    n = len(df)

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Employees", f"{n:,}", "filtered records")
    with cols[1]:
        kpi_card("High Risk", f"{high:.1f}%", "of selection")
    with cols[2]:
        kpi_card("Moderate Risk", f"{moderate:.1f}%", "needs monitoring")
    with cols[3]:
        kpi_card("Avg Stress", f"{avg_stress:.1f}/40", "scale 0–40")

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Avg Sleep", f"{avg_sleep:.1f}h", "WHO target ≥7h")
    with cols[1]:
        kpi_card("Avg Burnout", f"{avg_burnout:.1f}/10", "0=none, 10=severe")
    with cols[2]:
        kpi_card("Avg Work Hrs", f"{avg_work:.1f}h", "per day")
    with cols[3]:
        kpi_card("Wellness Score", f"{avg_wellness:.0f}", "higher = healthier")


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Standard sidebar filters used across most pages."""
    st.sidebar.markdown("### Filters")
    work_modes = st.sidebar.multiselect(
        "Work Mode", sorted(df["WorkMode"].unique()),
        default=list(sorted(df["WorkMode"].unique())))
    genders = st.sidebar.multiselect(
        "Gender", sorted(df["Gender"].unique()),
        default=list(sorted(df["Gender"].unique())))
    age_range = st.sidebar.slider(
        "Age range", int(df["Age"].min()), int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())))
    departments = st.sidebar.multiselect(
        "Departments", sorted(df["Department"].unique()),
        default=list(sorted(df["Department"].unique())))

    filt = df[
        df["WorkMode"].isin(work_modes)
        & df["Gender"].isin(genders)
        & df["Department"].isin(departments)
        & df["Age"].between(age_range[0], age_range[1])
    ]
    st.sidebar.markdown(f"**Filtered:** {len(filt):,} of {len(df):,} records")
    return filt


def risk_badge(category: str) -> str:
    if category == "Low":
        return "<span class='badge badge-low'>LOW RISK</span>"
    if category == "Moderate":
        return "<span class='badge badge-mod'>MODERATE RISK</span>"
    return "<span class='badge badge-high'>HIGH RISK</span>"
