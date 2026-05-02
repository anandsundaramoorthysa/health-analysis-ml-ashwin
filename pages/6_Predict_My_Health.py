"""Predict My Health - 7-input form, ML predictions, rule engine, gauges, radar."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_data
from utils.feature_engineering import build_user_record
from utils.ml_models import load_or_train, predict_for_user
from utils.rule_engine import assess
from utils.styles import inject_css, risk_badge, section

st.set_page_config(page_title="Predict My Health",
                   page_icon=":material/monitor_heart:", layout="wide")
inject_css()

st.title(":material/monitor_heart: Predict My Health")
st.caption("Enter your daily lifestyle metrics - get an instant ML + rule-based "
           "health risk assessment.")

df = load_data()
artefacts = load_or_train(df)
models = artefacts["models"]

# ---------- 7 inputs ----------
section("Your 7 Daily Inputs", icon="edit_note")
c1, c2, c3, c4 = st.columns(4)
with c1:
    work_hours = st.slider("Work Hours / day", 4.0, 16.0, 9.0, 0.5)
    sleep_hours = st.slider("Sleep Hours", 3.0, 11.0, 6.5, 0.5)
with c2:
    coffee = st.slider("Coffee / Caffeine cups", 0, 10, 3, 1)
    activity = st.slider("Activity (min/day)", 0, 240, 45, 5)
with c3:
    screen = st.slider("Screen Time (h)", 2.0, 16.0, 10.0, 0.5)
    water = st.slider("Water Intake (L)", 0.5, 5.0, 1.8, 0.1)
with c4:
    job_sat = st.slider("Job Satisfaction (1-10)", 1, 10, 6, 1)
    age = st.number_input("Age (optional)", 18, 70, 30, 1)

with st.expander("Optional advanced inputs"):
    a1, a2, a3 = st.columns(3)
    with a1:
        bmi = st.number_input("BMI", 16.0, 45.0, 24.0, 0.1)
    with a2:
        burnout = st.slider("Self-rated burnout (0-10)", 0.0, 10.0, 5.0, 0.5)
    with a3:
        pain = st.selectbox("Body pain level",
                             ["None", "Mild", "Moderate", "Severe"])

# ---------- Build user row ----------
user_row = build_user_record(
    work_hours=work_hours, sleep_hours=sleep_hours, coffee_cups=coffee,
    activity_min=activity, screen_time=screen, water_l=water,
    job_sat=job_sat, age=age, bmi=bmi, burnout_level=burnout, pain_level=pain,
)
user_payload = user_row.iloc[0].to_dict()

# ---------- Run rules + ML ----------
rule_result = assess(user_payload)
ml_result = predict_for_user(models, user_row)

# ---------- Headline result ----------
section("Your Health Risk Assessment", icon="insights")
hcol1, hcol2, hcol3, hcol4 = st.columns([1.2, 1, 1, 1])
with hcol1:
    st.markdown(
        f"<div class='kpi-card' style='border-color:{rule_result['color']}'>"
        f"<div class='kpi-label'>Risk Category</div>"
        f"<div class='kpi-value' style='color:{rule_result['color']}'>"
        f"{rule_result['category']}</div>"
        f"<div class='kpi-sub'>Rule engine total: {rule_result['total']} pts</div>"
        f"</div>", unsafe_allow_html=True)
with hcol2:
    proba_dict = ml_result.get("RiskProba") or {"-": 0}
    st.metric("ML Risk (XGBoost)", ml_result.get("RiskCategory", "-"),
              delta=f"P={max(proba_dict.values()):.2f}")
with hcol3:
    st.metric("Predicted Stress (GBM)",
              f"{ml_result.get('StressPrediction', '-')} / 40")
with hcol4:
    hr = ml_result.get("HighRiskProb", 0.0)
    st.metric("High-Risk probability (LR)", f"{hr*100:.1f}%",
              delta="urgent" if hr > 0.5 else "ok",
              delta_color="inverse" if hr > 0.5 else "normal")

st.markdown(risk_badge(rule_result["category"]), unsafe_allow_html=True)

# ---------- 7+1 gauge charts ----------
section("Per-Input Gauge Charts", icon="speed")

def gauge(value, title, low, mid, scale_min, scale_max, reverse=False):
    """`reverse=True` means LOW values are good (e.g. work hours)."""
    if reverse:
        ranges = [(scale_min, low, "#22c55e"),
                  (low, mid, "#f59e0b"),
                  (mid, scale_max, "#ef4444")]
    else:
        ranges = [(scale_min, low, "#ef4444"),
                  (low, mid, "#f59e0b"),
                  (mid, scale_max, "#22c55e")]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value, title={"text": title},
        gauge={
            "axis": {"range": [scale_min, scale_max]},
            "bar": {"color": "#2E86AB"},
            "steps": [{"range": [a, b], "color": c} for a, b, c in ranges],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
    return fig

g_cols = st.columns(4)
gauges = [
    (work_hours, "Work Hours", 9, 12, 0, 16, True),
    (sleep_hours, "Sleep", 5, 7, 0, 11, False),
    (coffee, "Coffee", 3, 6, 0, 10, True),
    (activity, "Activity (min)", 30, 150, 0, 240, False),
    (screen, "Screen Time", 9, 12, 0, 16, True),
    (water, "Water (L)", 1.5, 2.0, 0, 5, False),
    (job_sat, "Job Sat", 4, 7, 0, 10, False),
    (rule_result["total"], "Total Risk Score", 20, 50, 0, 110, True),
]
for i, args in enumerate(gauges):
    with g_cols[i % 4]:
        st.plotly_chart(gauge(*args), use_container_width=True)

# ---------- Radar chart ----------
section("Lifestyle Radar - You vs Dataset Average", icon="radar")
metrics = ["WorkHours", "SleepHours", "CaffeineCups", "ActivityMinutes",
           "ScreenTime", "WaterIntakeL", "JobSatisfaction"]
labels = ["Work hrs", "Sleep", "Coffee", "Activity", "Screen", "Water", "JobSat"]

def _norm(series, val):
    lo, hi = float(series.min()), float(series.max())
    return 100 * (val - lo) / (hi - lo + 1e-9)

you = [_norm(df[m], float(user_row.iloc[0][m])) for m in metrics]
avg = [_norm(df[m], float(df[m].mean())) for m in metrics]
radar = go.Figure()
radar.add_trace(go.Scatterpolar(r=you + [you[0]], theta=labels + [labels[0]],
                                 fill="toself", name="You",
                                 line_color="#2E86AB"))
radar.add_trace(go.Scatterpolar(r=avg + [avg[0]], theta=labels + [labels[0]],
                                 fill="toself", name="Dataset avg",
                                 line_color="#A23B72", opacity=0.6))
radar.update_layout(height=480,
                     polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
st.plotly_chart(radar, use_container_width=True)

# ---------- Issue & Healthy area cards ----------
section("Per-Rule Breakdown", icon="rule")
br = rule_result["breakdown"]
left, right = st.columns(2)
with left:
    st.markdown("##### :material/warning: Issues to Address")
    if not rule_result["issues"]:
        st.success("No issues found - keep it up!")
    for area in rule_result["issues"]:
        r = br[area]
        color = "#ef4444" if r.points >= 15 else "#f59e0b"
        st.markdown(
            f"<div class='tip-card' style='border-left-color:{color}'>"
            f"<b>{area}</b> | +{r.points} pts | "
            f"<span style='color:{color}'>{r.label}</span><br>"
            f"<small>{r.note}</small></div>",
            unsafe_allow_html=True)
with right:
    st.markdown("##### :material/check_circle: Healthy Areas")
    for area in rule_result["healthy"]:
        r = br[area]
        st.markdown(
            f"<div class='tip-card' style='border-left-color:#22c55e'>"
            f"<b>{area}</b> | OK<br><small>{r.note}</small></div>",
            unsafe_allow_html=True)

# ---------- Issue score bar ----------
section("Issue Score Breakdown", icon="bar_chart")
score_rows = [{"Area": k, "Points": v.points, "Status": v.label}
              for k, v in br.items()]
fig = px.bar(pd.DataFrame(score_rows).sort_values("Points", ascending=True),
             x="Points", y="Area", orientation="h", color="Status",
             color_discrete_map={"OK": "#22c55e", "Watch": "#f59e0b",
                                 "Critical": "#ef4444", "Neutral": "#94a3b8"})
fig.update_layout(height=440)
st.plotly_chart(fig, use_container_width=True)

# Stash on session for the recommendation page
st.session_state["last_user_row"] = user_row
st.session_state["last_rule_result"] = rule_result
st.session_state["last_ml_result"] = ml_result

st.success("Assessment complete. Open the **Recommendations** page for "
           "your personalised action plan.")
