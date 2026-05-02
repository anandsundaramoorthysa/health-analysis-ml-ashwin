"""Recommendations - personalised action plan + peer comparison."""
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
from utils.recommendations import recommend
from utils.rule_engine import assess
from utils.styles import inject_css, section

st.set_page_config(page_title="Recommendations",
                   page_icon=":material/lightbulb:", layout="wide")
inject_css()

st.title(":material/lightbulb: Personalised Recommendations")
st.caption("ML + rule-engine output translated into concrete next steps.")

df = load_data()
artefacts = load_or_train(df)
models = artefacts["models"]

# ---------- Fetch from session, or build a fresh default ----------
user_row = st.session_state.get("last_user_row")
rule_result = st.session_state.get("last_rule_result")
ml_result = st.session_state.get("last_ml_result")

if user_row is None:
    st.info("Tip: open **Predict My Health** first to get a personalised "
            "result, or adjust the quick form below.")
    with st.expander("Quick form (defaults shown)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            wh = st.slider("Work hrs/day", 4.0, 16.0, 11.0, 0.5)
            sh = st.slider("Sleep hrs", 3.0, 11.0, 5.5, 0.5)
        with c2:
            cc = st.slider("Coffee cups", 0, 10, 5, 1)
            am = st.slider("Activity (min)", 0, 240, 25, 5)
        with c3:
            sc = st.slider("Screen hrs", 2.0, 16.0, 12.0, 0.5)
            wt = st.slider("Water (L)", 0.5, 5.0, 1.2, 0.1)
        js = st.slider("Job satisfaction", 1, 10, 5, 1)
        user_row = build_user_record(wh, sh, cc, am, sc, wt, js)
    rule_result = assess(user_row.iloc[0].to_dict())
    ml_result = predict_for_user(models, user_row)

rec = recommend(user_row, rule_result, ml_result, df)

# ---------- Headline ----------
section("Your Result", icon="summarize")
hcol1, hcol2, hcol3 = st.columns(3)
with hcol1:
    st.markdown(
        f"<div class='kpi-card' style='border-color:{rec['color']}'>"
        f"<div class='kpi-label'>Risk Category</div>"
        f"<div class='kpi-value' style='color:{rec['color']}'>{rec['category']}</div>"
        f"<div class='kpi-sub'>{rec['total_points']} pts | "
        f"{len(rec['actions'])} issue(s) flagged</div></div>",
        unsafe_allow_html=True)
with hcol2:
    st.metric("Predicted Stress",
              f"{ml_result.get('StressPrediction', '-')} / 40")
with hcol3:
    st.metric("High-Risk probability",
              f"{ml_result.get('HighRiskProb', 0)*100:.1f}%")

# ---------- Action plan ----------
section(f"Your Action Plan - {len(rec['actions'])} priorities",
        icon="checklist")
if not rec["actions"]:
    st.success("No issues flagged - keep up your healthy habits!")
for i, action in enumerate(rec["actions"], 1):
    color = ("#ef4444" if action["points"] >= 15
             else "#f59e0b" if action["points"] >= 8
             else "#94a3b8")
    with st.expander(f"#{i} | {action['area']} | +{action['points']} pts "
                     f"({action['severity']})", expanded=(i <= 3)):
        st.markdown(f"<div class='tip-card' style='border-left-color:{color}'>"
                    f"{action['note']}</div>", unsafe_allow_html=True)
        for tip in action["tips"]:
            st.markdown(f"- {tip}")

# ---------- Goals tracker ----------
if rec["goals"]:
    section("Concrete Goals", icon="flag")
    g_df = pd.DataFrame(rec["goals"])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=g_df["area"], y=g_df["current"],
                         marker_color="#A23B72"))
    fig.add_trace(go.Bar(name="Target",  x=g_df["area"], y=g_df["target"],
                         marker_color="#22c55e"))
    fig.update_layout(barmode="group", height=380,
                      yaxis_title="Value (mixed units)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(g_df, use_container_width=True, hide_index=True)

# ---------- You vs dataset average ----------
section("You vs Dataset Average", icon="compare")
mean_df = pd.DataFrame({
    "Metric": list(rec["dataset_means"].keys()),
    "Dataset average": list(rec["dataset_means"].values()),
    "You": [rec["user_vals"][k] for k in rec["dataset_means"]],
    "Delta vs avg": [rec["delta"][k] for k in rec["dataset_means"]],
})
fig = px.bar(mean_df.melt(id_vars="Metric",
                           value_vars=["You", "Dataset average"]),
             x="Metric", y="value", color="variable", barmode="group",
             color_discrete_map={"You": "#2E86AB", "Dataset average": "#A23B72"})
fig.update_layout(height=420, xaxis_tickangle=-30, yaxis_title="")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(mean_df.round(2), use_container_width=True, hide_index=True)

# ---------- Peer summary ----------
section("People With Similar Lifestyle", icon="groups")
peer = rec["peer"]
p1, p2, p3 = st.columns(3)
with p1:
    st.metric("Peers compared", peer["n_peers"])
with p2:
    st.metric("Avg risk score (peers)", peer["peer_avg_risk_score"])
with p3:
    st.metric("% high risk among peers", f"{peer['peer_high_risk_pct']}%")

st.caption("Peers are the 50 dataset employees most similar to your inputs "
           "(Euclidean distance over the 7 input features).")

# ---------- Download report ----------
section("Download Report", icon="download")
report_md = f"""# WorkHealth Analytics - Personalised Health Report

**Risk category:** {rec['category']}  |  **Total points:** {rec['total_points']}
**ML predicted stress:** {ml_result.get('StressPrediction', '-')}/40
**ML high-risk probability:** {ml_result.get('HighRiskProb', 0)*100:.1f}%

## Issues flagged ({len(rec['actions'])})
""" + "\n".join(
    f"- **{a['area']}** (+{a['points']} pts, {a['severity']}): {a['note']}"
    for a in rec["actions"]
) + "\n\n## Action tips\n" + "\n".join(
    f"### {a['area']}\n" + "\n".join(f"- {t}" for t in a['tips'])
    for a in rec["actions"]
) + f"\n\n## Peers\nCompared with {peer['n_peers']} similar employees | " \
    f"average risk score {peer['peer_avg_risk_score']} | " \
    f"{peer['peer_high_risk_pct']}% of peers are in the High Risk band.\n"

st.download_button("Download report (Markdown)",
                   report_md.encode("utf-8"),
                   file_name="workhealth_report.md", mime="text/markdown")
