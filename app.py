"""WorkHealth AI Pro - Home / Landing page.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_data
from utils.ml_models import load_or_train
from utils.styles import inject_css, kpi_row, sidebar_filters, section

st.set_page_config(
    page_title="WorkHealth AI Pro",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---------- Header ----------
st.title(":material/health_and_safety: WorkHealth AI Pro")
st.markdown(
    "**IT Workplace Health Monitoring | ML Risk Prediction | "
    "Rule-Based Wellness | Personalised Recommendations**"
)
st.caption(
    "Hybrid ML + Rules system analysing 15,000 IT professionals across "
    "WFH / Office / Hybrid work modes."
)

# ---------- Load data + models (cached) ----------
df = load_data()
artefacts = load_or_train(df)

# ---------- Sidebar ----------
filt = sidebar_filters(df)

# ---------- KPI Strip ----------
section("Live KPI Snapshot", icon="speed")
kpi_row(filt)

# ---------- Risk distribution + Work mode mix ----------
left, right = st.columns([1.4, 1])
with left:
    section("Risk Category Distribution", icon="bar_chart")
    risk_counts = (filt["RiskCategory"].value_counts()
                   .reindex(["Low", "Moderate", "High"]).fillna(0).reset_index())
    risk_counts.columns = ["RiskCategory", "Count"]
    fig = px.bar(
        risk_counts, x="RiskCategory", y="Count",
        color="RiskCategory",
        color_discrete_map={"Low": "#22c55e", "Moderate": "#f59e0b",
                            "High": "#ef4444"},
        text="Count",
    )
    fig.update_layout(height=380, showlegend=False,
                      xaxis_title="", yaxis_title="Employees")
    st.plotly_chart(fig, use_container_width=True)
with right:
    section("Work Mode Mix", icon="donut_large")
    fig = px.pie(filt, names="WorkMode", hole=0.55,
                 color="WorkMode",
                 color_discrete_map={"WFH": "#2E86AB", "Office": "#A23B72",
                                     "Hybrid": "#F18F01"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Page directory ----------
section("Pages in this Dashboard", icon="menu_book")
pages = [
    ("Data Overview", "Univariate stats, demographics, distributions"),
    ("EDA Charts", "30+ interactive charts: histograms, box, violin, sunburst, treemap"),
    ("Correlations", "Heatmap, scatter explorer, top-N feature correlations"),
    ("ML Insights", "8 ML models | risk vs stress prediction | feature importance"),
    ("Model Fitness", "Train / Test / CV scores | learning curves | overfitting fix"),
    ("Predict My Health", "7-input prediction form | ML + rules | gauge charts"),
    ("Recommendations", "Personalised action plan | peer comparison | goals tracker"),
    ("Compare Peers", "How do you compare with employees in similar roles?"),
    ("About", "Project documentation, model details, references"),
]
st.dataframe(
    pd.DataFrame(pages, columns=["Page", "Description"]),
    use_container_width=True, hide_index=True,
)

# ---------- ML model banners ----------
section("Trained ML Models", icon="model_training")
reports = artefacts["reports"]
cols = st.columns(4)
for i, (name, rep) in enumerate(reports.items()):
    with cols[i % 4]:
        score_label = "R-squared" if rep.kind == "regressor" else "Accuracy"
        st.metric(
            label=name,
            value=f"{rep.test_score:.3f} {score_label}",
            delta=rep.status,
            delta_color="normal" if rep.status == "Good Fit" else "inverse",
        )

st.divider()
st.caption(
    "Built with Streamlit | scikit-learn | XGBoost | Plotly. "
    "Use the sidebar to navigate between pages."
)
