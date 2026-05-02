"""Model Fitness - train / test / CV breakdown, learning curves, gap analysis.

Memory note: learning curves are pre-computed at training time and stored as
JSON (`models/learning_curves.json`). The page reads them — no live retraining
runs in the user request path, keeping memory well under Streamlit Cloud's
1 GB cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_data
from utils.ml_models import load_or_train
from utils.styles import inject_css, kpi_row, section, sidebar_filters

st.set_page_config(page_title="Model Fitness",
                   page_icon=":material/equalizer:", layout="wide")
inject_css()

st.title(":material/equalizer: Model Fitness Analysis")
st.caption("Train | Test | 5-fold CV scores | learning curves | gap analysis.")

df = load_data()
filt = sidebar_filters(df)
kpi_row(filt)

artefacts = load_or_train(df)
reports = artefacts["reports"]
learning_curves = artefacts.get("learning_curves", {})

# ---------- Score table ----------
section("Train / Test / CV Scores", icon="scoreboard")
rows = []
for name, rep in reports.items():
    rows.append({
        "Model": name, "Target": rep.target,
        "Train": rep.train_score, "Test": rep.test_score,
        "CV": rep.cv_mean, "Gap": rep.gap, "Status": rep.status,
    })
score_df = pd.DataFrame(rows)
st.dataframe(score_df, use_container_width=True, hide_index=True)

# ---------- Grouped bar comparison ----------
section("Train vs Test vs CV - Visual Comparison", icon="bar_chart")
melt = score_df.melt(id_vars=["Model", "Status"],
                     value_vars=["Train", "Test", "CV"],
                     var_name="Split", value_name="Score")
fig = px.bar(melt, x="Model", y="Score", color="Split", barmode="group",
             color_discrete_map={"Train": "#2E86AB", "Test": "#A23B72",
                                 "CV": "#F18F01"})
fig.update_layout(height=460, xaxis_tickangle=-25)
st.plotly_chart(fig, use_container_width=True)

# ---------- Gap analysis ----------
section("Train-Test Gap Analysis (smaller = better generalisation)",
        icon="straighten")
gap_df = score_df[["Model", "Gap", "Status"]].sort_values("Gap")
fig = px.bar(gap_df, x="Gap", y="Model", color="Status", orientation="h",
             color_discrete_map={"Good Fit": "#22c55e",
                                 "Overfitting": "#ef4444",
                                 "Underfitting": "#f59e0b"})
fig.add_vline(x=0.10, line_dash="dash", line_color="red",
              annotation_text="Overfit threshold (>0.10)")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

# ---------- Learning curve (precomputed) ----------
section("Learning Curve (pre-computed)", icon="trending_up")
if learning_curves:
    lc_options = [n for n in learning_curves.keys()
                  if isinstance(learning_curves[n], dict)
                  and "sizes" in learning_curves[n]]
    if lc_options:
        chosen = st.selectbox("Model", lc_options, index=0)
        lc = learning_curves[chosen]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lc["sizes"], y=lc["train_mean"],
                                  mode="lines+markers", name="Train",
                                  line=dict(color="#2E86AB", width=3)))
        fig.add_trace(go.Scatter(x=lc["sizes"], y=lc["val_mean"],
                                  mode="lines+markers", name="Validation",
                                  line=dict(color="#F18F01", width=3)))
        fig.update_layout(height=440, xaxis_title="Training samples",
                          yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Curves were computed once during training and persisted "
                   "to `models/learning_curves.json`. The dashboard reads "
                   "the file — no live retraining occurs here.")
    else:
        st.info("Learning curves are not available for this build.")
else:
    st.info("Re-train the models to generate learning curves "
            "(delete `models/` and reload the app).")

# ---------- Why these 3 models ----------
section("Why these 3 models", icon="hub")
st.markdown(
    """
    The dashboard now ships only the **best-in-class model per prediction
    task** — no redundant baselines, no extra memory cost.

    | Task                     | Model               | Why it was picked          |
    |---|---|---|
    | Multi-class risk         | **XGBoost**         | Highest test accuracy (~0.93) on RiskCategory across all candidates |
    | Stress score regression  | **Gradient Boosting** | Best R-squared (~0.75); beats Ridge / Lasso |
    | Binary high-risk flag    | **Logistic Regression** | Saturated accuracy (~0.99); cheapest in RAM and inference time |

    Removed: Random Forest, Decision Tree, Ridge, Lasso — each was beaten
    on its task by one of the three above.

    **Hyperparameters** (in `utils/ml_models.py`):

    ```python
    XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1,
                  subsample=0.9, colsample_bytree=0.9, tree_method="hist")

    GradientBoostingRegressor(n_estimators=80, max_depth=3,
                              learning_rate=0.08)

    LogisticRegression(max_iter=500)   # wrapped in a StandardScaler pipeline
    ```

    All three are deliberately small to fit Streamlit Community Cloud's
    **1 GB RAM** budget. Pre-trained pickles total **< 0.5 MB** and live
    Streamlit RSS stays under **400 MB** with all 10 pages warm.
    """
)
