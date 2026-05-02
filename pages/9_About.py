"""About - project documentation, ML model card, references."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils.styles import inject_css, section

st.set_page_config(page_title="About",
                   page_icon=":material/info:", layout="wide")
inject_css()

st.title(":material/info: About WorkHealth Analytics")

section("What it is", icon="description")
st.markdown("""
**WorkHealth Analytics** is an end-to-end IT-workplace health analytics platform
combining **3 best-in-class ML models** with a **rule-based scoring engine** to
classify employees into Low / Moderate / High health-risk bands and produce
**personalised, actionable recommendations**.

The system analyses 15,000 synthetic IT-professional records spanning
**WFH / Office / Hybrid** work modes and a 10-feature lifestyle vector
(Work hours, Sleep, Caffeine, Activity, Screen time, Water, Job satisfaction,
Burnout, BMI, Pain).
""")

section("ML Model Card - the 3 selected models", icon="model_training")
st.markdown("""
Each model was chosen for being **best-in-class for its specific task** —
not just the highest raw score. Hyperparameters are tuned to fit Streamlit
Community Cloud's **1 GB RAM** budget while staying accurate.

| Model | Role | Target | Hyperparameters | Test score |
|---|---|---|---|---|
| **XGBoost** | Multi-class risk classifier | RiskCategory (Low/Mod/High) | n_estimators=120, max_depth=4, lr=0.1, tree_method=hist | ~0.93 acc |
| **Gradient Boosting** | Stress regressor | StressScore (continuous 0-40) | n_estimators=80, max_depth=3, lr=0.08 | ~0.75 R-squared |
| **Logistic Regression** | Binary urgent-action flag | HighRiskFlag (1 = high risk) | StandardScaler + L2, max_iter=500 | ~0.99 acc |
""")

section("Rule-Based Engine", icon="rule")
st.markdown("""
The rule engine scores 11 health parameters with weighted point penalties:

| Parameter | Low (0) | Moderate (5-10) | High (15-20) |
|---|---|---|---|
| Sleep | >= 7 h | 5-7 h | < 5 h |
| Work Hours | <= 9 h | 9-12 h | > 12 h |
| Activity | >= 150 min | 30-150 min | < 30 min |
| BMI | 18.5-25 | 25-30 | > 30 |
| Stress | < 20/40 | 20-30 | > 30 |
| Burnout | < 5/10 | 5-7 | > 7 |
| Screen Time | <= 9 h | 9-12 h | > 12 h |
| Water | >= 2 L | 1.5-2 L | < 1.5 L |
| Caffeine | <= 3 cups | 3-6 cups | > 6 cups |
| Breaks | >= 4 / day | 2-4 | < 2 |
| Job Satisfaction | >= 7 | 4-7 | < 4 |

**Decision rule:** `<20 -> Low | 20-50 -> Moderate | >50 -> High`
""")

section("Architecture", icon="architecture")
st.markdown("""
```
WorkHealth Analytics
|-- app.py                         # Home / KPIs / model banner
|-- pages/                         # 9 Streamlit pages (auto-discovered)
|   |-- 1_Data_Overview.py
|   |-- 2_EDA_Charts.py
|   |-- 3_Correlations.py
|   |-- 4_ML_Insights.py
|   |-- 5_Model_Fitness.py
|   |-- 6_Predict_My_Health.py     # 7-input prediction form
|   |-- 7_Recommendations.py
|   |-- 8_Compare_Peers.py
|   `-- 9_About.py
|-- utils/
|   |-- data_loader.py             # synthetic 15k-row generator (float32)
|   |-- feature_engineering.py     # Risk/Wellness/BMI/Sleep/AgeGroup
|   |-- rule_engine.py             # deterministic point scoring
|   |-- ml_models.py               # 7 models, train + save + load
|   |-- recommendations.py         # action plan + peer KNN search
|   `-- styles.py                  # CSS, KPI strip, sidebar filters
|-- models/                        # pre-trained joblib + JSON (committed)
`-- data/                          # workhealth_data.csv (auto-generated)
```
""")

section("Streamlit Cloud Memory Strategy", icon="memory")
st.markdown("""
The free tier guarantees **1 GB RAM, up to 3 GB peak**. To stay comfortably
under that limit:

- **Pre-trained models are committed** — Streamlit Cloud only loads pickled
  artefacts; no live training happens at request time (saves 300-500 MB
  transient memory on cold start).
- **Compressed pickles** — `joblib.dump(..., compress=3)` cuts the model file
  ~3x.
- **float32 dataframe** — numeric columns cast on load (~2x smaller than
  default float64).
- **Pre-computed learning curves** — saved to `models/learning_curves.json`
  during training; the Model Fitness page reads the file (no live retrain).
- **Trimmed estimators** — Random Forest cut from 300 -> 100 trees and
  depth 8 -> 6; XGBoost depth 5 -> 4.
- **Lazy ML loading** — pages that don't need ML (Data Overview, EDA,
  Correlations, Compare Peers, About) skip the model load entirely.
""")

section("References", icon="menu_book")
st.markdown("""
1. Kivimaki M et al. - *Job strain as a risk factor for coronary heart disease*. The Lancet (2012).
2. Maslach C, Leiter MP - *Burnout*. In Stress: Concepts, Cognition, Emotion, and Behavior (2016).
3. WHO (2020) - *Guidelines on physical activity and sedentary behaviour*.
4. Bloom N et al. - *The effects of remote work on collaboration*. Nature Human Behaviour (2022).
5. Breiman L - *Random Forests*. Machine Learning 45 (2001).
6. Friedman JH - *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics (2001).
7. Chen T, Guestrin C - *XGBoost: A Scalable Tree Boosting System*. KDD (2016).
""")

section("How to run", icon="play_circle")
st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

section("Icon system", icon="palette")
st.markdown("""
This dashboard uses **Google Material Symbols** (the `:material/icon_name:`
shortcode built into Streamlit 1.36+) for all icons — no emoji are used in
the code or filenames. Material Symbols are vector icons rendered through
Streamlit's markdown engine.

Reference list of icon names: <https://fonts.google.com/icons>.
""")
