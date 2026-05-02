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

st.title(":material/info: About WorkHealth AI Pro")

section("What it is", icon="description")
st.markdown("""
**WorkHealth AI Pro** is an end-to-end IT-workplace health analytics platform
combining **8 supervised ML models** with a **rule-based scoring engine** to
classify employees into Low / Moderate / High health-risk bands and produce
**personalised, actionable recommendations**.

The system analyses 15,000 IT-professional records spanning **WFH / Office / Hybrid**
work modes and a 10-feature lifestyle vector (Work hours, Sleep, Caffeine,
Activity, Screen time, Water, Job satisfaction, Burnout, BMI, Pain).
""")

section("ML Model Card", icon="model_training")
st.markdown("""
| Model | Type | Target | Purpose |
|---|---|---|---|
| Random Forest | Classification | RiskCategory | Primary risk classifier (regularised: max_depth=8, min_samples_leaf=5) |
| XGBoost | Classification | RiskCategory | Boosted-tree challenger model |
| Gradient Boosting | Regression | StressScore | Continuous stress estimator |
| Logistic Regression | Binary classification | HighRiskFlag | Urgent-action flag (1 = high risk) |
| Ridge Regression | Regularised regression | StressScore | L2-penalty baseline |
| Lasso Regression | Regularised regression | StressScore | L1 sparsity, feature selection |
| KNN | Classification | RiskCategory | Similarity baseline (k=15) |
| Decision Tree | Classification | RiskCategory | Interpretable baseline (max_depth=6) |
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
WorkHealth AI Pro
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
|   |-- data_loader.py             # synthetic 15k-row generator + CSV cache
|   |-- feature_engineering.py     # Risk/Wellness/BMI/Sleep/AgeGroup
|   |-- rule_engine.py             # deterministic point scoring
|   |-- ml_models.py               # 8 models, train+save+load
|   |-- recommendations.py         # action plan + peer KNN
|   `-- styles.py                  # CSS, KPI strip, sidebar filters
|-- models/                        # joblib + JSON reports (auto-saved)
`-- data/                          # workhealth_data.csv (auto-generated)
```
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
shortcode built into Streamlit 1.36+) for all icons - no emoji are used in
the code or filenames. Material Symbols are vector icons rendered through
Streamlit's markdown engine.

Reference list of icon names: <https://fonts.google.com/icons>.
""")
