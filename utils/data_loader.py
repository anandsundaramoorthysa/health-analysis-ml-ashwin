"""Synthetic IT-Professional health dataset generator and loader.

Generates a realistic 15,000-row dataset of IT employees across WFH / Office /
Hybrid work modes with correlated lifestyle, body, and wellbeing metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "workhealth_data.csv"

DEPARTMENTS = [
    "Software Engineering", "Data Science", "DevOps", "QA & Testing",
    "Product Management", "UI/UX Design", "Cybersecurity", "IT Support",
    "Cloud Engineering", "Project Management",
]
WORK_MODES = ["WFH", "Office", "Hybrid"]
GENDERS = ["Male", "Female", "Other"]
PAIN_LEVELS = ["None", "Mild", "Moderate", "Severe"]


def _truncated_normal(rng, mu, sigma, low, high, size):
    out = rng.normal(mu, sigma, size)
    return np.clip(out, low, high)


def generate_dataset(n: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """Generate a correlated synthetic dataset of IT professionals."""
    rng = np.random.default_rng(seed)

    age = _truncated_normal(rng, 33, 8, 21, 60, n).astype(int)
    gender = rng.choice(GENDERS, size=n, p=[0.62, 0.36, 0.02])
    work_mode = rng.choice(WORK_MODES, size=n, p=[0.38, 0.30, 0.32])
    department = rng.choice(DEPARTMENTS, size=n)
    experience_years = np.clip(age - rng.integers(20, 24, n), 0, 35)

    # Work hours influenced by mode (WFH tends to be longer)
    base_work = np.where(work_mode == "WFH", 9.8,
                np.where(work_mode == "Office", 8.6, 9.1))
    # Wider variance + a tail of overworkers to get realistic high-risk records
    work_hours = np.round(_truncated_normal(rng, base_work, 2.4, 5, 16, n), 1)

    # Screen time strongly correlates with work hours
    screen_time = np.round(np.clip(work_hours * 0.95
                                   + rng.normal(1.5, 1.2, n), 4, 16), 1)

    # Sleep — inverse relationship with workload (wider variance for spread)
    sleep = np.round(np.clip(8.2 - 0.22 * (work_hours - 8)
                             + rng.normal(0, 1.3, n), 3, 11), 1)

    # Activity — Office workers walk to work, WFH less active
    base_activity = np.where(work_mode == "WFH", 45,
                    np.where(work_mode == "Office", 70, 60))
    activity = np.clip(base_activity + rng.normal(0, 55, n), 0, 240).astype(int)

    # Caffeine — correlates with longer work hours
    caffeine = np.round(np.clip(2.0 + 0.25 * (work_hours - 8)
                                + rng.normal(0, 1.4, n), 0, 10), 0).astype(int)

    # Water intake (litres)
    water = np.round(_truncated_normal(rng, 2.1, 0.7, 0.4, 5.0, n), 1)

    # Breaks
    breaks = np.clip(5 - 0.25 * (work_hours - 8)
                     + rng.normal(0, 1.5, n), 0, 12).astype(int)

    # BMI
    bmi = np.round(_truncated_normal(rng, 25.4, 4.2, 16, 42, n), 1)

    # Job satisfaction — depends on burnout drivers
    job_sat = np.clip(8.5 - 0.18 * (work_hours - 8)
                      + 0.12 * (sleep - 7)
                      - 0.04 * (caffeine - 3)
                      + rng.normal(0, 1.4, n), 1, 10).round(0).astype(int)

    # Inject 8% of "workaholic" employees with extreme stress patterns
    extreme_mask = rng.random(n) < 0.08
    extreme_boost = np.where(extreme_mask, rng.uniform(8, 15, n), 0)

    # Stress score (0–40) — derived from lifestyle pressure
    stress_raw = (
        1.1 * (work_hours - 8)
        + 0.8 * (screen_time - 8)
        - 1.4 * (sleep - 7)
        + 0.015 * (150 - activity)
        + 0.7 * (caffeine - 3)
        - 0.8 * (job_sat - 6)
        + 0.12 * (bmi - 24)
        + rng.normal(8, 5.0, n)
        + extreme_boost
    )
    stress_score = np.clip(stress_raw + 18, 0, 40).round(1)

    # Burnout level (0–10)
    burnout_raw = (
        0.22 * stress_score
        + 0.25 * (work_hours - 8)
        - 0.18 * (sleep - 7)
        - 0.15 * (job_sat - 6)
        + rng.normal(0, 1.4, n)
        + 1.5 * extreme_mask
    )
    burnout = np.clip(burnout_raw + 2.0, 0, 10).round(1)

    # Pain level — increases with screen time + low activity
    pain_score_num = np.clip(
        0.10 * (screen_time - 8)
        + 0.012 * (150 - activity)
        + 0.05 * (bmi - 24)
        + rng.normal(0.6, 0.7, n), 0, 4)
    pain = pd.cut(pain_score_num, bins=[-0.1, 0.6, 1.5, 2.5, 4.1],
                  labels=PAIN_LEVELS).astype(str)

    df = pd.DataFrame({
        "EmployeeID": [f"EMP{i:05d}" for i in range(1, n + 1)],
        "Age": age,
        "Gender": gender,
        "Department": department,
        "WorkMode": work_mode,
        "ExperienceYears": experience_years,
        "WorkHours": work_hours,
        "ScreenTime": screen_time,
        "SleepHours": sleep,
        "ActivityMinutes": activity,
        "CaffeineCups": caffeine,
        "WaterIntakeL": water,
        "BreaksPerDay": breaks,
        "BMI": bmi,
        "JobSatisfaction": job_sat,
        "StressScore": stress_score,
        "BurnoutLevel": burnout,
        "PainLevel": pain,
    })

    return df


@st.cache_data(show_spinner=False)
def load_data(force_regen: bool = False) -> pd.DataFrame:
    """Load the dataset, generating and caching it on first call."""
    DATA_DIR.mkdir(exist_ok=True)
    if force_regen or not DATA_FILE.exists():
        df = generate_dataset()
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
    from utils.feature_engineering import enrich
    return enrich(df)


def get_data_path() -> str:
    return str(DATA_FILE)
