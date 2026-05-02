"""Feature engineering: derives Risk Score, Wellness Score, and category labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIN_MAP = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}


def compute_risk_score(row) -> float:
    """Composite risk: high stress, low sleep/activity, high pain/burnout drive it up."""
    sleep_deficit = max(0, 7 - row["SleepHours"])
    activity_deficit = max(0, 150 - row["ActivityMinutes"])
    pain = PAIN_MAP.get(row["PainLevel"], 0) if isinstance(row["PainLevel"], str) else row["PainLevel"]
    return (
        0.35 * row["StressScore"]
        + 3.5 * sleep_deficit
        + 0.06 * activity_deficit
        + 5.0 * pain
        + 0.8 * row["BurnoutLevel"]
        + 0.6 * max(0, row["WorkHours"] - 9)
        + 0.4 * max(0, row["CaffeineCups"] - 3)
    )


def compute_wellness_score(row) -> float:
    """Positive lifestyle indicator (higher = healthier)."""
    return (
        5.0 * row["SleepHours"]
        + 0.10 * row["ActivityMinutes"]
        + 3.0 * row["JobSatisfaction"]
        + 4.0 * row["WaterIntakeL"]
        - 0.5 * row["StressScore"]
        - 2.0 * row["BurnoutLevel"]
        - 0.3 * max(0, row["WorkHours"] - 8)
    )


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def sleep_category(hours: float) -> str:
    if hours < 5:
        return "Deprived"
    if hours < 7:
        return "Adequate"
    return "Well-Rested"


def age_group(age: int) -> str:
    if age < 30:
        return "20s"
    if age < 40:
        return "30s"
    if age < 50:
        return "40s"
    return "50+"


def risk_category(score: float) -> str:
    if score < 20:
        return "Low"
    if score <= 50:
        return "Moderate"
    return "High"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features to the dataframe."""
    df = df.copy()
    df["PainScore"] = df["PainLevel"].map(PAIN_MAP).fillna(0).astype(int)
    df["RiskScore"] = df.apply(compute_risk_score, axis=1).round(1)
    df["WellnessScore"] = df.apply(compute_wellness_score, axis=1).round(1)
    df["BMICategory"] = df["BMI"].apply(bmi_category)
    df["SleepCategory"] = df["SleepHours"].apply(sleep_category)
    df["AgeGroup"] = df["Age"].apply(age_group)
    df["RiskCategory"] = df["RiskScore"].apply(risk_category)
    df["HighRiskFlag"] = (df["RiskCategory"] == "High").astype(int)
    return df


# ---------- Single user record (used by Predict page) ----------

def build_user_record(
    work_hours: float,
    sleep_hours: float,
    coffee_cups: int,
    activity_min: int,
    screen_time: float,
    water_l: float,
    job_sat: int,
    age: int = 30,
    bmi: float = 24.0,
    burnout_level: float = 5.0,
    pain_level: str = "Mild",
) -> pd.DataFrame:
    """Convert 7 user inputs (+ defaults) into a one-row enriched dataframe."""
    # Estimate stress score from inputs using same approximate relationship
    stress_est = np.clip(
        18
        + 0.9 * (work_hours - 8)
        + 0.7 * (screen_time - 8)
        - 1.1 * (sleep_hours - 7)
        + 0.012 * (150 - activity_min)
        + 0.6 * (coffee_cups - 3)
        - 0.7 * (job_sat - 6),
        0, 40,
    )
    burnout_est = np.clip(
        3.0
        + 0.18 * stress_est
        + 0.20 * (work_hours - 8)
        - 0.15 * (sleep_hours - 7)
        - 0.12 * (job_sat - 6),
        0, 10,
    )
    record = pd.DataFrame([{
        "EmployeeID": "USER001",
        "Age": age,
        "Gender": "Other",
        "Department": "Software Engineering",
        "WorkMode": "Hybrid",
        "ExperienceYears": max(0, age - 22),
        "WorkHours": work_hours,
        "ScreenTime": screen_time,
        "SleepHours": sleep_hours,
        "ActivityMinutes": activity_min,
        "CaffeineCups": coffee_cups,
        "WaterIntakeL": water_l,
        "BreaksPerDay": 4,
        "BMI": bmi,
        "JobSatisfaction": job_sat,
        "StressScore": round(float(stress_est), 1),
        "BurnoutLevel": round(float(burnout_est), 1),
        "PainLevel": pain_level,
    }])
    return enrich(record)
