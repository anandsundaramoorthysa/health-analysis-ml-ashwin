"""Personalised recommendation engine.

Given a user record + rule-engine result + ML predictions, returns:
- Top-priority issues
- Specific recommended actions
- Comparison to dataset averages
- Lifestyle goals (Sleep, Activity, Hydration, Caffeine)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ACTION_BANK = {
    "Sleep": [
        "Aim for 7–8h of sleep — set a consistent bedtime alarm.",
        "Cut screens 60 min before bed; replace with reading or stretching.",
        "Avoid caffeine after 2 PM if you're sleeping <7h.",
    ],
    "Work Hours": [
        "Block-out a hard end-of-day stop time and protect it.",
        "Use the Pomodoro technique (25 min focus / 5 min break).",
        "Discuss workload with your manager if >12h is routine.",
    ],
    "Activity": [
        "Take a 10-minute walk every 2 hours of work.",
        "Aim for the WHO target of 150 min of moderate activity per week.",
        "Add 2 short bodyweight workouts (push-ups, squats, planks) per day.",
    ],
    "BMI": [
        "Track meals for one week to spot calorie patterns.",
        "Replace sugary drinks with water or unsweetened tea.",
        "Schedule 3 cardio sessions (20–30 min) weekly.",
    ],
    "Stress": [
        "Try 10 min of guided breathing or meditation each morning.",
        "Journal one stressor + one win at the end of each day.",
        "Talk to a counsellor if stress feels persistent.",
    ],
    "Burnout": [
        "Take an unplugged half-day this week — no email, no slack.",
        "Reconnect with a hobby or social activity outside work.",
        "Speak with HR about workload re-balancing options.",
    ],
    "Screen Time": [
        "Apply the 20-20-20 rule: every 20 min look 20 ft away for 20 sec.",
        "Use blue-light filters in the evening.",
        "Set up an hourly stand-and-stretch reminder.",
    ],
    "Water Intake": [
        "Keep a 1-litre bottle on your desk and refill once daily.",
        "Drink a glass of water before each meal and coffee.",
        "Set 4 hydration reminders in your calendar.",
    ],
    "Caffeine": [
        "Cap caffeine at 3 cups/day (~300 mg).",
        "Switch your afternoon coffee for green tea or water.",
        "Watch hidden caffeine in energy drinks and sodas.",
    ],
    "Breaks": [
        "Schedule a 5-min movement break every hour.",
        "Take your full lunch break away from the desk.",
        "Use a break-reminder app (Stretchly, Time Out).",
    ],
    "Job Satisfaction": [
        "List 3 elements you enjoy and 3 that drain you — share with your manager.",
        "Pursue one skill-development goal this quarter.",
        "Consider an internal mobility chat if dissatisfaction persists.",
    ],
}


def recommend(user_row: pd.DataFrame, rule_result: dict, ml_result: dict,
              dataset: pd.DataFrame) -> dict[str, Any]:
    """Build a structured recommendation payload."""
    row = user_row.iloc[0]
    issues = rule_result["issues"]
    breakdown = rule_result["breakdown"]

    # Sort issues by point penalty (worst first)
    sorted_issues = sorted(issues, key=lambda n: breakdown[n].points, reverse=True)
    actions: list[dict] = []
    for name in sorted_issues:
        actions.append({
            "area": name,
            "severity": breakdown[name].label,
            "points": breakdown[name].points,
            "note": breakdown[name].note,
            "tips": ACTION_BANK.get(name, []),
        })

    # Compare with dataset means
    compare_cols = [
        "WorkHours", "SleepHours", "ActivityMinutes", "ScreenTime",
        "CaffeineCups", "WaterIntakeL", "JobSatisfaction",
        "StressScore", "BurnoutLevel", "WellnessScore", "RiskScore",
    ]
    dataset_means = dataset[compare_cols].mean().round(2).to_dict()
    user_vals = {c: float(row[c]) for c in compare_cols}
    delta = {c: round(user_vals[c] - dataset_means[c], 2) for c in compare_cols}

    # Key goals (concrete numeric targets)
    goals = []
    if row["SleepHours"] < 7:
        goals.append({"area": "Sleep", "current": float(row["SleepHours"]),
                       "target": 7.5, "unit": "h"})
    if row["ActivityMinutes"] < 150:
        goals.append({"area": "Activity", "current": int(row["ActivityMinutes"]),
                       "target": 150, "unit": "min/day"})
    if row["WaterIntakeL"] < 2.0:
        goals.append({"area": "Hydration", "current": float(row["WaterIntakeL"]),
                       "target": 2.5, "unit": "L"})
    if row["CaffeineCups"] > 3:
        goals.append({"area": "Caffeine", "current": int(row["CaffeineCups"]),
                       "target": 3, "unit": "cups"})
    if row["WorkHours"] > 9:
        goals.append({"area": "Work Hours", "current": float(row["WorkHours"]),
                       "target": 9, "unit": "h/day"})
    if row["ScreenTime"] > 9:
        goals.append({"area": "Screen Time", "current": float(row["ScreenTime"]),
                       "target": 9, "unit": "h"})

    # Find similar peers (k-nearest) for "people like you" comparison
    feats = ["WorkHours", "SleepHours", "ActivityMinutes",
             "ScreenTime", "CaffeineCups", "WaterIntakeL", "JobSatisfaction"]
    user_vec = np.asarray([float(row[f]) for f in feats], dtype=float)
    ds_arr = dataset[feats].astype(float).to_numpy()
    diffs = pd.Series(((ds_arr - user_vec) ** 2).sum(axis=1),
                       index=dataset.index, dtype=float)
    nearest = dataset.iloc[diffs.nsmallest(50).index]

    peer_summary = {
        "n_peers": len(nearest),
        "peer_avg_risk_score": round(float(nearest["RiskScore"].mean()), 2),
        "peer_avg_wellness": round(float(nearest["WellnessScore"].mean()), 2),
        "peer_high_risk_pct": round(
            float((nearest["RiskCategory"] == "High").mean() * 100), 1),
    }

    return {
        "category": rule_result["category"],
        "color": rule_result["color"],
        "total_points": rule_result["total"],
        "actions": actions,
        "healthy": rule_result["healthy"],
        "dataset_means": dataset_means,
        "user_vals": user_vals,
        "delta": delta,
        "goals": goals,
        "ml_summary": ml_result,
        "peer": peer_summary,
    }
