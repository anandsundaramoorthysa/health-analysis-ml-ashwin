"""Deterministic rule-based health risk scoring engine.

Mirrors the spec from the project abstract: each parameter contributes a
weighted point penalty; the total maps to Low / Moderate / High risk.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleResult:
    points: int
    label: str
    note: str


def score_sleep(h: float) -> RuleResult:
    if h >= 7:
        return RuleResult(0, "OK", "Healthy sleep duration")
    if h >= 5:
        return RuleResult(10, "Watch", "Sleep is below the 7h healthy mark")
    return RuleResult(20, "Critical", "Severe sleep deprivation (<5h)")


def score_work_hours(h: float) -> RuleResult:
    if h <= 9:
        return RuleResult(0, "OK", "Reasonable workday length")
    if h <= 12:
        return RuleResult(10, "Watch", "Long workday — risk of fatigue")
    return RuleResult(20, "Critical", "Excessive overwork (>12h/day)")


def score_activity(min_per_day: int) -> RuleResult:
    if min_per_day >= 150:
        return RuleResult(0, "OK", "Meets WHO weekly activity target")
    if min_per_day >= 30:
        return RuleResult(10, "Watch", "Active but below recommended levels")
    return RuleResult(20, "Critical", "Sedentary lifestyle")


def score_bmi(bmi: float) -> RuleResult:
    if 18.5 <= bmi < 25:
        return RuleResult(0, "OK", "Healthy BMI")
    if 25 <= bmi < 30:
        return RuleResult(5, "Watch", "Overweight range")
    if bmi >= 30:
        return RuleResult(15, "Critical", "Obese range — health risk")
    return RuleResult(5, "Watch", "Underweight range")


def score_stress(score: float) -> RuleResult:
    if score < 20:
        return RuleResult(0, "OK", "Stress within manageable range")
    if score <= 30:
        return RuleResult(10, "Watch", "Elevated stress")
    return RuleResult(20, "Critical", "High chronic stress")


def score_burnout(level: float) -> RuleResult:
    if level < 5:
        return RuleResult(0, "OK", "Burnout under control")
    if level <= 7:
        return RuleResult(8, "Watch", "Mild-to-moderate burnout")
    return RuleResult(15, "Critical", "Severe burnout signals")


def score_screen(h: float) -> RuleResult:
    if h <= 9:
        return RuleResult(0, "OK", "Reasonable screen exposure")
    if h <= 12:
        return RuleResult(8, "Watch", "Heavy screen use")
    return RuleResult(15, "Critical", "Excessive screen exposure")


def score_water(litres: float) -> RuleResult:
    if litres >= 2.0:
        return RuleResult(0, "OK", "Well hydrated")
    if litres >= 1.5:
        return RuleResult(5, "Watch", "Hydration below recommended 2L")
    return RuleResult(10, "Critical", "Dehydration risk")


def score_caffeine(cups: int) -> RuleResult:
    if cups <= 3:
        return RuleResult(0, "OK", "Caffeine within healthy range")
    if cups <= 6:
        return RuleResult(5, "Watch", "High caffeine — may affect sleep")
    return RuleResult(10, "Critical", "Excessive caffeine intake")


def score_breaks(per_day: int) -> RuleResult:
    if per_day >= 4:
        return RuleResult(0, "OK", "Adequate work breaks")
    if per_day >= 2:
        return RuleResult(5, "Watch", "Few breaks — fatigue risk")
    return RuleResult(10, "Critical", "Almost no breaks during work")


def score_job_sat(score: int) -> RuleResult:
    if score >= 7:
        return RuleResult(0, "OK", "Strong job satisfaction")
    if score >= 4:
        return RuleResult(0, "Neutral", "Average job satisfaction")
    return RuleResult(10, "Critical", "Low job satisfaction — burnout risk")


SCORERS = {
    "Sleep": score_sleep,
    "Work Hours": score_work_hours,
    "Activity": score_activity,
    "BMI": score_bmi,
    "Stress": score_stress,
    "Burnout": score_burnout,
    "Screen Time": score_screen,
    "Water Intake": score_water,
    "Caffeine": score_caffeine,
    "Breaks": score_breaks,
    "Job Satisfaction": score_job_sat,
}


def assess(payload: dict) -> dict:
    """Run every rule on the payload dict, return totals + per-rule breakdown."""
    breakdown = {}
    total = 0
    breakdown["Sleep"] = score_sleep(payload.get("SleepHours", 7))
    breakdown["Work Hours"] = score_work_hours(payload.get("WorkHours", 8))
    breakdown["Activity"] = score_activity(payload.get("ActivityMinutes", 60))
    breakdown["BMI"] = score_bmi(payload.get("BMI", 24))
    breakdown["Stress"] = score_stress(payload.get("StressScore", 20))
    breakdown["Burnout"] = score_burnout(payload.get("BurnoutLevel", 5))
    breakdown["Screen Time"] = score_screen(payload.get("ScreenTime", 8))
    breakdown["Water Intake"] = score_water(payload.get("WaterIntakeL", 2.0))
    breakdown["Caffeine"] = score_caffeine(payload.get("CaffeineCups", 3))
    breakdown["Breaks"] = score_breaks(payload.get("BreaksPerDay", 4))
    breakdown["Job Satisfaction"] = score_job_sat(payload.get("JobSatisfaction", 7))

    total = sum(r.points for r in breakdown.values())
    if total < 20:
        category, color = "Low", "#22c55e"
    elif total <= 50:
        category, color = "Moderate", "#f59e0b"
    else:
        category, color = "High", "#ef4444"

    return {
        "total": total,
        "category": category,
        "color": color,
        "breakdown": breakdown,
        "issues": [name for name, r in breakdown.items() if r.points > 0],
        "healthy": [name for name, r in breakdown.items() if r.points == 0],
    }
