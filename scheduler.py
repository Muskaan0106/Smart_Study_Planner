"""
scheduler.py
------------
Core scheduling algorithm for StudyFlow.

Priority formula:
    score = (difficulty_weight * difficulty) + (urgency_weight * urgency)
    urgency = 1 / days_until_deadline

Time allocation:
    Each subject's daily hours = (score / total_score) * hours_per_day
    Minimum 0.5h per subject to avoid tiny slivers.
    Hours are then normalised so the total always equals hours_per_day exactly.
"""

from __future__ import annotations
from typing import Any


# ─── Weights ────────────────────────────────────────────────────────
DIFFICULTY_WEIGHT = 0.6
URGENCY_WEIGHT    = 0.4

# Difficulty label → numeric value sent by the frontend
DIFF_LABELS = {3: "Hard", 2: "Medium", 1: "Easy"}

# Minimum hours allocated to any subject in a given day
MIN_HOURS = 0.5


def priority_score(difficulty: int, days: int) -> float:
    """
    Calculate the priority score for one subject.

    Args:
        difficulty: 1 (Easy), 2 (Medium), 3 (Hard)
        days:       Days until deadline (must be >= 1)

    Returns:
        Float priority score — higher means more urgent/important.
    """
    if days < 1:
        days = 1
    urgency = 1 / days
    return round(DIFFICULTY_WEIGHT * difficulty + URGENCY_WEIGHT * urgency, 4)


def allocate_hours(subjects: list[dict], hours_per_day: float) -> list[dict]:
    """
    Given a list of subjects (each with a score), distribute hours_per_day
    proportionally by score.  Returns a new list with an 'hours' key added.

    Args:
        subjects:     List of dicts with at least {'score': float, ...}
        hours_per_day: Total study hours available today

    Returns:
        List of dicts with 'hours' key.  Sum of hours == hours_per_day.
    """
    if not subjects:
        return []

    total_score = sum(s["score"] for s in subjects)

    # Raw proportional allocation
    result = []
    for s in subjects:
        raw = (s["score"] / total_score) * hours_per_day
        hours = max(MIN_HOURS, round(raw, 1))
        result.append({**s, "hours": hours})

    # Normalise so the total is exactly hours_per_day
    current_total = sum(r["hours"] for r in result)
    if current_total > 0:
        factor = hours_per_day / current_total
        for r in result:
            r["hours"] = round(max(MIN_HOURS, r["hours"] * factor), 1)

    return result


def generate_schedule(
    subjects: list[dict],
    hours_per_day: float = 5.0,
) -> dict[str, list[dict]]:
    """
    Build a day-by-day schedule from today (day 1) through the furthest deadline.

    Each day only includes subjects whose deadline hasn't passed yet.
    Subjects are sorted by priority score descending within each day.

    Args:
        subjects:     List of subject dicts:
                        {id, name, days, difficulty, color, diffLabel}
        hours_per_day: How many hours the student has available per day.

    Returns:
        Dict keyed by string day number ("1", "2", …):
            [{"subjectId", "name", "color", "diffLabel", "hours", "score"}, …]
    """
    if not subjects:
        return {}

    max_days = max(s["days"] for s in subjects)
    schedule: dict[str, list[dict]] = {}

    for day in range(1, max_days + 1):
        # Subjects still within their deadline window
        available = [s for s in subjects if s["days"] >= day]
        if not available:
            continue

        # Attach scores
        scored = [
            {
                **s,
                "score": priority_score(s["difficulty"], s["days"] - day + 1),
                "diffLabel": DIFF_LABELS.get(s["difficulty"], "Medium"),
            }
            for s in available
        ]

        # Sort highest priority first
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Allocate hours
        allocated = allocate_hours(scored, hours_per_day)

        schedule[str(day)] = [
            {
                "subjectId": a["id"],
                "name":      a["name"],
                "color":     a["color"],
                "diffLabel": a["diffLabel"],
                "hours":     a["hours"],
                "score":     a["score"],
            }
            for a in allocated
        ]

    return schedule


def get_ai_suggestions(subjects: list[dict], schedule: dict) -> list[str]:
    """
    Generate rule-based AI suggestions based on subjects and current schedule.

    Returns a list of suggestion strings (2-4 items).
    """
    suggestions = []

    if not subjects:
        return ["Add subjects to get personalised recommendations."]

    # Urgent subjects (deadline <= 2 days)
    urgent = [s for s in subjects if s["days"] <= 2]
    if urgent:
        names = ", ".join(s["name"] for s in urgent)
        suggestions.append(
            f"⚠ Deadline alert — {names} {'is' if len(urgent)==1 else 'are'} "
            f"due in {urgent[0]['days']} day{'s' if urgent[0]['days']!=1 else ''}. "
            "Front-load these sessions today."
        )

    # Hard subjects
    hard = [s for s in subjects if s["difficulty"] == 3]
    if hard:
        suggestions.append(
            f"📚 {hard[0]['name']} is your hardest subject. "
            "Tackle it first while your focus is sharpest."
        )

    # Today's top priority
    if schedule.get("1"):
        top = schedule["1"][0]
        suggestions.append(
            f"⚡ Today's top priority: {top['name']} "
            f"({top['hours']}h recommended, priority score {top['score']})."
        )

    # General study tip (rotates based on subject count)
    tips = [
        "Use active recall instead of re-reading — test yourself after each topic.",
        "Space your sessions across multiple days rather than cramming.",
        "Take a 5-minute break every 50 minutes to maintain concentration quality.",
        "Review the previous session's notes at the start of each new session.",
    ]
    suggestions.append(tips[len(subjects) % len(tips)])

    return suggestions[:4]
