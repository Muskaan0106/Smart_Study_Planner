"""
app.py
------
StudyFlow — Flask backend.

All routes return JSON.  The frontend (static/index.html) is served
from the /static directory.

Routes
------
POST /api/schedule          Generate optimised schedule
POST /api/burnout           Run ML burnout prediction
GET  /api/suggestions       Get AI suggestions for current subjects + schedule
GET  /api/progress          Load saved progress
POST /api/progress          Save progress update
GET  /api/sessions          Load Pomodoro session history
POST /api/sessions          Save completed Pomodoro session
DELETE /api/sessions        Clear session history
GET  /api/health            Health check

Running locally:
    python app.py

Running in production (e.g. Render, Railway, Fly.io):
    gunicorn app:app
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime

from scheduler import generate_schedule, get_ai_suggestions, priority_score, DIFF_LABELS
from ml_model import get_predictor

# ─── App setup ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")

# CORS: allows the frontend to call the API.
# In production set ALLOWED_ORIGINS in your environment:
#   export ALLOWED_ORIGINS="https://yourdomain.com"
# Leave unset (or "*") for open access during development.
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, origins=_origins)

# ─── Persistent storage ──────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

os.makedirs(DATA_DIR, exist_ok=True)


def read_json(path: str, default) :
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Helper ──────────────────────────────────────────────────────────
def validate_subjects(subjects: list) -> tuple[list, str | None]:
    """
    Validate and normalise the subjects list from the request body.
    Returns (cleaned_subjects, error_message_or_None).
    """
    if not isinstance(subjects, list) or not subjects:
        return [], "subjects must be a non-empty list"

    cleaned = []
    for i, s in enumerate(subjects):
        name = str(s.get("name", "")).strip()
        if not name:
            return [], f"subject at index {i} is missing a name"

        try:
            days = int(s.get("days", 0))
            if days < 1:
                raise ValueError
        except (TypeError, ValueError):
            return [], f"subject '{name}' has an invalid deadline (must be >= 1 day)"

        try:
            diff = int(s.get("difficulty", 2))
            if diff not in (1, 2, 3):
                raise ValueError
        except (TypeError, ValueError):
            return [], f"subject '{name}' difficulty must be 1, 2, or 3"

        cleaned.append({
            "id":         s.get("id", i),
            "name":       name,
            "days":       days,
            "difficulty": diff,
            "color":      str(s.get("color", "#B85C3A")),
            "diffLabel":  DIFF_LABELS.get(diff, "Medium"),
        })

    return cleaned, None


# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    """Simple health check — used by uptime monitors and deploy pipelines."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


# ── Schedule ──────────────────────────────────────────────────────────

@app.route("/api/schedule", methods=["POST"])
def schedule():
    """
    Generate an optimised study schedule.

    Request body (JSON):
        {
          "subjects": [
            { "id": 1, "name": "Math", "days": 3, "difficulty": 3, "color": "#B85C3A" },
            ...
          ],
          "hours_per_day": 5
        }

    Response:
        {
          "schedule": {
            "1": [ {"subjectId", "name", "color", "diffLabel", "hours", "score"}, ... ],
            "2": [ ... ],
            ...
          },
          "suggestions": [ "string", ... ],
          "priority_scores": { "Math": 1.933, ... }
        }
    """
    body = request.get_json(silent=True) or {}

    subjects, err = validate_subjects(body.get("subjects", []))
    if err:
        return jsonify({"error": err}), 400

    try:
        hours_per_day = float(body.get("hours_per_day", 5))
        if not (0.5 <= hours_per_day <= 16):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "hours_per_day must be between 0.5 and 16"}), 400

    sched  = generate_schedule(subjects, hours_per_day)
    suggestions = get_ai_suggestions(subjects, sched)

    # Priority scores per subject (for display / CV explainability)
    priority_scores = {
        s["name"]: priority_score(s["difficulty"], s["days"])
        for s in subjects
    }

    return jsonify({
        "schedule":        sched,
        "suggestions":     suggestions,
        "priority_scores": priority_scores,
    })


# ── Burnout ───────────────────────────────────────────────────────────

@app.route("/api/burnout", methods=["POST"])
def burnout():
    """
    Run the ML burnout prediction.

    Request body (JSON):
        {
          "sleep":          7.0,   // hours of sleep
          "breaks":         3,     // planned 5-min breaks
          "stress":         5,     // stress level 1–10
          "planned_hours":  5.0,   // how many hours the student plans to study
          "num_subjects":   3      // optional, defaults to 3
        }

    Response:
        {
          "recommended_hours": 5.8,
          "planned_hours":     5.0,
          "risk_pct":          43,
          "risk_level":        "Moderate",
          "over_limit":        false,
          "message":           "Your 5h plan is within...",
          "tips":              ["...", "..."],
          "coefficients":      {"sleep_hours": 0.65, ...},
          "model_info":        "Linear Regression · 600 synthetic samples · 4 features"
        }
    """
    body = request.get_json(silent=True) or {}

    try:
        sleep         = float(body.get("sleep", 7))
        breaks        = int(body.get("breaks", 3))
        stress        = int(body.get("stress", 5))
        planned_hours = float(body.get("planned_hours", 5))
        num_subjects  = int(body.get("num_subjects", 3))

        # Range validation
        if not (1 <= sleep <= 24):   raise ValueError("sleep")
        if not (0 <= breaks <= 20):  raise ValueError("breaks")
        if not (1 <= stress <= 10):  raise ValueError("stress")
        if not (0 < planned_hours <= 20): raise ValueError("planned_hours")
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    predictor = get_predictor()
    result    = predictor.predict_with_planned(
        sleep=sleep,
        breaks=breaks,
        stress=stress,
        planned_hours=planned_hours,
        num_subjects=num_subjects,
    )

    result["model_info"] = (
        "Linear Regression · 600 synthetic training samples · "
        "4 features: sleep, breaks, stress, subject count"
    )

    return jsonify(result)


# ── AI Suggestions ────────────────────────────────────────────────────

@app.route("/api/suggestions", methods=["POST"])
def suggestions():
    """
    Get AI suggestions for the current subjects + schedule.

    Request body (JSON):
        { "subjects": [...], "schedule": {...} }

    Response:
        { "suggestions": ["...", "..."] }
    """
    body = request.get_json(silent=True) or {}

    subjects, err = validate_subjects(body.get("subjects", []))
    if err:
        return jsonify({"error": err}), 400

    sched = body.get("schedule", {})
    suggs = get_ai_suggestions(subjects, sched)

    return jsonify({"suggestions": suggs})


# ── Progress ──────────────────────────────────────────────────────────

@app.route("/api/progress", methods=["GET"])
def get_progress():
    """Return saved progress dict { subject_id: pct }."""
    data = read_json(PROGRESS_FILE, {})
    return jsonify({"progress": data})


@app.route("/api/progress", methods=["POST"])
def save_progress():
    """
    Upsert progress for one or many subjects.

    Request body:
        { "progress": { "101": 60, "102": 30 } }
    """
    body = request.get_json(silent=True) or {}
    incoming = body.get("progress", {})

    if not isinstance(incoming, dict):
        return jsonify({"error": "progress must be an object"}), 400

    existing = read_json(PROGRESS_FILE, {})
    existing.update({str(k): int(v) for k, v in incoming.items()})
    write_json(PROGRESS_FILE, existing)

    return jsonify({"saved": True, "progress": existing})


# ── Pomodoro sessions ─────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """Return session history list."""
    data = read_json(SESSIONS_FILE, [])
    return jsonify({"sessions": data})


@app.route("/api/sessions", methods=["POST"])
def save_session():
    """
    Record a completed Pomodoro session.

    Request body:
        {
          "type":    "focus" | "break",
          "subject": "Math",
          "minutes": 25
        }
    """
    body = request.get_json(silent=True) or {}

    session = {
        "type":      body.get("type", "focus"),
        "subject":   str(body.get("subject", "General")),
        "minutes":   int(body.get("minutes", 25)),
        "timestamp": datetime.utcnow().isoformat(),
    }

    history = read_json(SESSIONS_FILE, [])
    history.insert(0, session)          # newest first
    history = history[:200]             # keep last 200
    write_json(SESSIONS_FILE, history)

    return jsonify({"saved": True, "session": session})


@app.route("/api/sessions", methods=["DELETE"])
def clear_sessions():
    write_json(SESSIONS_FILE, [])
    return jsonify({"cleared": True})


# ─────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # host="0.0.0.0" makes the server reachable on your local network
    # and is required by most cloud platforms.
    app.run(host="0.0.0.0", port=port, debug=debug)
