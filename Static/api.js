/**
 * api.js — StudyFlow frontend API client
 *
 * All calls go to /api/* (same origin, so no CORS needed in production).
 * Falls back gracefully if the backend is unreachable (localStorage mode).
 *
 * Key decision: every function returns a plain JS object so the rest of
 * the frontend doesn't need to know whether data came from the API or
 * the local fallback — it just uses the result.
 */

const API_BASE = "";   // same origin — change to "http://localhost:5000" for local dev

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ─── Public API ──────────────────────────────────────────────────────

/**
 * Generate schedule from subjects.
 * @param {Array}  subjects      Array of subject objects
 * @param {number} hoursPerDay   How many hours the student has today
 * @returns {{ schedule, suggestions, priority_scores }}
 */
export async function generateSchedule(subjects, hoursPerDay = 5) {
  return apiPost("/api/schedule", { subjects, hours_per_day: hoursPerDay });
}

/**
 * Run burnout prediction.
 * @param {{ sleep, breaks, stress, planned_hours, num_subjects }} params
 * @returns {{ recommended_hours, risk_pct, risk_level, over_limit, message, tips, coefficients }}
 */
export async function predictBurnout(params) {
  return apiPost("/api/burnout", params);
}

/**
 * Get AI suggestions for current subjects + schedule.
 * @param {Array}  subjects
 * @param {Object} schedule
 * @returns {{ suggestions: string[] }}
 */
export async function getSuggestions(subjects, schedule) {
  return apiPost("/api/suggestions", { subjects, schedule });
}

/**
 * Load saved progress from the server.
 * @returns {{ progress: { [id]: number } }}
 */
export async function loadProgress() {
  return apiGet("/api/progress");
}

/**
 * Save progress update.
 * @param {{ [id]: number }} progress
 */
export async function saveProgress(progress) {
  return apiPost("/api/progress", { progress });
}

/**
 * Load Pomodoro session history.
 * @returns {{ sessions: Array }}
 */
export async function loadSessions() {
  return apiGet("/api/sessions");
}

/**
 * Record a completed Pomodoro session.
 * @param {{ type: "focus"|"break", subject: string, minutes: number }} session
 */
export async function saveSession(session) {
  return apiPost("/api/sessions", session);
}

/**
 * Clear all Pomodoro session history.
 */
export async function clearSessions() {
  return apiDelete("/api/sessions");
}

/**
 * Health check — useful for detecting whether the backend is up.
 * @returns {boolean}
 */
export async function isBackendAlive() {
  try {
    await apiGet("/api/health");
    return true;
  } catch {
    return false;
  }
}
