"""
ml_model.py
-----------
Burnout prediction model using scikit-learn Linear Regression.

The model is trained on synthetic data that encodes domain knowledge:
  - More sleep  → higher safe study limit
  - More breaks → higher safe study limit
  - More stress → lower safe study limit
  - More subjects → slightly lower safe study limit (cognitive load)

In production this would be trained on real user data.
Synthetic generation with controlled noise keeps it scientifically plausible.

Usage:
    from ml_model import BurnoutPredictor
    model = BurnoutPredictor()            # trains on first call
    result = model.predict(sleep=7, study_hours=5, breaks=3, stress=4)
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


# ─── True coefficient vector (domain knowledge) ─────────────────────
# recommended_hours ≈ intercept + w_sleep*sleep + w_breaks*breaks - w_stress*stress
_INTERCEPT   =  1.2
_W_SLEEP     =  0.65   # each extra hour of sleep adds 0.65h capacity
_W_BREAKS    =  0.35   # each break adds 0.35h capacity
_W_STRESS    = -0.28   # each stress point removes 0.28h capacity
_W_SUBJECTS  = -0.05   # more subjects slightly lowers sustainable limit
_NOISE_STD   =  0.4    # Gaussian noise added to labels


def _generate_synthetic_data(n_samples: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic training dataset.

    Features (X): [sleep_hours, breaks, stress_level, num_subjects]
    Label   (y): recommended_study_hours

    Ranges chosen to reflect realistic student conditions.
    """
    rng = np.random.RandomState(42)

    sleep      = rng.uniform(3.0, 10.0,  n_samples)    # 3 – 10 hours
    breaks     = rng.randint(0, 11,      n_samples).astype(float)  # 0 – 10
    stress     = rng.randint(1, 11,      n_samples).astype(float)  # 1 – 10
    n_subjects = rng.randint(1, 9,       n_samples).astype(float)  # 1 – 8

    # Ground-truth label using domain formula + noise
    y = (
        _INTERCEPT
        + _W_SLEEP     * sleep
        + _W_BREAKS    * breaks
        + _W_STRESS    * stress
        + _W_SUBJECTS  * n_subjects
        + rng.normal(0, _NOISE_STD, n_samples)
    )

    # Clamp to realistic range: 1h – 12h
    y = np.clip(y, 1.0, 12.0)

    X = np.column_stack([sleep, breaks, stress, n_subjects])
    return X, y


class BurnoutPredictor:
    """
    Wraps a scikit-learn LinearRegression pipeline.

    The pipeline standardises features (zero mean, unit variance) before
    fitting, which improves numerical stability and makes coefficients
    more interpretable.
    """

    FEATURE_NAMES = ["sleep_hours", "breaks", "stress_level", "num_subjects"]

    def __init__(self):
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ])
        self._trained = False
        self._train()

    def _train(self):
        X, y = _generate_synthetic_data()
        self._pipeline.fit(X, y)
        self._trained = True

    def predict(
        self,
        sleep: float,
        breaks: int,
        stress: int,
        num_subjects: int = 3,
    ) -> dict:
        """
        Predict safe study limit and return a rich result dict.

        Args:
            sleep:        Sleep hours last night (3–10)
            breaks:       Number of planned 5-min breaks (0–10)
            stress:       Stress level 1–10
            num_subjects: How many subjects the student is managing

        Returns:
            {
              "recommended_hours": float,
              "risk_pct":          int,      # 0–100
              "risk_level":        str,      # "Low" / "Moderate" / "High"
              "coefficients":      dict,     # for CV / explainability
              "tips":              list[str]
            }
        """
        X = np.array([[
            float(sleep),
            float(breaks),
            float(stress),
            float(num_subjects),
        ]])

        raw = float(self._pipeline.predict(X)[0])
        recommended = round(max(1.0, min(12.0, raw)), 1)

        # Risk = how close the student is to their limit
        # (populated by the route using the actual planned hours)
        return {
            "recommended_hours": recommended,
            "coefficients": self._get_coefficients(),
        }

    def predict_with_planned(
        self,
        sleep: float,
        breaks: int,
        stress: int,
        planned_hours: float,
        num_subjects: int = 3,
    ) -> dict:
        """
        Full prediction including burnout risk vs. the student's planned hours.
        """
        base = self.predict(sleep, breaks, stress, num_subjects)
        rec  = base["recommended_hours"]

        risk_pct   = min(100, int((planned_hours / rec) * 50))
        risk_level = "Low" if risk_pct < 40 else "Moderate" if risk_pct < 70 else "High"
        over_limit = planned_hours > rec

        tips = self._generate_tips(sleep, breaks, stress, planned_hours, rec, over_limit)

        # Warning flags exist even when hours are within limit.
        # Used to prevent contradictory messages like "great conditions" when sleep=5h.
        warning_flags = (sleep < 6) or (stress >= 7) or (breaks < 2)

        if over_limit:
            message = (
                f"You're planning {planned_hours}h but your body can safely handle "
                f"~{rec}h today given your sleep and stress levels. "
                "Pushing past that raises burnout risk significantly."
            )
        elif warning_flags:
            message = (
                f"Your {planned_hours}h plan fits within today's limit, but "
                "poor sleep or high stress means focus quality will drop faster "
                "than usual. Take breaks seriously and don't skip them."
            )
        else:
            message = (
                f"Your {planned_hours}h plan looks solid for today. "
                f"You're well within your safe limit of {rec}h with "
                "good sleep and manageable stress — good conditions to work."
            )

        return {
            "recommended_hours": rec,
            "planned_hours":     round(planned_hours, 1),
            "risk_pct":          risk_pct,
            "risk_level":        risk_level,
            "over_limit":        over_limit,
            "warning_flags":     warning_flags,
            "message":           message,
            "tips":              tips,
            "coefficients":      base["coefficients"],
        }

    def _get_coefficients(self) -> dict:
        """Return model coefficients — useful for CV / explainability display."""
        model   = self._pipeline.named_steps["model"]
        scaler  = self._pipeline.named_steps["scaler"]
        # Un-scale coefficients back to original feature units
        coefs   = model.coef_ / scaler.scale_
        return {
            name: round(float(c), 4)
            for name, c in zip(self.FEATURE_NAMES, coefs)
        }

    @staticmethod
    def _generate_tips(
        sleep: float,
        breaks: int,
        stress: int,
        planned: float,
        rec: float,
        over_limit: bool,
    ) -> list[str]:
        """
        Generate tips that are NEVER contradictory to the input conditions.
        Rules:
          - If sleep < 6, ALWAYS warn about sleep — never say "conditions look great".
          - If stress >= 7, ALWAYS address stress — never say "go hard today".
          - The positive/encouraging tip only appears when ALL conditions are actually good.
          - over_limit tips are always shown when applicable, regardless of other flags.
        """
        tips = []

        # ── Sleep ──────────────────────────────────────────────────────
        if sleep < 5:
            tips.append(
                "💤 Critical: under 5h sleep severely impairs memory encoding "
                "and concentration. A 20-min nap would help more than extra study time."
            )
        elif sleep < 6:
            tips.append(
                "💤 Under 6h sleep detected — cognitive performance drops noticeably. "
                "Prioritise shorter, high-focus sessions over long grinding ones today."
            )
        elif sleep < 7:
            tips.append(
                "😴 Slightly below ideal sleep (7–9h is optimal). "
                "You can still study effectively — just keep sessions under 90 min each."
            )

        # ── Stress ─────────────────────────────────────────────────────
        if stress >= 9:
            tips.append(
                "🧘 Very high stress (9–10). Studying while this stressed has poor ROI. "
                "Try 5 min of 4-7-8 breathing first: inhale 4s, hold 7s, exhale 8s."
            )
        elif stress >= 7:
            tips.append(
                "🧘 High stress detected — cortisol actively impairs memory formation. "
                "A short walk or breathing exercise before studying will improve retention."
            )

        # ── Breaks ─────────────────────────────────────────────────────
        if breaks < 1:
            tips.append(
                "☕ No breaks planned — this will sharply reduce focus quality after ~45 min. "
                "Add at least 2 five-minute breaks using the Pomodoro timer."
            )
        elif breaks < 2:
            tips.append(
                "☕ Only 1 break planned. Add a second — the brain consolidates information "
                "during downtime, not just during active study."
            )

        # ── Over limit ─────────────────────────────────────────────────
        if over_limit:
            tips.append(
                f"⏱ {planned}h exceeds today's safe limit. Split into two separate blocks "
                "with at least a 30-min break between them — or cut the second block entirely."
            )

        # ── Positive tip — ONLY when conditions genuinely support it ───
        all_clear = sleep >= 7 and stress <= 5 and breaks >= 2 and not over_limit
        if all_clear:
            tips.append(
                "✅ All conditions are good — solid sleep, manageable stress, breaks planned. "
                "Start with your hardest subject while focus is at its peak."
            )

        # ── Filler if nothing flagged ───────────────────────────────────
        if not tips:
            tips.append(
                "📖 Interleave subjects rather than blocking them — alternating topics "
                "improves long-term retention compared to single-subject marathons."
            )

        return tips[:3]


# ─── Module-level singleton (instantiated once on import) ─────────────
_predictor: BurnoutPredictor | None = None


def get_predictor() -> BurnoutPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BurnoutPredictor()
    return _predictor