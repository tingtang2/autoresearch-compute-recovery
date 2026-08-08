"""
Reviewer: Context-aware execution analysis.

Key principle (aligned with experiments/DEBUGGING_CONSULTANT_DESIGN.md):
the reviewer should see the SAME curated context as the actor saw, so it can
attribute failures to plan vs implementation and provide sharper bug diagnoses.

This module intentionally returns the same schema as the legacy reviewer
(`is_bug`, `summary`, `metric`, `lower_is_better`) for drop-in integration.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, cast

from .backend import FunctionSpec, query
from .utils.response import wrap_code

logger = logging.getLogger("aide")


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": (
                    "true if the run failed OR if required outputs are missing/invalid/inconsistent.\n"
                    "Required tool fields you must return: `summary` (non-empty string), "
                    "`metric` (number when non-buggy; null if unknown/buggy), "
                    "`lower_is_better` (boolean).\n"
                    "Sanity checks for `metric`: must be finite. Treat metric equal to exactly 0 or 1 as invalid/placeholder."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "If there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) "
                    "describing the empirical findings."
                ),
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the validation metric; else null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized; false if it should be maximized.",
            },
            "cv_folds": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "description": (
                    "Extract ALL cross-validation fold scores from the output. "
                    "Look for patterns like: 'cv_folds': [0.72, 0.73, 0.72] or "
                    "'Fold 1: 0.72', 'Fold 2: 0.73', etc. "
                    "Must extract ALL fold scores or return null if CV was not performed or fold scores are incomplete."
                ),
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better", "cv_folds"],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Reviewer:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        logger.info("📋 Reviewer initialized (model=%s)", model)

    def review(
        self,
        task_desc: str,
        code: str,
        term_out: str,
        *,
        actor_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        prompt: dict[str, Any] = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output. "
                "Determine if there were any bugs and report empirical findings."
            ),
            "Task description": task_desc,
        }
        if actor_context:
            prompt["Actor Context (What the generator saw)"] = actor_context

        prompt["Implementation"] = wrap_code(code)
        prompt["Execution output"] = wrap_code(term_out, lang="")

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.model,
                temperature=self.temperature,
            ),
        )

        # Coerce metric to float or None.
        metric = response.get("metric")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            response["metric"] = None
        else:
            metric_f = float(metric)
            response["metric"] = metric_f if math.isfinite(metric_f) else None

        # Guardrail: treat exact 0/1 metrics as invalid placeholders.
        if response.get("metric") in (0.0, 1.0):
            response["metric"] = None
            response["is_bug"] = True

        return response

