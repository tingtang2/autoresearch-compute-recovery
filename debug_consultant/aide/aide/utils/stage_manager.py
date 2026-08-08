from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

StageType = Literal["exploration", "refinement", "validation"]


@dataclass(frozen=True)
class StageInfo:
    name: StageType
    step: int
    total_steps: int

    data_fraction: float
    disable_hparam_tuning: bool
    max_cv_folds: int
    hparam_iter_limit: Optional[int]
    subsample_method: str

    data_instruction: str
    hparam_instruction: str
    cv_folds_instruction: str


class StageManager:
    """
    Progressive complexity (prompt-based).

    The StageManager does NOT inject code or validate generated scripts.
    It only provides stage-specific instructions to add to prompts.
    """

    def __init__(self, progressive_cfg, total_steps: int):
        self.cfg = progressive_cfg
        self.total_steps = max(int(total_steps), 1)

        exploration_end = float(getattr(progressive_cfg, "exploration_end", 0.80))
        refinement_end = float(getattr(progressive_cfg, "refinement_end", 0.90))
        exploration_end = min(max(exploration_end, 0.0), 1.0)
        refinement_end = min(max(refinement_end, exploration_end), 1.0)

        self.exploration_threshold = int(self.total_steps * exploration_end)
        self.refinement_threshold = int(self.total_steps * refinement_end)

    def get_stage(self, step: int) -> StageInfo:
        step_i = max(int(step), 0)

        if step_i < self.exploration_threshold:
            stage_cfg = getattr(self.cfg, "exploration")
            name: StageType = "exploration"
        elif step_i < self.refinement_threshold:
            stage_cfg = getattr(self.cfg, "refinement")
            name = "refinement"
        else:
            stage_cfg = getattr(self.cfg, "validation")
            name = "validation"

        data_fraction = float(getattr(stage_cfg, "data_fraction", 1.0))
        data_fraction = min(max(data_fraction, 0.0), 1.0)
        subsample_method = str(getattr(stage_cfg, "subsample_method", "stratified"))

        disable_hparam_tuning = bool(getattr(stage_cfg, "disable_hparam_tuning", False))
        max_cv_folds = int(getattr(stage_cfg, "max_cv_folds", 5))
        max_cv_folds = max(max_cv_folds, 1)
        hparam_iter_limit = getattr(stage_cfg, "hparam_iter_limit", None)
        hparam_iter_limit = int(hparam_iter_limit) if hparam_iter_limit is not None else None

        pct = int(round(data_fraction * 100))
        if data_fraction >= 1.0:
            data_instruction = "Use FULL 100% of training data."
        else:
            if subsample_method == "first_n":
                how = "take the first N rows deterministically"
            elif subsample_method == "random":
                how = "randomly subsample rows"
            else:
                how = "use stratified sampling when possible (classification); otherwise random sampling"
            data_instruction = f"Use EXACTLY {pct}% of training data ({how})."

        if disable_hparam_tuning:
            hparam_instruction = (
                "Do NOT use GridSearchCV or RandomizedSearchCV; use simple default hyperparameters."
            )
        else:
            limit = hparam_iter_limit or 10
            hparam_instruction = (
                f"Hyperparameter tuning is allowed: use RandomizedSearchCV with n_iter ≤ {limit} "
                "and keep it lightweight (avoid huge search spaces)."
            )

        cv_folds_instruction = f"Use up to {max_cv_folds}-fold cross-validation (only if appropriate for the task)."

        return StageInfo(
            name=name,
            step=step_i,
            total_steps=self.total_steps,
            data_fraction=data_fraction,
            disable_hparam_tuning=disable_hparam_tuning,
            max_cv_folds=max_cv_folds,
            hparam_iter_limit=hparam_iter_limit,
            subsample_method=subsample_method,
            data_instruction=data_instruction,
            hparam_instruction=hparam_instruction,
            cv_folds_instruction=cv_folds_instruction,
        )
