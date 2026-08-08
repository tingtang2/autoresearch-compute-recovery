from __future__ import annotations

import logging
import os
import time
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from ..agent import Agent
from ..interpreter import ExecutionResult, Interpreter
from ..journal import Journal
from ..utils.config import _load_cfg, load_task_desc, prep_agent_workspace, prep_cfg, save_run
from ..utils.post_search import select_final_node

logger = logging.getLogger("aide")


@dataclass
class RunSummary:
    exp_name: str
    log_dir: str
    workspace_dir: str
    seed: int
    steps: int
    buggy_nodes: int
    good_nodes: int
    selected_valid: float | None
    selected_submission_sha256: str | None


def _apply_overrides(cfg: Any, overrides: list[str]) -> Any:
    if not overrides:
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def run_once(
    *,
    data_dir: str,
    goal: str | None,
    eval: str | None,
    desc_file: str | None,
    steps: int,
    seed: int,
    exp_name: str,
    overrides: list[str] | None = None,
) -> RunSummary:
    overrides = overrides or []

    base = _load_cfg(use_cli_args=False)
    base.data_dir = data_dir
    base.goal = goal
    base.eval = eval
    base.desc_file = desc_file
    base.exp_name = exp_name
    base.agent.steps = steps

    cfg = prep_cfg(_apply_overrides(base, overrides))

    random.seed(seed)
    np.random.seed(seed)

    task_desc = load_task_desc(cfg)
    prep_agent_workspace(cfg)

    journal = Journal()
    agent = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
    interpreter = Interpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec))  # type: ignore[arg-type]

    def exec_callback(*args, **kwargs):
        max_retries = int(os.environ.get("AIDE_REPL_RETRIES", "2") or "2")
        for attempt in range(max_retries + 1):
            try:
                return interpreter.run(*args, **kwargs)
            except RuntimeError as e:
                if "REPL child process" not in str(e):
                    raise
                logger.error(
                    "REPL crash at step %s (attempt %s/%s): %s",
                    len(journal),
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                try:
                    interpreter.cleanup_session()
                except Exception as cleanup_err:
                    logger.error(f"Cleanup after REPL crash failed: {cleanup_err}")
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return ExecutionResult(
                    term_out=[f"RuntimeError: {e}\n"],
                    exec_time=0.0,
                    exc_type="RuntimeError",
                    exc_info={"args": [str(e)]},
                    exc_stack=None,
                )

    for _ in range(steps):
        agent.step(exec_callback=exec_callback)
        save_run(cfg, journal)

    interpreter.cleanup_session()

    selected = select_final_node(
        journal,
        selection=cfg.post_search.selection,
        top_k=cfg.post_search.top_k,
        k_std=cfg.post_search.k_std,
        z_threshold=cfg.post_search.z_threshold,
        guard_std=cfg.post_search.guard_std,
        elite_top_k=cfg.post_search.elite_top_k,
        elite_ratio=cfg.post_search.elite_ratio,
        elite_k_std=cfg.post_search.elite_k_std,
        only_good=False,
    )
    # No fallback - if selection method fails, selected stays None

    return RunSummary(
        exp_name=cfg.exp_name,
        log_dir=str(cfg.log_dir),
        workspace_dir=str(cfg.workspace_dir),
        seed=seed,
        steps=steps,
        buggy_nodes=len(journal.buggy_nodes),
        good_nodes=len(journal.good_nodes),
        selected_valid=getattr(selected, "valid_metric", None)
        if selected is not None
        else None,
        selected_submission_sha256=getattr(selected, "submission_csv_sha256", None)
        if selected is not None
        else None,
    )
