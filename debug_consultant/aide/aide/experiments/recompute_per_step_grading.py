#!/usr/bin/env python3
"""Recompute per-step grading artifacts for an existing AIDE run directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from aide.journal import Journal
from aide.utils import serialize
from aide.utils.mlebench_grading import setup_per_step_grading


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the run directory containing journal.json and config yaml",
    )
    parser.add_argument(
        "--competition-id",
        default=None,
        help="Override competition id (default: config `competition_id` or env `COMPETITION_ID`).",
    )
    parser.add_argument(
        "--mlebench-data-dir",
        default=None,
        help="Override the MLE-bench data dir (default: config `per_step_grading.mlebench_data_dir`).",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only recompute up to this step (default: max step in journal).",
    )
    return parser.parse_args()


def _locate_config(run_dir: Path) -> Path:
    for name in ("config_mcts.yaml", "config.yaml"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to find config_mcts.yaml or config.yaml under {run_dir}")


def _ensure_path(value, default: Path) -> Path:
    if value is None:
        return default
    try:
        return Path(value)
    except TypeError:
        return Path(str(value))


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()

    cfg_path = _locate_config(run_dir)
    journal_path = run_dir / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing journal: {journal_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.log_dir = str(run_dir)
    if not getattr(cfg, "exp_name", None):
        cfg.exp_name = run_dir.name

    per_step_cfg = getattr(cfg, "per_step_grading", None)
    if per_step_cfg is None:
        raise ValueError("Config is missing `per_step_grading` section")
    per_step_cfg.enabled = True
    if args.mlebench_data_dir is not None:
        per_step_cfg.mlebench_data_dir = args.mlebench_data_dir

    competition_id = (
        args.competition_id
        or os.environ.get("COMPETITION_ID")
        or getattr(cfg, "competition_id", None)
    )
    if not competition_id:
        raise ValueError(
            "No competition id provided (set --competition-id, COMPETITION_ID, or cfg.competition_id)"
        )

    journal = serialize.load_json(journal_path, Journal)
    steps = [getattr(node, "step", None) for node in getattr(journal, "nodes", [])]
    steps = [int(s) for s in steps if isinstance(s, int)]
    max_step = args.max_step if args.max_step is not None else (max(steps) if steps else 0)

    workspace_dir = _ensure_path(getattr(cfg, "workspace_dir", None), run_dir / "workspace")
    workspace_dir = workspace_dir.resolve()

    callback = setup_per_step_grading(cfg, competition_id)
    if callback is None:
        raise RuntimeError(
            "setup_per_step_grading returned None (check cfg.per_step_grading.* and competition id)"
        )

    for step in range(1, max_step + 1):
        callback.on_step_complete(journal, step, workspace_dir, cfg)

    try:
        callback.save_results()
    except Exception:
        # Intermediate CSV/JSON data may already exist; do not fail recomputation on final save.
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
