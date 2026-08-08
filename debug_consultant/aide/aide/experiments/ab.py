from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path

from .core import run_once


def _parse_seeds(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="aide-ab",
        description="Run a standardized A/B comparison where the control is always AIDE with 50 steps.",
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--goal", default=None)
    p.add_argument("--eval", default=None)
    p.add_argument("--desc-file", default=None)
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds.")
    p.add_argument(
        "--treatment-overrides",
        action="append",
        default=[],
        help='OmegaConf dotlist override, e.g. "agent.search.debug_prob=0.0". Repeatable.',
    )
    p.add_argument(
        "--control-overrides",
        action="append",
        default=[],
        help="Optional overrides applied to both arms (rare). Repeatable.",
    )
    p.add_argument(
        "--treatment-steps",
        type=int,
        default=50,
        help="Treatment steps (control is fixed at 50).",
    )
    p.add_argument(
        "--out",
        default="logs/ab_summary.csv",
        help="Write a CSV summary to this path.",
    )
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds)

    rows: list[dict] = []
    for seed in seeds:
        ctrl = run_once(
            data_dir=args.data_dir,
            goal=args.goal,
            eval=args.eval,
            desc_file=args.desc_file,
            steps=50,
            seed=seed,
            exp_name=f"ab-control-seed{seed}",
            overrides=args.control_overrides,
        )
        rows.append({"arm": "control", **ctrl.__dict__})

        trt = run_once(
            data_dir=args.data_dir,
            goal=args.goal,
            eval=args.eval,
            desc_file=args.desc_file,
            steps=args.treatment_steps,
            seed=seed,
            exp_name=f"ab-treatment-seed{seed}",
            overrides=args.control_overrides + args.treatment_overrides,
        )
        rows.append({"arm": "treatment", **trt.__dict__})

        _write_submission_diff(
            dataset_name=Path(args.data_dir).name,
            seed=seed,
            out_csv_path=Path(args.out),
            control_log_dir=Path(ctrl.log_dir),
            treatment_log_dir=Path(trt.log_dir),
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_path}")
    return 0


def _write_submission_diff(
    *,
    dataset_name: str,
    seed: int,
    out_csv_path: Path,
    control_log_dir: Path,
    treatment_log_dir: Path,
) -> None:
    """
    If both arms produced a selected `submission.csv`, write a unified diff for inspection.
    """
    # Place diffs next to the summary CSV, but outside AIDE logs.
    diffs_dir = out_csv_path.parent / "submission_diffs" / dataset_name
    diffs_dir.mkdir(parents=True, exist_ok=True)

    def load_selection(log_dir: Path) -> dict | None:
        p = log_dir / "final_selection.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    ctrl = load_selection(control_log_dir) or {}
    trt = load_selection(treatment_log_dir) or {}

    ctrl_step = (ctrl.get("best_valid") or {}).get("step")
    trt_step = (trt.get("post_search") or {}).get("step")

    if ctrl_step is None or trt_step is None:
        (diffs_dir / f"seed{seed}.txt").write_text(
            "Missing selection steps; cannot diff submissions.\n"
        )
        return

    ctrl_sub = control_log_dir / "solutions" / f"submission_node_{ctrl_step}.csv"
    trt_sub = treatment_log_dir / "solutions" / f"submission_node_{trt_step}.csv"

    if not (ctrl_sub.exists() and trt_sub.exists()):
        (diffs_dir / f"seed{seed}.txt").write_text(
            f"Missing submissions:\n- control: {ctrl_sub} exists={ctrl_sub.exists()}\n- treatment: {trt_sub} exists={trt_sub.exists()}\n"
        )
        return

    # Avoid writing gigantic diffs for large submissions.
    max_bytes = 5 * 1024 * 1024
    if ctrl_sub.stat().st_size > max_bytes or trt_sub.stat().st_size > max_bytes:
        (diffs_dir / f"seed{seed}.txt").write_text(
            "Submission diff skipped due to file size.\n"
            f"- control: {ctrl_sub} ({ctrl_sub.stat().st_size} bytes)\n"
            f"- treatment: {trt_sub} ({trt_sub.stat().st_size} bytes)\n"
        )
        return

    a = ctrl_sub.read_text().splitlines(keepends=True)
    b = trt_sub.read_text().splitlines(keepends=True)
    diff = difflib.unified_diff(
        a,
        b,
        fromfile=f"control(seed={seed}, step={ctrl_step})",
        tofile=f"treatment(seed={seed}, step={trt_step})",
    )
    (diffs_dir / f"seed{seed}.diff").write_text("".join(diff))


if __name__ == "__main__":
    raise SystemExit(main())
