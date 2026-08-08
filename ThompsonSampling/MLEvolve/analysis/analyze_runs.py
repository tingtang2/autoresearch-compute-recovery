#!/usr/bin/env python3
"""
Aggregate MLEvolve grading reports into per-method CSVs and a LaTeX comparison.

Each *run* (one seed) produces one ``*_grading_report.json`` file containing a
``competition_reports`` list. Given a folder of such reports for the **baseline**
(``use_thompson_sampling: false``) and another for **Thompson Sampling**
(``use_thompson_sampling: true``), this script:

  1. Writes one CSV per method in the MLEvolve report format:
        Competition Name, Run_01 ... Run_NN,
        Non-Zero Average, Zero Average,
        Gold Medal Rate, Silver Medal Rate, Bronze Medal Rate
  2. Writes a LaTeX table comparing TS vs. baseline (mean +/- SEM, winner per
     competition, ties bolded in both columns, dagger for lower-is-better).

Runs (Run_01 .. Run_NN) are ordered by grading-report filename (i.e. timestamp).

Usage:
    python analyze_runs.py \
        --baseline-dir /path/to/MLEvolve_Baselines \
        --ts-dir       /path/to/MLEvolve_TS \
        --outdir       ./analysis_out

Dependencies: standard library only.
"""

import csv
import json
import math
import argparse
from pathlib import Path
from typing import Optional


# ── Display config ────────────────────────────────────────────────────────────────
# Pretty names + fixed ordering for the LaTeX comparison table. Any competition not
# listed here is appended afterwards in alphabetical order using its raw id.
DISPLAY_NAME = {
    "class-prediction-of-cirrhosis-outcomes": "Cirrhosis",
    "gnss-classification":                    "GNSS",
    "spaceship-titanic":                      "Spaceship",
    "wine-quality-ordinal":                   "Wine",
    "playground-series-s5e3":                 "S5E3",
    "playground-series-s5e6":                 "S5E6",
    "playground-series-s5e7":                 "S5E7",
    "playground-series-s5e8":                 "S5E8",
    "playground-series-s5e12":                "S5E12",
}
TABLE_ORDER = [
    "class-prediction-of-cirrhosis-outcomes",
    "gnss-classification",
    "spaceship-titanic",
    "wine-quality-ordinal",
    "playground-series-s5e3",
    "playground-series-s5e6",
    "playground-series-s5e7",
    "playground-series-s5e8",
    "playground-series-s5e12",
]


# ── Loading ─────────────────────────────────────────────────────────────────────
def load_reports(folder: Path) -> list[dict]:
    """Load grading reports, sorted by filename (== run order Run_01, Run_02, ...)."""
    paths = sorted(folder.glob("*_grading_report.json"))
    if not paths:
        # Fall back to any *.json that looks like a grading report.
        paths = sorted(
            p for p in folder.glob("*.json")
            if "competition_reports" in p.read_text(errors="replace")
        )
    if not paths:
        raise SystemExit(f"No grading reports found in {folder}")
    return [json.loads(p.read_text()) for p in paths]


def build_table(reports: list[dict]) -> tuple[dict, list[str]]:
    """Return per-competition aggregated data plus the sorted competition id list.

    data[comp_id] = {
        "is_lower_better": bool,
        "scores": [score_or_None per run],
        "gold":   [bool per run],
        "silver": [bool per run],
        "bronze": [bool per run],
    }
    """
    n_runs = len(reports)
    data: dict[str, dict] = {}

    for run_idx, report in enumerate(reports):
        for cr in report.get("competition_reports", []):
            comp = cr["competition_id"]
            entry = data.setdefault(comp, {
                "is_lower_better": bool(cr.get("is_lower_better", False)),
                "scores": [None] * n_runs,
                "gold":   [False] * n_runs,
                "silver": [False] * n_runs,
                "bronze": [False] * n_runs,
            })
            entry["is_lower_better"] = bool(cr.get("is_lower_better", entry["is_lower_better"]))
            entry["scores"][run_idx] = cr.get("score")
            entry["gold"][run_idx]   = bool(cr.get("gold_medal", False))
            entry["silver"][run_idx] = bool(cr.get("silver_medal", False))
            entry["bronze"][run_idx] = bool(cr.get("bronze_medal", False))

    return data, sorted(data.keys())


# ── Formatting helpers ────────────────────────────────────────────────────────────
def _fmt_score(value) -> str:
    """Raw per-run score cell: 'null' or the round-tripped number."""
    if value is None:
        return "null"
    return str(value)


def _fmt_avg(value: Optional[float]) -> str:
    if value is None:
        return "null"
    return str(round(value, 5))


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _sem(values: list[float]) -> float:
    """Standard error of the mean (sample std, ddof=1). 0 for < 2 samples."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


# ── CSV writing ───────────────────────────────────────────────────────────────────
def write_csv(data: dict, comps: list[str], n_runs: int, csv_path: Path) -> None:
    run_cols = [f"Run_{i + 1:02d}" for i in range(n_runs)]
    fieldnames = (
        ["Competition Name"] + run_cols
        + ["Non-Zero Average", "Zero Average",
           "Gold Medal Rate", "Silver Medal Rate", "Bronze Medal Rate"]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for comp in comps:
            entry = data[comp]
            scores = entry["scores"]
            valid = [s for s in scores if s is not None]

            non_zero_avg = _mean(valid)                       # ignore missing runs
            zero_avg = (sum(valid) / n_runs) if n_runs else None  # missing = 0

            row = [comp] + [_fmt_score(s) for s in scores] + [
                _fmt_avg(non_zero_avg),
                _fmt_avg(zero_avg),
                f"{sum(entry['gold'])}/{n_runs}",
                f"{sum(entry['silver'])}/{n_runs}",
                f"{sum(entry['bronze'])}/{n_runs}",
            ]
            writer.writerow(row)

    print(f"CSV written → {csv_path}")


# ── LaTeX comparison ──────────────────────────────────────────────────────────────
def _ordered_comps(all_comps: set[str]) -> list[str]:
    ordered = [c for c in TABLE_ORDER if c in all_comps]
    extras = sorted(c for c in all_comps if c not in TABLE_ORDER)
    return ordered + extras


def _cell(mean: Optional[float], sem: float, bold: bool) -> str:
    if mean is None:
        return "--"
    body = f"{mean:.3f} $\\pm$ {sem:.3f}"
    return f"\\textbf{{{body}}}" if bold else body


def write_latex(ts_data: dict, base_data: dict, tex_path: Path) -> None:
    all_comps = set(ts_data) | set(base_data)
    comps = _ordered_comps(all_comps)

    rows: list[str] = []
    ts_wins = base_wins = ties = 0

    for comp in comps:
        ts_entry = ts_data.get(comp)
        base_entry = base_data.get(comp)
        lower_better = (
            (ts_entry or base_entry)["is_lower_better"]
        )

        ts_valid = [s for s in ts_entry["scores"] if s is not None] if ts_entry else []
        base_valid = [s for s in base_entry["scores"] if s is not None] if base_entry else []
        ts_mean, base_mean = _mean(ts_valid), _mean(base_valid)
        ts_sem, base_sem = _sem(ts_valid), _sem(base_valid)

        # Winner: compare true means; tie if equal at displayed (3dp) precision.
        ts_bold = base_bold = False
        if ts_mean is None or base_mean is None:
            winner = "TS" if base_mean is None else "Baseline"
            ts_bold = ts_mean is not None
            base_bold = base_mean is not None
        elif round(ts_mean, 3) == round(base_mean, 3):
            winner, ts_bold, base_bold, ties = "Tie", True, True, ties + 1
        else:
            ts_better = (ts_mean < base_mean) if lower_better else (ts_mean > base_mean)
            if ts_better:
                winner, ts_bold, ts_wins = "TS", True, ts_wins + 1
            else:
                winner, base_bold, base_wins = "Baseline", True, base_wins + 1

        name = DISPLAY_NAME.get(comp, comp)
        if lower_better:
            name += "$^{\\dagger}$"

        rows.append(
            f"    {name} & {_cell(ts_mean, ts_sem, ts_bold)} "
            f"& {_cell(base_mean, base_sem, base_bold)} & {winner} \\\\"
        )

    n = len(comps)
    caption = (
        "\\textbf{Thompson Sampling vs.\\ the MLEvolve baseline.} "
        f"With all other settings held fixed, TS wins {ts_wins} of {n} competitions"
        + (f", ties {ties}," if ties else "")
        + f" and the baseline wins {base_wins}. "
        "Values are mean $\\pm$ SEM over the valid runs per competition. "
        "Bold indicates the winning method per competition; values tied at the "
        "displayed precision are bolded in both columns. $^{\\dagger}$Lower is better."
    )

    table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:mlevolve_ts}\n"
        "\\small\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "\\textbf{Competition} & \\textbf{MLEvolve + TS} & "
        "\\textbf{MLEvolve Baseline} & \\textbf{Winner} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    tex_path.write_text(table)
    print(f"LaTeX written → {tex_path}")
    print(f"  TS wins: {ts_wins}  Baseline wins: {base_wins}  Ties: {ties}")


# ── Entry point ───────────────────────────────────────────────────────────────────
def process(folder: Path, csv_path: Path) -> dict:
    reports = load_reports(folder)
    data, comps = build_table(reports)
    write_csv(data, comps, len(reports), csv_path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MLEvolve grading reports into CSVs + a LaTeX TS/baseline comparison."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Folder of baseline (UCT) grading reports.")
    parser.add_argument("--ts-dir", type=Path, required=True,
                        help="Folder of Thompson Sampling grading reports.")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory (default: current dir).")
    parser.add_argument("--baseline-csv-name", default="MLEvolve_Baseline_Report.csv")
    parser.add_argument("--ts-csv-name", default="MLEvolve_TS_Report.csv")
    parser.add_argument("--tex-name", default="mlevolve_ts_comparison.tex")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    base_data = process(args.baseline_dir, args.outdir / args.baseline_csv_name)
    ts_data = process(args.ts_dir, args.outdir / args.ts_csv_name)

    write_latex(ts_data, base_data, args.outdir / args.tex_name)


if __name__ == "__main__":
    main()
