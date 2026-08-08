"""configuration and setup utils"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, cast, List, Optional

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize
from .post_search import select_final_node, select_final_node_with_info

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("aide")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    # Optional OpenAI-compatible endpoint override (used by some run scripts).
    base_url: Optional[str] = None
    # Optional explicit key override (env var is typical).
    api_key: Optional[str] = None


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int

    # Debugging consultant / memory (optional; prompt-only effects in agent)
    use_bug_consultant: bool = True
    bug_context_mode: str = "consultant"  # "buggy_code" | "consultant" | "both"
    inject_shared_context: bool = True  # set False to ablate cross-branch memory injection
    bug_context_count: int = 3
    observation_window_size: int = 10
    # Prompt budget for retrieved/organized bug context (rare safety valve; may truncate injected memory only).
    advice_budget_chars: int = 20000

    # Memory safety valves (intentionally high defaults; should rarely trigger)
    max_bug_records: int = 500
    max_active_bugs: int = 200
    max_trials_per_bug: int = 20
    delete_pruned_bug_files: bool = False


@dataclass
class ProgressiveStageConfig:
    """Per-stage computational constraints (prompt-based; no code injection)."""

    data_fraction: float = 1.0
    disable_hparam_tuning: bool = False
    max_cv_folds: int = 5
    subsample_method: str = "stratified"  # "stratified" | "random" | "first_n"
    hparam_iter_limit: int | None = None


@dataclass
class ProgressiveConfig:
    """Progressive complexity schedule across the search budget."""

    enabled: bool = False
    exploration_end: float = 0.80
    refinement_end: float = 0.80

    exploration: ProgressiveStageConfig = field(
        default_factory=lambda: ProgressiveStageConfig(
            data_fraction=1.0,
            disable_hparam_tuning=True,
            max_cv_folds=3,
            subsample_method="stratified",
            hparam_iter_limit=None,
        )
    )
    refinement: ProgressiveStageConfig = field(
        default_factory=lambda: ProgressiveStageConfig(
            data_fraction=1.0,
            disable_hparam_tuning=True,
            max_cv_folds=5,
            subsample_method="stratified",
            hparam_iter_limit=None,
        )
    )
    validation: ProgressiveStageConfig = field(
        default_factory=lambda: ProgressiveStageConfig(
            data_fraction=1.0,
            disable_hparam_tuning=False,
            max_cv_folds=5,
            subsample_method="stratified",
            hparam_iter_limit=20,
        )
    )


@dataclass
class PlanConstraintsConfig:
    """Prompt-only constraints on the natural-language sketch length (no truncation/regeneration is performed)."""

    enabled: bool = True
    max_sentences: int | None = 5


@dataclass
class TimingConfig:
    """Lightweight timing/logging toggles (does not affect execution)."""

    enabled: bool = False
    track_cumulative_time: bool = True


@dataclass
class PostSearchConfig:
    """
    Configuration for selecting a final solution after search.

    This is separate from the search policy (which optimizes `node.metric`).
    """

    selection: str
    top_k: int

    # Robust selection parameters (used by some strategies)
    k_std: float
    z_threshold: float
    guard_std: float

    # Elite maximin parameters (takes max of all three)
    elite_top_k: int
    elite_ratio: float
    elite_k_std: float


@dataclass
class ExportConfig:
    """Controls additional artifacts written to the log dir (safe, no behavior change)."""

    save_solutions: bool
    save_submissions: bool
    save_metrics_table: bool
    save_final_selection: bool


@dataclass
class PerStepGradingConfig:
    """
    Configuration for per-step grading (generalization gap experiments).
    Grades all selection methods at each step using MLE-bench ground truth.
    """
    enabled: bool = False
    mlebench_data_dir: str = "mle-bench/data/competitions"
    methods: List[str] = field(default_factory=lambda: ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"])
    grade_every_n_steps: int = 1
    # Write `per_step_grading/grading_history.*` incrementally during the run.
    save_every_n_steps: int = 1


@dataclass
class AgentConfig:
    steps: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig

    search: SearchConfig


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool
    num_threads: int = 8


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    # If true, reuse the latest existing run directory matching `*-<exp_name>`
    # under `log_dir`/`workspace_dir` (instead of creating a new indexed run).
    resume: bool

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    progressive: ProgressiveConfig
    plan_constraints: PlanConstraintsConfig
    timing: TimingConfig
    post_search: PostSearchConfig
    export: ExportConfig
    per_step_grading: PerStepGradingConfig

    # Optional competition ID for per-step grading
    competition_id: Optional[str] = None


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _parse_indexed_run_dirname(name: str) -> tuple[int, str] | None:
    parts = name.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    return index, parts[1]


def _find_latest_run_dir(top_dir: Path, base_exp_name: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for p in top_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_indexed_run_dirname(p.name)
        if parsed is None:
            continue
        index, suffix = parsed
        if suffix != base_exp_name:
            continue
        if not (p / "journal.json").exists():
            continue
        candidates.append((index, p))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)

    # Resume mode: reuse latest existing run dir matching `*-<exp_name>`.
    if getattr(cfg, "resume", False):
        latest_log_dir = _find_latest_run_dir(top_log_dir, cfg.exp_name)
        if latest_log_dir is not None:
            cfg.exp_name = latest_log_dir.name
            cfg.log_dir = latest_log_dir.resolve()
            cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()
            cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
        else:
            ind = max(
                _get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir)
            )
            cfg.exp_name = f"{ind}-{cfg.exp_name}"
            cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
            cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()
    else:
        # New run: prefix with consecutive index.
        ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
        cfg.exp_name = f"{ind}-{cfg.exp_name}"
        cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
        cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    best_node = journal.get_best_node(only_good=False)
    with open(cfg.log_dir / "best_solution.py", "w") as f:
        f.write(best_node.code)

    _export_experiment_artifacts(cfg, journal)


def export_final_submissions(cfg: Config, journal, strategies: list[str] | None = None):
    """
    Export submission.csv files from different selection strategies.

    Args:
        cfg: Configuration object
        journal: Journal with all nodes
        strategies: List of strategy names to export. Defaults to ['raw', 'max_min', 'mean_minus_k_std', 'post_search'].
    """
    if strategies is None:
        strategies = ["raw", "max_min", "mean_minus_k_std", "post_search"]

    for strategy in strategies:
        if strategy == "raw":
            node = journal.get_best_node(only_good=False)
            out_name = "submission_raw.csv"
        elif strategy in {"max_min", "maximin", "maximin_no_filter"}:
            node = select_final_node(
                journal,
                selection="maximin_no_filter",
                top_k=cfg.post_search.top_k,
                k_std=cfg.post_search.k_std,
                z_threshold=cfg.post_search.z_threshold,
                guard_std=cfg.post_search.guard_std,
                elite_top_k=cfg.post_search.elite_top_k,
                elite_ratio=cfg.post_search.elite_ratio,
                elite_k_std=cfg.post_search.elite_k_std,
                only_good=False,
            )
            out_name = "submission_max_min.csv"
        elif strategy in {"mean_minus_k_std", "mean_k_std"}:
            node = select_final_node(
                journal,
                selection="mean_minus_k_std",
                top_k=cfg.post_search.top_k,
                k_std=cfg.post_search.k_std,
                z_threshold=cfg.post_search.z_threshold,
                guard_std=cfg.post_search.guard_std,
                elite_top_k=cfg.post_search.elite_top_k,
                elite_ratio=cfg.post_search.elite_ratio,
                elite_k_std=cfg.post_search.elite_k_std,
                only_good=False,
            )
            out_name = "submission_mean_minus_k_std.csv"
        elif strategy in {"post_search", "config"}:
            node = select_final_node(
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
            out_name = "submission_post_search.csv"
        else:
            node = select_final_node(
                journal,
                selection=strategy,
                top_k=cfg.post_search.top_k,
                k_std=cfg.post_search.k_std,
                z_threshold=cfg.post_search.z_threshold,
                guard_std=cfg.post_search.guard_std,
                elite_top_k=cfg.post_search.elite_top_k,
                elite_ratio=cfg.post_search.elite_ratio,
                elite_k_std=cfg.post_search.elite_k_std,
                only_good=False,
            )
            out_name = f"submission_{strategy}.csv"

        if node and node.submission_csv_path:
            src = Path(node.submission_csv_path)
            if src.exists():
                dst = cfg.log_dir / out_name
                dst.write_bytes(src.read_bytes())
                valid = node.valid_metric
                valid_str = f"{valid:.4f}" if isinstance(valid, (int, float)) else "N/A"
                print(f"✓ {dst.name} (step={node.step}, valid={valid_str})")
        else:
            print(f"✗ {strategy}: no submission found")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_experiment_artifacts(cfg: Config, journal) -> None:
    """
    Export standardized artifacts used by ICML-grade experiments.

    This function must not affect agent behavior; it only writes files under `cfg.log_dir`.
    """
    if not getattr(cfg, "export", None):
        return

    solutions_dir = cfg.log_dir / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts for the latest node only.
    # This avoids incorrectly attributing a later `submission.csv` to an earlier node that didn't produce one.
    last = journal.nodes[-1] if journal.nodes else None
    if last is not None and (cfg.export.save_solutions or cfg.export.save_submissions):
        if cfg.export.save_solutions:
            node_py = solutions_dir / f"node_{last.step}.py"
            if not node_py.exists():
                node_py.write_text(last.code)

        if cfg.export.save_submissions:
            src = cfg.workspace_dir / "working" / "submission.csv"
            if src.exists():
                dst = solutions_dir / f"submission_node_{last.step}.csv"
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
                try:
                    last.submission_csv_path = str(dst)  # type: ignore[attr-defined]
                    last.submission_csv_sha256 = _sha256_file(dst)  # type: ignore[attr-defined]
                except Exception:
                    # Node may not have these fields if a user runs an older journal schema.
                    pass

    # Write a metrics table for downstream post-search selection analysis.
    if cfg.export.save_metrics_table:
        rows: list[dict[str, Any]] = []
        for n in journal.nodes:
            cv_folds = getattr(n, "cv_folds", None)
            worst_fold = None
            if isinstance(cv_folds, list) and cv_folds:
                maximize = getattr(getattr(n, "metric", None), "maximize", True) is not False
                worst_fold = min(cv_folds) if maximize else max(cv_folds)

            rows.append(
                {
                    "step": n.step,
                    "stage": getattr(n, "stage_name", None),
                    "is_buggy": getattr(n, "is_buggy", None),
                    "metric": getattr(getattr(n, "metric", None), "value", None),
                    "maximize": getattr(getattr(n, "metric", None), "maximize", None),
                    "valid_metric": getattr(n, "valid_metric", None),
                    "cv_mean": getattr(n, "cv_mean", None),
                    "cv_std": getattr(n, "cv_std", None),
                    "cv_worst_fold": worst_fold,
                    "cv_folds": json.dumps(cv_folds) if isinstance(cv_folds, list) else None,
                    "submission_csv_path": getattr(n, "submission_csv_path", None),
                    "submission_csv_sha256": getattr(n, "submission_csv_sha256", None),
                }
            )

        metrics_csv = solutions_dir / "metrics.csv"
        metrics_jsonl = solutions_dir / "metrics.jsonl"
        with open(metrics_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)

    # Write selection report (updated every save_run; the final one is on disk at end-of-run).
    if cfg.export.save_final_selection:
        selected, info = select_final_node_with_info(
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
        best_valid = journal.get_best_node(only_good=False)
        report = {
            "best_valid": _selection_obj(best_valid),
            "post_search": _selection_obj(selected),
            "post_search_info": info,
            "population": _population_stats(journal),
        }
        (cfg.log_dir / "final_selection.json").write_text(json.dumps(report, indent=2))


def _selection_obj(node) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "step": getattr(node, "step", None),
        "valid_metric": getattr(node, "valid_metric", None),
        "cv_mean": getattr(node, "cv_mean", None),
        "cv_std": getattr(node, "cv_std", None),
        "cv_folds": getattr(node, "cv_folds", None),
        "submission_csv_path": getattr(node, "submission_csv_path", None),
        "submission_csv_sha256": getattr(node, "submission_csv_sha256", None),
    }


def _population_stats(journal) -> dict[str, Any]:
    cv_means = [getattr(n, "cv_mean", None) for n in journal.nodes]
    cv_means = [v for v in cv_means if isinstance(v, (int, float))]
    if not cv_means:
        return {"std_cv_mean_across_nodes": None, "mean_cv_mean_across_nodes": None}
    mean = sum(cv_means) / len(cv_means)
    var = sum((v - mean) ** 2 for v in cv_means) / len(cv_means)
    return {
        "mean_cv_mean_across_nodes": mean,
        "std_cv_mean_across_nodes": var**0.5,
        "n": len(cv_means),
    }
