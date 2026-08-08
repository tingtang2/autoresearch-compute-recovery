"""
Integration with MLE-bench for per-step grading.
Allows grading submissions during search to track generalization gap.
"""

import logging
from pathlib import Path
from typing import Optional
import json
import traceback
import csv
import os
import tempfile
import math

try:  # Optional dependency; present in all runner envs but guard just in case.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - fallback if numpy missing
    _np = None

logger = logging.getLogger("aide")


def _json_default(obj):
    """Convert numpy scalars (e.g., np.bool_) to native Python types for JSON."""
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _sanitize_for_csv_cell(value: str | None) -> str | None:
    """
    Ensure a CSV cell value is single-line.

    Embedded newlines inside quoted CSV fields are valid CSV, but they break common
    line-oriented workflows (e.g., `tail -f`, `awk`, simple greps) used to monitor
    per-step grading outputs in real time.
    """
    if value is None:
        return None
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value.replace("\n", "\\n")

def _read_csv_header(path: Path) -> list[str]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _maybe_coerce_submission_header(
    submission_path: Path, expected_columns: list[str]
) -> tuple[Path | None, str | None]:
    """
    Best-effort: coerce submission header *and column order* to match MLE-bench sample submission.

    Common issues:
    - `id` column not first (e.g. `C,CL,D,id`)
    - class columns missing the label prefix (e.g. `C,CL,D` vs `Status_C,Status_CL,Status_D`)

    If a unique mapping exists from `expected_columns` to the provided columns, this rewrites
    the CSV to the expected schema (header + per-row column reordering) into a sidecar file.
    """
    try:
        got = _read_csv_header(submission_path)
    except Exception as e:
        return None, f"Failed to read submission header: {e}"

    if got == expected_columns:
        return None, None

    norm = lambda s: s.strip().lower()
    got_norm = [norm(c) for c in got]
    expected_norm = [norm(c) for c in expected_columns]

    if not got_norm or not expected_norm:
        return None, None

    # Build a unique mapping from expected column -> actual column.
    # Prefer exact match; otherwise, allow suffix match for columns like "Status_C" -> "C".
    mapping: dict[str, str] = {}
    used_actual_norm: set[str] = set()

    got_norm_to_actual: dict[str, list[str]] = {}
    for actual, actual_norm in zip(got, got_norm):
        got_norm_to_actual.setdefault(actual_norm, []).append(actual)

    for exp_col, exp_col_norm in zip(expected_columns, expected_norm):
        candidates: list[str] = []

        exact = got_norm_to_actual.get(exp_col_norm, [])
        candidates.extend(exact)

        if not candidates and "_" in exp_col_norm:
            suffix = exp_col_norm.split("_")[-1]
            candidates.extend(got_norm_to_actual.get(suffix, []))

        # Unique match required.
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            return None, None

        actual = candidates[0]
        actual_norm = norm(actual)
        if actual_norm in used_actual_norm:
            return None, None
        used_actual_norm.add(actual_norm)
        mapping[exp_col] = actual

    # Must have an id column, and expected must want one.
    if "id" not in got_norm or "id" not in expected_norm:
        return None, None
    if mapping.get(expected_columns[expected_norm.index("id")]) is None:
        return None, None

    last_err: Exception | None = None
    # Prefer system temp dir first: some filesystems (e.g., NFS) can behave poorly
    # with NamedTemporaryFile + atomic rename semantics.
    for tmp_dir in (None, submission_path.parent):
        tmp_path: Path | None = None
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                delete=False,
                suffix=".csv",
                dir=str(tmp_dir) if tmp_dir is not None else None,
            )
            tmp_path = Path(tmp.name)
            with open(submission_path, "r", newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                writer = csv.DictWriter(tmp, fieldnames=expected_columns, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    out_row = {exp: row.get(mapping[exp], "") for exp in expected_columns}
                    writer.writerow(out_row)
            return tmp_path, None
        except Exception as e:
            last_err = e
            try:
                if tmp is not None:
                    tmp.close()
            except Exception:
                pass
            try:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        finally:
            try:
                if tmp is not None:
                    tmp.close()
            except Exception:
                pass

    return None, f"Failed to rewrite coerced submission: {last_err}"

def _ensure_mlebench_importable(data_dir: Path) -> None:
    """
    Ensure the `mlebench` package is importable.

    In this setup the `mle-bench` repo often lives outside the AIDE tree, and we
    may only have a data directory like `.../mle-bench/data/competitions`.
    """
    try:
        import mlebench  # noqa: F401
        return
    except Exception:
        pass

    import sys

    candidates: list[Path] = []
    env_root = os.environ.get("MLEBENCH_REPO_DIR")
    if env_root:
        candidates.append(Path(env_root))

    # If data_dir is `.../mle-bench/data/competitions`, repo root is `.../mle-bench`.
    candidates.append(data_dir.parent.parent)
    candidates.append(Path.home() / "mle-bench")

    # In mlebench containers, mlebench is installed as editable at /mlebench
    candidates.append(Path("/mlebench"))

    for root in candidates:
        # Check for site-packages path (conda env) or repo root
        if (root / "mlebench" / "__init__.py").exists() or (root.name == "site-packages" and (root / "mlebench").exists()):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                import mlebench  # noqa: F401
                return
            except Exception:
                pass


def _filter_submission_to_private_answers(
    submission_path: Path, answers_path: Path, *, id_key_hint: str | None = None
) -> tuple[Path | None, str | None]:
    """
    MLE-bench grades against `prepared/private/answers.csv`, which is often a subset
    of Kaggle's public test rows. Convert a full submission into the private subset
    (ordered to match `answers.csv`) before grading.
    """
    try:
        def _norm_key(name: str) -> str:
            # Trim whitespace and strip a UTF-8 BOM if present.
            return name.strip().lstrip("\ufeff").lower()

        def _pick_join_key(fieldnames: list[str] | None, *, hints: list[str]) -> str | None:
            if not fieldnames:
                return None
            # Prefer explicit hints (e.g., sample submission id column).
            for hint in hints:
                if not hint:
                    continue
                hint_norm = _norm_key(hint)
                for name in fieldnames:
                    if _norm_key(name) == hint_norm:
                        return name

            # Common case: explicit `id` column.
            for name in fieldnames:
                if _norm_key(name) == "id":
                    return name

            # Otherwise, accept a unique "*id" column (e.g., PassengerId, image_id).
            id_like = [name for name in fieldnames if _norm_key(name).endswith("id")]
            if len(id_like) == 1:
                return id_like[0]

            # Last resort: use the first column as the join key (common for Kaggle submissions/answers).
            return fieldnames[0]

        with open(answers_path, newline="", encoding="utf-8") as f:
            answers_reader = csv.DictReader(f)
            answers_id_key = _pick_join_key(
                answers_reader.fieldnames,
                hints=[id_key_hint] if id_key_hint else [],
            )
            if not answers_id_key:
                return None, f"Invalid answers file (missing join key): {answers_path}"
            answer_ids = [row[answers_id_key] for row in answers_reader]

        with open(submission_path, newline="", encoding="utf-8") as f:
            sub_reader = csv.DictReader(f)
            hints: list[str] = [answers_id_key]
            if id_key_hint:
                hints.append(id_key_hint)
            sub_id_key = _pick_join_key(sub_reader.fieldnames, hints=hints)
            if not sub_id_key:
                return None, f"Invalid submission file (missing join key): {submission_path}"
            fieldnames = list(sub_reader.fieldnames)
            sub_by_id = {row[sub_id_key]: row for row in sub_reader}

        missing = [i for i in answer_ids if i not in sub_by_id]
        if missing:
            return None, f"Submission missing {len(missing)} private ids (e.g. {missing[0]})"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False, suffix=".csv"
        )
        tmp_path = Path(tmp.name)
        try:
            w = csv.DictWriter(tmp, fieldnames=fieldnames)
            w.writeheader()
            for i in answer_ids:
                w.writerow(sub_by_id[i])
        finally:
            tmp.close()

        return tmp_path, None
    except Exception as e:
        return None, str(e)


def _compute_cv_stats(node) -> tuple[float | None, float | None, list[float] | None]:
    """
    Source of truth is `node.cv_folds` when present; compute mean/std from it.
    Falls back to `cv_mean`/`cv_std`/`valid_metric`/`metric.value`.
    """
    folds = getattr(node, "cv_folds", None)
    if isinstance(folds, list) and folds:
        vals: list[float] = []
        for v in folds:
            try:
                f = float(v)
            except Exception:
                continue
            if math.isfinite(f):
                vals.append(f)
        if vals:
            m = sum(vals) / len(vals)
            if len(vals) == 1:
                return m, 0.0, vals
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            return m, var**0.5, vals

    mean_v = getattr(node, "cv_mean", None)
    std_v = getattr(node, "cv_std", None)
    if mean_v is None:
        mean_v = getattr(node, "valid_metric", None)
    if mean_v is None:
        metric = getattr(node, "metric", None)
        mean_v = getattr(metric, "value", None) if metric is not None else None
    return mean_v, std_v, None


def _looks_like_continuous_classification_error(message: str | None) -> bool:
    """
    Heuristic: sklearn raises errors like:
      "Classification metrics can't handle a mix of multiclass and continuous targets"

    This commonly happens when the submission contains continuous floats for a discrete-label
    competition graded with accuracy.
    """
    if not message:
        return False
    m = message.lower()
    return ("can't handle a mix of" in m) and ("continuous" in m) and ("targets" in m)


_DISCRETE_LABEL_CACHE: dict[tuple[str, str], list[float]] = {}


def _maybe_coerce_continuous_preds_to_discrete_labels(
    submission_path: Path,
    competition,
    expected_cols: list[str] | None,
) -> tuple[Path | None, str | None]:
    """
    Best-effort: if a submission predicts continuous floats for a discrete-label target,
    coerce predictions onto the set of labels observed in the private answers.

    This is especially useful for ordinal targets where a regressor is used and then
    predictions should be rounded/clipped for classification-style grading.
    """
    if not expected_cols or len(expected_cols) < 2:
        return None, "Cannot infer id/target columns from sample submission"

    id_col = expected_cols[0]
    target_col = expected_cols[1]

    try:
        import numpy as np
        import pandas as pd
        from mlebench.utils import load_answers

        answers_path = str(getattr(competition, "answers", ""))
        cache_key = (answers_path, target_col)
        allowed = _DISCRETE_LABEL_CACHE.get(cache_key)
        if allowed is None:
            answers_df = load_answers(competition.answers)
            if target_col not in answers_df.columns:
                return None, f"Target column `{target_col}` not found in answers"
            labels = answers_df[target_col].dropna()
            if labels.empty:
                return None, "No labels found in answers"
            # Only attempt for small discrete label sets.
            uniq = pd.to_numeric(labels, errors="coerce").dropna().unique()
            if len(uniq) == 0:
                return None, "Non-numeric labels; cannot coerce"
            if len(uniq) > 200:
                return None, f"Too many unique labels ({len(uniq)}); refusing to coerce"
            allowed = sorted(float(v) for v in uniq)
            _DISCRETE_LABEL_CACHE[cache_key] = allowed

        sub = pd.read_csv(submission_path)
        if id_col not in sub.columns or target_col not in sub.columns:
            return None, f"Submission missing `{id_col}` or `{target_col}` columns"

        preds = pd.to_numeric(sub[target_col], errors="coerce")
        if preds.isna().any():
            return None, "Predictions are not numeric; cannot coerce"

        allowed_arr = np.asarray(allowed, dtype=float)
        pred_arr = preds.to_numpy(dtype=float)
        # Map each prediction to the nearest allowed label.
        nearest = allowed_arr[np.abs(pred_arr[:, None] - allowed_arr[None, :]).argmin(axis=1)]
        sub[target_col] = nearest

        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", prefix="mlebench_discrete_", mode="w", newline="", encoding="utf-8"
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        sub.to_csv(tmp_path, index=False)
        return tmp_path, None
    except Exception as e:
        return None, str(e)


def grade_submission_with_mlebench(
    submission_path: Path,
    competition_id: str,
    data_dir: Path,
) -> dict:
    """
    Grade a submission using MLE-bench.

    Args:
        submission_path: Path to submission CSV
        competition_id: MLE-bench competition ID
        data_dir: MLE-bench data directory

    Returns:
        dict with keys: score, percentile, error (if failed)
    """
    tmp_filtered: Path | None = None
    tmp_coerced: Path | None = None
    tmp_discrete: Path | None = None
    try:
        _ensure_mlebench_importable(data_dir)

        from mlebench.grade import grade_csv, validate_submission
        from mlebench.registry import Registry

        registry = Registry().set_data_dir(data_dir)
        competition = registry.get_competition(competition_id)

        expected_cols = _read_csv_header(competition.sample_submission)
        id_key_hint = expected_cols[0] if expected_cols else None

        answers_path = data_dir / competition_id / "prepared" / "private" / "answers.csv"
        if answers_path.exists():
            tmp_filtered, err = _filter_submission_to_private_answers(
                submission_path, answers_path, id_key_hint=id_key_hint
            )
            if err is not None:
                return {"score": None, "percentile": None, "error": _sanitize_for_csv_cell(err)}
            submission_for_grading = tmp_filtered
        else:
            submission_for_grading = submission_path

        tmp_coerced, coerce_err = _maybe_coerce_submission_header(submission_for_grading, expected_cols)
        if coerce_err is not None:
            logger.warning("Submission schema auto-coercion failed (%s): %s", competition_id, coerce_err)
        submission_for_grading = tmp_coerced or submission_for_grading

        is_valid, message = validate_submission(submission_for_grading, competition)
        if (not is_valid) and _looks_like_continuous_classification_error(message):
            tmp_discrete, _fix_err = _maybe_coerce_continuous_preds_to_discrete_labels(
                submission_for_grading, competition, expected_cols
            )
            if tmp_discrete is not None:
                is_valid2, message2 = validate_submission(tmp_discrete, competition)
                if is_valid2:
                    submission_for_grading = tmp_discrete
                    is_valid, message = is_valid2, message2
        if not is_valid:
            if tmp_coerced is None and coerce_err is not None:
                message = f"{message} (auto-coercion failed: {coerce_err})"
            message = _sanitize_for_csv_cell(message)
            return {
                "score": None,
                "percentile": None,
                "error": message,
                "gold_threshold": None,
                "silver_threshold": None,
                "bronze_threshold": None,
                "median_threshold": None,
                "gold_medal": None,
                "silver_medal": None,
                "bronze_medal": None,
                "above_median": None,
                "is_lower_better": None,
            }

        report = grade_csv(submission_for_grading, competition)

        percentile = None
        try:
            leaderboard_path = Path(str(getattr(competition, "leaderboard", "")))
            if leaderboard_path.exists():
                # Skip git-lfs pointer files.
                try:
                    with open(leaderboard_path, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    if not first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                        scores: list[float] = []
                        with open(leaderboard_path, "r", newline="", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            if reader.fieldnames:
                                score_col = _pick_leaderboard_score_column(reader.fieldnames)
                                if score_col is not None:
                                    for row in reader:
                                        v = row.get(score_col)
                                        if v is None or v == "":
                                            continue
                                        try:
                                            scores.append(float(v))
                                        except Exception:
                                            continue
                        if scores and report.score is not None:
                            n = len(scores)
                            is_lower_better = None
                            if len(scores) >= 2:
                                if scores[0] < scores[-1]:
                                    is_lower_better = True
                                elif scores[0] > scores[-1]:
                                    is_lower_better = False
                            if is_lower_better is None:
                                is_lower_better = bool(getattr(report, "is_lower_better", False))
                            if is_lower_better:
                                worse_or_equal = sum(1 for s in scores if s >= report.score)
                            else:
                                worse_or_equal = sum(1 for s in scores if s <= report.score)
                            percentile = max(0.0, min(100.0, 100.0 * worse_or_equal / n))
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "score": report.score,
            "percentile": percentile,
            "error": None if report.score is not None else _sanitize_for_csv_cell("Invalid submission for grading"),
            "gold_threshold": getattr(report, "gold_threshold", None),
            "silver_threshold": getattr(report, "silver_threshold", None),
            "bronze_threshold": getattr(report, "bronze_threshold", None),
            "median_threshold": getattr(report, "median_threshold", None),
            "gold_medal": getattr(report, "gold_medal", None),
            "silver_medal": getattr(report, "silver_medal", None),
            "bronze_medal": getattr(report, "bronze_medal", None),
            "above_median": getattr(report, "above_median", None),
            "is_lower_better": getattr(report, "is_lower_better", None),
        }
    except Exception as e:
        logger.debug(f"MLE-bench grading failed: {e}\n{traceback.format_exc()}")
        return {
            "score": None,
            "percentile": None,
            "error": _sanitize_for_csv_cell(str(e)),
            "gold_threshold": None,
            "silver_threshold": None,
            "bronze_threshold": None,
            "median_threshold": None,
            "gold_medal": None,
            "silver_medal": None,
            "bronze_medal": None,
            "above_median": None,
            "is_lower_better": None,
        }
    finally:
        if tmp_discrete is not None:
            try:
                tmp_discrete.unlink(missing_ok=True)
            except Exception:
                pass
        if tmp_coerced is not None:
            try:
                tmp_coerced.unlink(missing_ok=True)
            except Exception:
                pass
        if tmp_filtered is not None:
            try:
                tmp_filtered.unlink(missing_ok=True)
            except Exception:
                pass


def _pick_leaderboard_score_column(fieldnames: list[str] | None) -> str | None:
    """
    Return the column name that contains the leaderboard score.

    MLE-bench leaderboard.csv files often use `_score` (leading underscore).
    """
    if not fieldnames:
        return None
    # Prefer exact score (after stripping leading underscores).
    for c in fieldnames:
        if c.strip().lower().lstrip("_") == "score":
            return c
    # Fallback: any column containing "score".
    for c in fieldnames:
        if "score" in c.strip().lower():
            return c
    return None


def setup_per_step_grading(cfg, competition_id: str = None):
    """
    Setup per-step grading if enabled in config.

    Returns:
        GradingCallback or None
    """
    if not hasattr(cfg, 'per_step_grading') or not cfg.per_step_grading.enabled:
        return None

    if competition_id is None:
        logger.warning("Per-step grading enabled but no competition_id provided")
        return None

    data_dir = Path(cfg.per_step_grading.mlebench_data_dir)

    # In mlebench containers, only the specific competition path is mounted:
    # /private/data/{competition_id}/prepared/private/
    # Permission errors can occur when checking paths - handle gracefully
    try:
        competition_private_dir = data_dir / competition_id / "prepared" / "private"
        dir_exists = competition_private_dir.exists() or data_dir.exists()
    except PermissionError:
        # Path exists but we can't stat it - proceed anyway, grading will fail gracefully if needed
        logger.info(f"Permission check failed for {data_dir}, proceeding with grading setup")
        dir_exists = True
    except Exception as e:
        logger.warning(f"Error checking MLE-bench data dir: {e}")
        dir_exists = False

    if not dir_exists:
        logger.warning(f"MLE-bench data dir not found: {data_dir}")
        return None

    logger.info(f"Per-step grading ENABLED for competition: {competition_id}")

    # Use methods from config (defaults handled by dataclass)
    methods = cfg.per_step_grading.methods or ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"]

    return GradingCallback(
        competition_id=competition_id,
        data_dir=data_dir,
        output_dir=Path(cfg.log_dir) / "per_step_grading",
        methods=methods,
        grade_every_n_steps=cfg.per_step_grading.grade_every_n_steps,
        save_every_n_steps=getattr(cfg.per_step_grading, "save_every_n_steps", 1),
    )


class GradingCallback:
    """
    Callback to grade all selection methods at each step.
    """

    def __init__(
        self,
        competition_id: str,
        data_dir: Path,
        output_dir: Path,
        methods: list[str] = None,
        grade_every_n_steps: int = 1,
        save_every_n_steps: int = 1,
    ):
        self.competition_id = competition_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.methods = methods or ["best_valid", "mean_minus_k_std", "maximin", "elite_maximin"]
        self.grade_every_n_steps = grade_every_n_steps
        self.save_every_n_steps = max(1, int(save_every_n_steps))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track grading history
        self.grading_history = {method: [] for method in self.methods}
        # Track buggy node counts per step
        self.buggy_node_history = []

        # Live CSV append state (stable schema for tail -f).
        self._grading_csv = self.output_dir / "grading_history.csv"
        self._buggy_csv = self.output_dir / "buggy_node_history.csv"
        self._grading_fields = [
            "method",
            "step",
            "node_id",
            "node_step",
            "validation_score",
            "cv_mean",
            "cv_std",
            "cv_folds",
            "test_score",
            "test_percentile",
            "test_gold_medal",
            "test_silver_medal",
            "test_bronze_medal",
            "test_above_median",
            "test_gold_threshold",
            "test_silver_threshold",
            "test_bronze_threshold",
            "test_median_threshold",
            "test_is_lower_better",
            "error",
        ]
        self._buggy_fields = [
            "step",
            "buggy_count",
            "good_count",
            "total_nodes",
            "buggy_ratio",
        ]
        self._written_grading: set[tuple[str, int]] = set()
        self._written_buggy: set[int] = set()
        self._load_existing_csv_keys()

    def _load_existing_csv_keys(self) -> None:
        try:
            if self._grading_csv.exists():
                with open(self._grading_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        method = r.get("method")
                        step = r.get("step")
                        if not method or step is None:
                            continue
                        try:
                            self._written_grading.add((method, int(step)))
                        except Exception:
                            continue
            if self._buggy_csv.exists():
                with open(self._buggy_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        step = r.get("step")
                        if step is None:
                            continue
                        try:
                            self._written_buggy.add(int(step))
                        except Exception:
                            continue
        except Exception:
            pass

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_bytes(data)
        tmp.replace(path)

    def _save_json_incremental(self) -> None:
        self._atomic_write_bytes(
            self.output_dir / "grading_history.json",
            json.dumps(self.grading_history, indent=2, default=_json_default).encode("utf-8"),
        )
        self._atomic_write_bytes(
            self.output_dir / "buggy_node_history.json",
            json.dumps(self.buggy_node_history, indent=2, default=_json_default).encode("utf-8"),
        )

    def _save_csv_incremental(self) -> None:
        # kept for backward compatibility; `_append_csv_incremental(step)` is used for live writing.
        return

    def _append_csv_incremental(self, step: int) -> None:
        if self._grading_csv.exists():
            write_header = False
        else:
            write_header = True

        with open(self._grading_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._grading_fields)
            if write_header:
                writer.writeheader()
            for method in self.methods:
                history = self.grading_history.get(method, [])
                if not history:
                    continue
                record = history[-1]
                if int(record.get("step", -1)) != step:
                    continue
                key = (method, step)
                if key in self._written_grading:
                    continue
                row = {"method": method}
                for field in self._grading_fields:
                    if field == "method":
                        continue
                    row[field] = record.get(field)
                writer.writerow(row)
                self._written_grading.add(key)

        if self._buggy_csv.exists():
            write_header = False
        else:
            write_header = True

        if self.buggy_node_history:
            last = self.buggy_node_history[-1]
            if int(last.get("step", -1)) == step and step not in self._written_buggy:
                with open(self._buggy_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._buggy_fields)
                    if write_header:
                        writer.writeheader()
                    row = {k: last.get(k) for k in self._buggy_fields}
                    writer.writerow(row)
                    self._written_buggy.add(step)

    def _flush_incremental(self, step: int) -> None:
        if step % self.save_every_n_steps != 0:
            return
        try:
            self._save_json_incremental()
            self._append_csv_incremental(step)
        except Exception as e:
            logger.warning(f"[Per-step grading] Failed to save incremental results: {e}")

    def on_step_complete(self, journal, step: int, workspace_dir: Path, cfg):
        """
        Called after each step completes.
        Grades all selection methods' current choices.
        """
        # Only grade every N steps
        if step % self.grade_every_n_steps != 0:
            return

        from .post_search import select_final_node_with_info

        logger.info(f"[Per-step grading] Step {step}: selecting and grading all methods")

        # Count nodes up to this step (do not use full journal-wide counts).
        nodes_up_to_step = [
            n
            for n in getattr(journal, "nodes", [])
            # `step` here is the *global step count* (i.e., len(journal) after the step),
            # so nodes available at this point are those with `node.step < step`.
            if isinstance(getattr(n, "step", None), int) and n.step < step
        ]
        total_nodes = len(nodes_up_to_step)
        buggy_count = sum(1 for n in nodes_up_to_step if getattr(n, "is_buggy", False) is True)
        # Treat anything not explicitly marked buggy (e.g., `None`) as non-buggy.
        good_count = total_nodes - buggy_count
        self.buggy_node_history.append({
            "step": step,
            "buggy_count": buggy_count,
            "good_count": good_count,
            "total_nodes": total_nodes,
            "buggy_ratio": buggy_count / total_nodes if total_nodes > 0 else 0.0,
        })
        logger.info(f"[Per-step grading] Step {step}: {buggy_count} buggy, {good_count} good, {total_nodes} total nodes")

        # Select current best for each method using post_search config
        from ..journal import Journal as JournalClass

        temp_journal = JournalClass()
        temp_journal.nodes = nodes_up_to_step

        selections = {}
        for method in self.methods:
            node, _info = select_final_node_with_info(
                temp_journal,
                selection=method,
                top_k=cfg.post_search.top_k,
                k_std=cfg.post_search.k_std,
                elite_top_k=cfg.post_search.elite_top_k,
                elite_ratio=cfg.post_search.elite_ratio,
                elite_k_std=cfg.post_search.elite_k_std,
                only_good=True,
            )
            selections[method] = node

        # Grade each selection
        for method, node in selections.items():
            if node is None:
                logger.warning(f"[Per-step grading] Step {step}, method {method}: no selection")
                self.grading_history[method].append({
                    "step": step,
                    "node_id": None,
                    "node_step": None,
                    "validation_score": None,
                    "cv_mean": None,
                    "cv_std": None,
                    "cv_folds": None,
                    "test_score": None,
                    "test_percentile": None,
                    "test_gold_medal": None,
                    "test_silver_medal": None,
                    "test_bronze_medal": None,
                    "test_above_median": None,
                    "test_gold_threshold": None,
                    "test_silver_threshold": None,
                    "test_bronze_threshold": None,
                    "test_median_threshold": None,
                    "test_is_lower_better": None,
                    "error": "No selection",
                })
                continue

            # Get submission path for this node
            submission_path = self._get_submission_path(node, workspace_dir)
            if submission_path is None or not submission_path.exists():
                logger.warning(f"[Per-step grading] Step {step}, method {method}: submission not found")
                cv_mean, cv_std, cv_folds = _compute_cv_stats(node)
                validation_score = cv_mean if cv_mean is not None else (node.valid_metric if hasattr(node, 'valid_metric') else (node.metric.value if node.metric else None))
                self.grading_history[method].append({
                    "step": step,
                    "node_id": node.id,
                    "node_step": getattr(node, "step", None),
                    "validation_score": validation_score,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "cv_folds": json.dumps(cv_folds) if isinstance(cv_folds, list) else None,
                    "test_score": None,
                    "test_percentile": None,
                    "test_gold_medal": None,
                    "test_silver_medal": None,
                    "test_bronze_medal": None,
                    "test_above_median": None,
                    "test_gold_threshold": None,
                    "test_silver_threshold": None,
                    "test_bronze_threshold": None,
                    "test_median_threshold": None,
                    "test_is_lower_better": None,
                    "error": "Submission file not found",
                })
                continue

            # Grade with MLE-bench
            grade_result = grade_submission_with_mlebench(
                submission_path,
                self.competition_id,
                self.data_dir,
            )

            # Record result
            cv_mean, cv_std, cv_folds = _compute_cv_stats(node)
            validation_score = cv_mean if cv_mean is not None else (node.valid_metric if hasattr(node, 'valid_metric') else (node.metric.value if node.metric else None))

            self.grading_history[method].append({
                "step": step,
                "node_id": node.id,
                "node_step": getattr(node, "step", None),
                "validation_score": validation_score,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_folds": json.dumps(cv_folds) if isinstance(cv_folds, list) else None,
                "test_score": grade_result["score"],
                "test_percentile": grade_result.get("percentile"),
                "test_gold_medal": grade_result.get("gold_medal"),
                "test_silver_medal": grade_result.get("silver_medal"),
                "test_bronze_medal": grade_result.get("bronze_medal"),
                "test_above_median": grade_result.get("above_median"),
                "test_gold_threshold": grade_result.get("gold_threshold"),
                "test_silver_threshold": grade_result.get("silver_threshold"),
                "test_bronze_threshold": grade_result.get("bronze_threshold"),
                "test_median_threshold": grade_result.get("median_threshold"),
                "test_is_lower_better": grade_result.get("is_lower_better"),
                "error": grade_result.get("error"),
            })

            if grade_result["score"] is not None:
                validation_str = (
                    f"{validation_score:.4f}"
                    if isinstance(validation_score, (int, float))
                    else "None"
                )
                logger.info(
                    f"[Per-step grading] Step {step}, method {method}: "
                    f"node {node.id}, validation={validation_str}, "
                    f"test={grade_result['score']:.4f}"
                )
            else:
                logger.debug(
                    f"[Per-step grading] Step {step}, method {method}: "
                    f"node {node.id}, grading failed: {grade_result.get('error')}"
                )

        # Persist progress so users can monitor in real time (e.g. `tail -f grading_history.csv`).
        self._flush_incremental(step)

    def save_results(self):
        """Save grading history and buggy node counts to JSON (CSV is appended live)."""
        # Save grading history JSON
        output_file = self.output_dir / "grading_history.json"
        with open(output_file, 'w') as f:
            json.dump(self.grading_history, f, indent=2, default=_json_default)

        logger.info(f"[Per-step grading] Saved grading history to {output_file}")

        # Save buggy node history JSON
        buggy_output_file = self.output_dir / "buggy_node_history.json"
        with open(buggy_output_file, 'w') as f:
            json.dump(self.buggy_node_history, f, indent=2, default=_json_default)

        logger.info(f"[Per-step grading] Saved buggy node history to {buggy_output_file}")
        # CSVs are written incrementally during the run to support `tail -f`.

    def _save_as_csv(self):
        """Save grading history as CSV for easy plotting"""
        try:
            import pandas as pd

            rows = []
            for method, history in self.grading_history.items():
                for record in history:
                    rows.append({
                        "method": method,
                        **record
                    })

            if rows:
                df = pd.DataFrame(rows)
                csv_file = self.output_dir / "grading_history.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"[Per-step grading] Saved grading history CSV to {csv_file}")
        except ImportError:
            logger.warning("[Per-step grading] pandas not available, skipping CSV export")
        except Exception as e:
            logger.warning(f"[Per-step grading] Failed to save CSV: {e}")

    def _save_buggy_node_csv(self):
        """Save buggy node history as CSV for easy plotting"""
        try:
            import pandas as pd

            if self.buggy_node_history:
                df = pd.DataFrame(self.buggy_node_history)
                csv_file = self.output_dir / "buggy_node_history.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"[Per-step grading] Saved buggy node history CSV to {csv_file}")
        except ImportError:
            logger.warning("[Per-step grading] pandas not available, skipping buggy node CSV export")
        except Exception as e:
            logger.warning(f"[Per-step grading] Failed to save buggy node CSV: {e}")

    def _get_submission_path(self, node, workspace_dir: Path) -> Path | None:
        """Get submission file path for a node"""
        if hasattr(node, 'submission_csv_path') and node.submission_csv_path:
            return Path(node.submission_csv_path)

        # Fallback: look for submission_{node_id}.csv
        submission_path = workspace_dir / "submission" / f"submission_{node.id}.csv"
        if submission_path.exists():
            return submission_path

        # Fallback: look for generic submission.csv (if node is the latest)
        submission_path = workspace_dir / "submission" / "submission.csv"
        if submission_path.exists():
            return submission_path

        return None
