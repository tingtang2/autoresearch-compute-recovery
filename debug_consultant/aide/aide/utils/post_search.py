from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math

from ..journal import Journal, Node


def _score(value: float, maximize: bool) -> float:
    return value if maximize else -value


def _node_opt_dir(journal: Journal) -> bool:
    best = journal.get_best_node(only_good=True)
    if best is None or best.metric is None or best.metric.maximize is None:
        return True
    return bool(best.metric.maximize)


def _best_by(values: Iterable[tuple[Node, float]], maximize: bool) -> Node | None:
    best_node: Node | None = None
    best_score: float | None = None
    for node, v in values:
        s = _score(v, maximize)
        if best_score is None or s > best_score:
            best_score = s
            best_node = node
    return best_node


def _topk(nodes: list[Node], maximize: bool, top_k: int) -> list[Node]:
    if top_k <= 0 or top_k >= len(nodes):
        return nodes

    def key(n: Node) -> float:
        v = _cv_mean(n)
        if v is None:
            return float("-inf") if maximize else float("inf")
        return v

    # Sort by score in the direction of optimization.
    rev = maximize
    return sorted(nodes, key=key, reverse=rev)[:top_k]


def _worst_fold(cv_folds: list[float], maximize: bool) -> float:
    return min(cv_folds) if maximize else max(cv_folds)


def _clean_cv_folds(node: Node) -> list[float] | None:
    folds = getattr(node, "cv_folds", None)
    if not isinstance(folds, list) or not folds:
        return None
    out: list[float] = []
    for v in folds:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            out.append(f)
    return out or None


def _cv_mean_std(node: Node) -> tuple[float | None, float | None]:
    """
    Compute CV mean/std from `cv_folds` when available (source of truth).
    Falls back to stored `cv_mean`/`cv_std`/`valid_metric`/`metric.value`.
    """
    folds = _clean_cv_folds(node)
    if folds is not None:
        m = sum(folds) / len(folds)
        if len(folds) == 1:
            return m, 0.0
        var = sum((x - m) ** 2 for x in folds) / len(folds)
        return m, var**0.5

    mean_v = getattr(node, "cv_mean", None)
    if mean_v is None:
        mean_v = getattr(node, "valid_metric", None)
    if mean_v is None:
        metric = getattr(node, "metric", None)
        mean_v = getattr(metric, "value", None) if metric is not None else None

    std_v = getattr(node, "cv_std", None)
    return mean_v, std_v


def _cv_mean(node: Node) -> float | None:
    return _cv_mean_std(node)[0]


def _cv_std(node: Node) -> float | None:
    return _cv_mean_std(node)[1]


@dataclass
class PostSearchInfo:
    method: str
    maximize: bool
    n_candidates: int
    n_with_cv: int
    population_mean_cv_mean: float | None = None
    population_std_cv_mean: float | None = None
    threshold: float | None = None
    elite_size: int | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "maximize": self.maximize,
            "n_candidates": self.n_candidates,
            "n_with_cv": self.n_with_cv,
            "population_mean_cv_mean": self.population_mean_cv_mean,
            "population_std_cv_mean": self.population_std_cv_mean,
            "threshold": self.threshold,
            "elite_size": self.elite_size,
            "notes": self.notes,
            # Backwards/visualization-friendly aliases (used by the HTML template).
            "mean_all": self.population_mean_cv_mean,
            "std_all": self.population_std_cv_mean,
            "candidates": self.elite_size,
        }


def select_final_node(
    journal: Journal,
    selection: str = "best_valid",
    top_k: int = 20,
    *,
    k_std: float = 2.0,
    z_threshold: float = 2.0,
    guard_std: float = 2.0,
    elite_top_k: int = 3,
    elite_ratio: float = 0.05,
    elite_k_std: float = 2.0,
    only_good: bool = True,
) -> Node | None:
    node, _info = select_final_node_with_info(
        journal,
        selection=selection,
        top_k=top_k,
        k_std=k_std,
        z_threshold=z_threshold,
        guard_std=guard_std,
        elite_top_k=elite_top_k,
        elite_ratio=elite_ratio,
        elite_k_std=elite_k_std,
        only_good=only_good,
    )
    return node


def select_final_node_with_info(
    journal: Journal,
    selection: str = "best_valid",
    top_k: int = 20,
    *,
    k_std: float = 2.0,
    z_threshold: float = 2.0,
    guard_std: float = 2.0,
    elite_top_k: int = 3,
    elite_ratio: float = 0.05,
    elite_k_std: float = 2.0,
    only_good: bool = True,
) -> tuple[Node | None, dict[str, Any]]:
    """
    Select which node should be treated as the final solution after search.

    This is intentionally separate from the search policy (which optimizes `node.metric`).
    """
    # Backwards compatibility: keep old selector names / aliases.
    if selection in {"stat_maximin", "best_train", "best_test", "gap_penalized"}:
        selection = "elite_maximin" if selection == "stat_maximin" else "best_valid"
    if selection in {"max_min", "maximin_all"}:
        selection = "maximin_no_filter"

    nodes = journal.good_nodes if only_good else journal.nodes
    if not nodes:
        return None, {"method": selection, "notes": "no_nodes"}

    maximize = _node_opt_dir(journal)
    candidates = nodes if selection == "maximin_no_filter" else _topk(nodes, maximize, top_k)

    info = PostSearchInfo(
        method=selection,
        maximize=maximize,
        n_candidates=len(candidates),
        n_with_cv=sum(1 for n in candidates if _clean_cv_folds(n) is not None or n.cv_mean is not None),
    )

    if selection == "best_valid":
        # Choose best node by CV mean (from folds when available), excluding buggy nodes
        non_buggy = [n for n in candidates if n.is_buggy is False]
        if not non_buggy:
            return None, {**info.to_dict(), "notes": "all_candidates_buggy"}
        scored: list[tuple[Node, float]] = []
        for n in non_buggy:
            v = _cv_mean(n)
            if v is None:
                continue
            scored.append((n, v))
        best = _best_by(scored, maximize)
        if best is None:
            return None, {**info.to_dict(), "notes": "no_candidates_with_valid_score"}
        return best, info.to_dict()

    if selection == "last_good":
        eligible_ids = {n.id for n in nodes}
        for n in reversed(journal.nodes):
            if n.id not in eligible_ids:
                continue
            if (not only_good) or (n.is_buggy is False):
                return n, info.to_dict()
        return None, {**info.to_dict(), "notes": "no_last_good"}

    if selection == "mean_minus_k_std":
        scored: list[tuple[Node, float]] = []
        for n in candidates:
            # Skip buggy nodes - never select nodes with metric=1.0, invalid cv_folds, etc.
            if n.is_buggy is not False:
                continue
            mean_v = _cv_mean(n)
            std_v = _cv_std(n)
            # Require BOTH mean and std - drop nodes missing either
            if mean_v is None or std_v is None:
                continue
            # maximize: lower bound; minimize: upper bound
            robust = mean_v - (k_std * std_v) if maximize else mean_v + (k_std * std_v)
            scored.append((n, robust))

        best = _best_by(scored, maximize)
        if best is not None:
            return best, {**info.to_dict(), "k_std": k_std, "n_scored": len(scored)}

        # No nodes with complete CV data - return None (fail clearly)
        return None, {**info.to_dict(), "k_std": k_std, "notes": "no_nodes_with_cv_std"}

    if selection == "maximin":
        scored: list[tuple[Node, float]] = []
        for n in candidates:
            # Skip buggy nodes - never select nodes with metric=1.0, invalid cv_folds, etc.
            if n.is_buggy is not False:
                continue
            # Require cv_folds - drop nodes without them
            if not n.cv_folds:
                continue
            scored.append((n, _worst_fold(n.cv_folds, maximize)))

        best = _best_by(scored, maximize)
        if best is not None:
            best_worst = _worst_fold(best.cv_folds, maximize) if best.cv_folds else None
            return best, {**info.to_dict(), "n_scored": len(scored), "best_worst_case": best_worst}

        # No nodes with cv_folds - fail clearly
        return None, {**info.to_dict(), "notes": "no_nodes_with_cv_folds"}

    if selection == "maximin_no_filter":
        scored: list[tuple[Node, float]] = []
        for n in candidates:
            # Skip buggy nodes - never select nodes with metric=1.0, invalid cv_folds, etc.
            if n.is_buggy is not False:
                continue
            # Require cv_folds - drop nodes without them
            if not n.cv_folds:
                continue
            scored.append((n, _worst_fold(n.cv_folds, maximize)))

        best = _best_by(scored, maximize)
        if best is not None:
            best_worst = _worst_fold(best.cv_folds, maximize) if best.cv_folds else None
            return best, {**info.to_dict(), "n_scored": len(scored), "best_worst_case": best_worst}

        # No nodes with cv_folds - fail clearly
        return None, {**info.to_dict(), "notes": "no_nodes_with_cv_folds"}

    if selection == "elite_maximin":
        # Elite filter: max(top_k, ratio%, best±k_std)
        # Takes the UNION of all three filters (largest elite set)
        # Skip buggy nodes when building elite set
        cv_means = [_cv_mean(n) for n in candidates if _cv_mean(n) is not None and n.is_buggy is False]
        if not cv_means:
            return None, {**info.to_dict(), "notes": "no_non_buggy_nodes_with_cv_mean"}

        candidates_with_cv = [n for n in candidates if _cv_mean(n) is not None and n.is_buggy is False]
        if not candidates_with_cv:
            return None, {**info.to_dict(), "notes": "no_non_buggy_cv_candidates"}

        # Sort by cv_mean (descending for maximize, ascending for minimize)
        candidates_with_cv.sort(key=lambda n: _cv_mean(n), reverse=maximize)

        # Calculate all three filter sizes
        n_total = len(candidates_with_cv)

        # Filter 1: Top-K (minimum elite size)
        size_topk = min(elite_top_k, n_total)

        # Filter 2: Ratio (5%)
        size_ratio = max(1, int(n_total * elite_ratio))

        # Filter 3: Best ± k_std (statistical filter)
        best_cv_mean = _cv_mean(candidates_with_cv[0])
        if best_cv_mean is None:
            return None, {**info.to_dict(), "notes": "best_candidate_missing_cv_mean"}
        mean_all = sum(cv_means) / len(cv_means)
        var_all = sum((v - mean_all) ** 2 for v in cv_means) / len(cv_means)
        std_all = var_all**0.5

        if maximize:
            threshold = best_cv_mean - (elite_k_std * std_all)
            size_stat = sum(1 for n in candidates_with_cv if (_cv_mean(n) is not None and _cv_mean(n) >= threshold))
        else:
            threshold = best_cv_mean + (elite_k_std * std_all)
            size_stat = sum(1 for n in candidates_with_cv if (_cv_mean(n) is not None and _cv_mean(n) <= threshold))

        # Take the MAXIMUM of all three
        elite_size = max(size_topk, size_ratio, size_stat)
        elite = candidates_with_cv[:elite_size]

        info.population_mean_cv_mean = mean_all
        info.population_std_cv_mean = std_all
        info.threshold = threshold
        info.elite_size = elite_size
        info.notes = f"max(top{size_topk}, ratio{size_ratio}, stat{size_stat})={elite_size}"

        if not elite:
            return None, {**info.to_dict(), "notes": "empty_elite_set"}

        # Now apply maximin on elite set: maximize min(cv_folds)
        scored_folds: list[tuple[Node, float]] = []
        for n in elite:
            # Double-check: skip buggy nodes (defensive, should already be filtered)
            if n.is_buggy is not False:
                continue
            if n.cv_folds:
                scored_folds.append((n, _worst_fold(n.cv_folds, maximize)))

        if scored_folds:
            best = _best_by(scored_folds, maximize)
            best_worst = None
            if best is not None and best.cv_folds:
                best_worst = _worst_fold(best.cv_folds, maximize)
            return best, {
                **info.to_dict(),
                "elite_k_std": elite_k_std,
                "elite_ratio": elite_ratio,
                "size_topk": size_topk,
                "size_ratio": size_ratio,
                "size_stat": size_stat,
                "best_worst_case": best_worst,
            }

        # No cv_folds in elite set - fail clearly, no fallback
        return None, {
            **info.to_dict(),
            "notes": f"{info.notes}_no_nodes_with_cv_folds",
        }

    # unknown strategy => safe fallback
    return journal.get_best_node(only_good=only_good), {**info.to_dict(), "notes": "unknown_selection_fallback"}
