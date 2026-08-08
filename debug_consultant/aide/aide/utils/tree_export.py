"""Export journal to an HTML visualization of tree + code + selection details."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph

from ..journal import Journal
from .post_search import select_final_node_with_info


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph."""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]."""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def strip_code_markers(code: str) -> str:
    """Remove markdown code block markers if present."""
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def _sel_obj(node):
    if node is None:
        return {"node": None, "cv_mean": None, "cv_std": None, "maximize": None}
    return {
        "node": getattr(node, "step", None),
        "cv_mean": getattr(node, "cv_mean", None),
        "cv_std": getattr(node, "cv_std", None),
        "maximize": getattr(getattr(node, "metric", None), "maximize", True)
        if node is not None
        else None,
    }


def cfg_to_tree_struct(cfg, jou: Journal):
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    metrics: list[float] = []
    metric_values: list[str] = []
    cv_std_values: list[float | None] = []
    cv_folds_values: list[list[float] | None] = []
    is_buggy: list[bool] = []
    metric_maximize: list[bool | None] = []

    for n in jou:
        maximize = getattr(getattr(n, "metric", None), "maximize", None)
        metric_maximize.append(maximize)
        node_is_buggy = bool(getattr(n, "is_buggy", False))
        is_buggy.append(node_is_buggy)

        raw_metric = getattr(getattr(n, "metric", None), "value", None)
        # Visualization sizing uses `1 - metrics[i]` as a relative-size signal.
        # For buggy / missing-metric nodes, use 1.0 so they render as the minimum size.
        metrics.append(float(raw_metric) if isinstance(raw_metric, (int, float)) else 1.0)

        cv_mean = getattr(n, "cv_mean", None)
        cv_std = getattr(n, "cv_std", None)
        cv_folds = getattr(n, "cv_folds", None)

        # Do not display scores for buggy nodes (debug nodes, timeouts, exceptions, etc.).
        if node_is_buggy:
            cv_std_values.append(None)
            cv_folds_values.append(None)
            metric_values.append("N/A")
            continue

        cv_std_values.append(cv_std if isinstance(cv_std, (int, float)) else None)
        cv_folds_values.append(cv_folds if isinstance(cv_folds, list) else None)

        if isinstance(cv_mean, (int, float)):
            if isinstance(cv_std, (int, float)) and cv_std > 0:
                metric_values.append(f"{cv_mean:.4f} ± {cv_std:.4f}")
            else:
                metric_values.append(f"{cv_mean:.4f}")
        else:
            valid = getattr(n, "valid_metric", None)
            if isinstance(valid, (int, float)):
                metric_values.append(f"{float(valid):.4f}")
            elif isinstance(raw_metric, (int, float)):
                metric_values.append(f"{float(raw_metric):.4f}")
            else:
                metric_values.append("N/A")

    # Selection summary: baseline (best_valid), configured post-search selector, and additional robust selectors.
    best_raw = jou.get_best_node(only_good=True)
    post_sel, post_info = select_final_node_with_info(
        jou,
        selection=getattr(cfg.post_search, "selection", "best_valid"),
        top_k=getattr(cfg.post_search, "top_k", 20),
        k_std=getattr(cfg.post_search, "k_std", 2.0),
        guard_std=getattr(cfg.post_search, "guard_std", 2.0),
        elite_top_k=getattr(cfg.post_search, "elite_top_k", 3),
        elite_ratio=getattr(cfg.post_search, "elite_ratio", 0.05),
        elite_k_std=getattr(cfg.post_search, "elite_k_std", 2.0),
        only_good=True,
    )

    mean_sel, mean_info = select_final_node_with_info(
        jou,
        selection="mean_minus_k_std",
        top_k=getattr(cfg.post_search, "top_k", 20),
        k_std=getattr(cfg.post_search, "k_std", 2.0),
        only_good=True,
    )

    maxmin_nf_sel, maxmin_nf_info = select_final_node_with_info(
        jou,
        selection="maximin_no_filter",
        top_k=getattr(cfg.post_search, "top_k", 20),
        guard_std=getattr(cfg.post_search, "guard_std", 2.0),
        only_good=True,
    )

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[textwrap.fill(n.plan, width=80) for n in jou.nodes],
        code=[strip_code_markers(n.code) for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=metrics,
        metric_values=metric_values,
        metric_maximize=metric_maximize,
        cv_std=cv_std_values,
        cv_folds=cv_folds_values,
        is_buggy=is_buggy,
        selected_for_summary=[False for _ in jou.nodes],
        seen_nodes_per_node=[[] for _ in jou.nodes],
        selection={
            "best_raw": _sel_obj(best_raw),
            "best_aux": _sel_obj(best_raw),
            "mean_minus_k_std": {**_sel_obj(mean_sel), "info": mean_info},
            "maximin_no_filter": {**_sel_obj(maxmin_nf_sel), "info": maxmin_nf_info},
            "post_search": {**_sel_obj(post_sel), "info": post_info},
        },
    )


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace("/*<placeholder>*/ {}", tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)
        return html


def generate(cfg, jou: Journal, out_path: Path):
    tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
    html = generate_html(tree_graph_str)
    with open(out_path, "w") as f:
        f.write(html)
