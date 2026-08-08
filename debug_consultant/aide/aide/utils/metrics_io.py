from __future__ import annotations

import json
from typing import Any


_PREFIXES = (
    "AIDE_METRICS=",
    "AIDE_METRICS_JSON=",
)


def parse_aide_metrics(term_out: str) -> dict[str, Any] | None:
    """
    Parse structured metrics emitted by a runfile.

    Supported formats (single line):
      - AIDE_METRICS=<json>
      - AIDE_METRICS_JSON=<json>
    """
    for line in term_out.splitlines():
        line = line.strip()
        for prefix in _PREFIXES:
            if line.startswith(prefix):
                payload = line[len(prefix) :].strip()
                if not payload:
                    return None
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _coerce_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _coerce_float_list(x: Any) -> list[float] | None:
    if x is None:
        return None
    if isinstance(x, str):
        return None
    if not isinstance(x, list):
        return None
    out: list[float] = []
    for v in x:
        fv = _coerce_float(v)
        if fv is None:
            return None
        out.append(fv)
    return out


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize common keys used across experiment scripts.

    Accepted keys:
      - valid / val / metric
      - lower_is_better (bool)
      - cv_mean / cv_std / cv_folds (optional)
    """
    valid = (
        _coerce_float(metrics.get("valid"))
        if "valid" in metrics
        else _coerce_float(metrics.get("val"))
        if "val" in metrics
        else _coerce_float(metrics.get("metric"))
    )
    lower_is_better = metrics.get("lower_is_better")
    if not isinstance(lower_is_better, bool):
        lower_is_better = None

    # CV fields (optional). If not provided, callers may treat `valid` as a proxy for `cv_mean`.
    cv_mean = _coerce_float(metrics.get("cv_mean"))
    cv_std = _coerce_float(metrics.get("cv_std"))
    cv_folds = (
        _coerce_float_list(metrics.get("cv_folds"))
        or _coerce_float_list(metrics.get("folds"))
    )

    return {
        "valid": valid,
        "lower_is_better": lower_is_better,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_folds": cv_folds,
    }
