from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


class BenchmarkEvaluationUnavailable(RuntimeError):
    """Raised when a suite cannot run because required tooling or data is missing."""


class BenchmarkPredictionExportUnavailable(BenchmarkEvaluationUnavailable):
    """Raised when evaluator-ready prediction artifacts could not be exported."""

    def __init__(
        self,
        message: str,
        *,
        artifacts: dict[str, Any] | None = None,
        raw_result: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.artifacts = dict(artifacts or {})
        self.raw_result = dict(raw_result or {})


def metric_value(
    value: float | int | None,
    *,
    label: str | None = None,
    unit: str = "",
    precision: int = 4,
    sort_value: float | int | None = None,
) -> dict[str, Any]:
    numeric = float(value) if value is not None else None
    return {
        "label": label or "",
        "value": numeric,
        "display_value": None if numeric is None else f"{numeric:.{precision}f}{unit}",
        "unit": unit,
        "sort_value": float(sort_value if sort_value is not None else numeric) if numeric is not None else None,
        "is_na": numeric is None,
    }


def na_metric(*, label: str | None = None) -> dict[str, Any]:
    return {
        "label": label or "",
        "value": None,
        "display_value": "N/A",
        "unit": "",
        "sort_value": None,
        "is_na": True,
    }


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def runtime_environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
