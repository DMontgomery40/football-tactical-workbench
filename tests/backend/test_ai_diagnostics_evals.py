from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.ai_diagnostics as ai_diagnostics  # noqa: E402
from app.ai_diagnostics import DiagnosticsAgentOutput  # noqa: E402


def _evaluate_diagnostics_contract_case(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        normalized = ai_diagnostics._normalize_diagnostics_agent_output(
            DiagnosticsAgentOutput.model_validate(candidate)
        )
    except Exception as exc:  # pragma: no cover - exercised through eval cases
        return {"accepted": False, "error": str(exc)}

    return {
        "accepted": True,
        "diagnostic_count": len(normalized.diagnostics),
        "warn_count": sum(1 for item in normalized.diagnostics if item.level == "warn"),
    }


def test_pydantic_evals_keeps_the_diagnostics_contract_cases_green() -> None:
    dataset = Dataset(
        name="diagnostics-contract-regressions",
        cases=[
            Case(
                name="valid_structured_output",
                inputs={
                    "summary_line": "Diagnostics look grounded in the implementation.",
                    "diagnostics": [
                        {
                            "level": "warn",
                            "title": "Detector class mapping drift",
                            "message": "Player classes do not line up with the emitted labels.",
                            "next_step": "Inspect the resolved class ids before touching tracker settings.",
                            "implementation_diagnosis": "The runtime filter is pointing at the wrong classes.",
                            "suggested_fix": "Resolve ids from checkpoint metadata or dataset YAML.",
                            "code_refs": ["backend/app/wide_angle.py::resolve_detector_spec"],
                            "evidence_keys": ["player_rows"],
                        },
                        {
                            "level": "good",
                            "title": "Tracker fallback",
                            "message": "Tracker wiring still initialized.",
                            "next_step": "Leave it unchanged until detections recover.",
                        },
                        {
                            "level": "good",
                            "title": "Projection context",
                            "message": "Projection stayed downstream-empty.",
                            "next_step": "Revisit projection only after detections recover.",
                        },
                    ],
                },
                expected_output={"accepted": True, "diagnostic_count": 3, "warn_count": 1},
            ),
            Case(
                name="invalid_warn_output",
                inputs={
                    "summary_line": "Diagnostics contract is broken.",
                    "diagnostics": [
                        {
                            "level": "warn",
                            "title": "Missing implementation details",
                            "message": "The output skipped code-level guidance.",
                            "next_step": "Reject the response instead of showing it.",
                            "suggested_fix": "Require the missing fields.",
                            "code_refs": ["backend/app/wide_angle.py::resolve_detector_spec"],
                        },
                        {
                            "level": "good",
                            "title": "Context 1",
                            "message": "Fallback exists.",
                            "next_step": "Use it only as backup.",
                        },
                        {
                            "level": "good",
                            "title": "Context 2",
                            "message": "Overlay still rendered.",
                            "next_step": "Keep the rest of the stack unchanged.",
                        },
                    ],
                },
                expected_output={"accepted": False, "error": "Warn diagnostic 'Missing implementation details' is missing implementation_diagnosis"},
            ),
        ],
        evaluators=[EqualsExpected()],
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        report = dataset.evaluate_sync(_evaluate_diagnostics_contract_case, progress=False)
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert report.averages().assertions == 1.0
