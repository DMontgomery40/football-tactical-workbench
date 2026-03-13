from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable
from .runtime_profiles import probe_runtime_profile, runtime_profile, runtime_unavailable_message


def run_external_json_command(
    *,
    command: list[str],
    cwd: str | Path,
    artifacts_dir: str | Path,
    runtime_key: str,
) -> dict[str, Any]:
    resolved_cwd = Path(cwd).expanduser().resolve()
    result_path = Path(artifacts_dir).expanduser().resolve() / "external_result.json"
    if not resolved_cwd.exists():
        result_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(resolved_cwd),
                    "runtime_profile": {"profile_id": runtime_key, "available": False},
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"Required evaluator checkout is missing: {resolved_cwd}",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise BenchmarkEvaluationUnavailable(f"Required evaluator checkout is missing: {resolved_cwd}")
    runtime_details = probe_runtime_profile(runtime_key)
    if not runtime_details.get("available"):
        result_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(resolved_cwd),
                    "runtime_profile": runtime_details,
                    "returncode": None,
                    "stdout": "",
                    "stderr": runtime_unavailable_message(runtime_key),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise BenchmarkEvaluationUnavailable(runtime_unavailable_message(runtime_key))
    resolved_command = list(command)
    if resolved_command and resolved_command[0] in {"python", "python3"}:
        resolved_command[0] = str(runtime_details.get("python_executable"))
    completed = subprocess.run(
        resolved_command,
        cwd=str(resolved_cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "command": resolved_command,
        "cwd": str(resolved_cwd),
        "runtime_profile": runtime_details,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip() or "External evaluator failed."
        raise BenchmarkEvaluationUnavailable(message)
    try:
        parsed = json.loads(completed.stdout)
    except Exception as exc:
        raise BenchmarkEvaluationUnavailable(f"External evaluator did not emit valid JSON: {exc}") from exc
    if isinstance(parsed, dict):
        parsed.setdefault("_runner", runtime_details)
        parsed.setdefault("_external_result_path", str(result_path))
        return parsed
    return {
        "raw": parsed,
        "_runner": runtime_details,
        "_external_result_path": str(result_path),
    }
