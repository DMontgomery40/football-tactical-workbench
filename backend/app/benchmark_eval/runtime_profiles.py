from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]


RUNTIME_PROFILES: dict[str, dict[str, Any]] = {
    "backend_default": {
        "profile_id": "backend_default",
        "label": "Backend default runtime",
        "mode": "current_process",
        "required_imports": [],
        "required_python_prefix": None,
        "max_numpy_major": None,
        "expected_env_dir": None,
    },
    "tracklab_gamestate_py39_np1": {
        "profile_id": "tracklab_gamestate_py39_np1",
        "label": "TrackLab + sn-gamestate (Python 3.9 / NumPy <2)",
        "mode": "isolated_python",
        "required_imports": ["tracklab", "sn_gamestate"],
        "required_python_prefix": "3.9",
        "max_numpy_major": 1,
        "expected_env_dir": BACKEND_DIR / ".venv-benchmark-gamestate-py39",
    },
    "sn_calibration_legacy": {
        "profile_id": "sn_calibration_legacy",
        "label": "sn-calibration legacy runtime",
        "mode": "isolated_python",
        "required_imports": [],
        "required_python_prefix": None,
        "max_numpy_major": 1,
        "expected_env_dir": BACKEND_DIR / ".venv-benchmark-calibration-legacy",
    },
    "modern_action_spotting": {
        "profile_id": "modern_action_spotting",
        "label": "Modern action spotting runtime",
        "mode": "isolated_python",
        "required_imports": [],
        "required_python_prefix": None,
        "max_numpy_major": None,
        "expected_env_dir": BACKEND_DIR / ".venv-benchmark-action-spotting",
    },
    "footpass_eval": {
        "profile_id": "footpass_eval",
        "label": "FOOTPASS evaluation runtime",
        "mode": "isolated_python",
        "required_imports": ["h5py"],
        "required_python_prefix": "3.11",
        "max_numpy_major": 1,
        "expected_env_dir": BACKEND_DIR / ".venv-benchmark-footpass",
    },
}


def runtime_profile(runtime_key: str) -> dict[str, Any]:
    try:
        return dict(RUNTIME_PROFILES[runtime_key])
    except KeyError as exc:  # pragma: no cover - invalid developer wiring
        raise KeyError(f"Unknown benchmark runtime profile: {runtime_key}") from exc


def runtime_python_executable(profile: dict[str, Any]) -> str | None:
    mode = str(profile.get("mode") or "")
    if mode == "current_process":
        return sys.executable
    expected_env_dir = profile.get("expected_env_dir")
    if not expected_env_dir:
        return None
    expected_env_path = Path(str(expected_env_dir)).expanduser().resolve()
    candidate = expected_env_path / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def _current_process_probe(profile: dict[str, Any]) -> dict[str, Any]:
    import numpy as np  # local import to avoid forcing it in helper subprocess snippets

    import_status: dict[str, bool] = {}
    for module_name in profile.get("required_imports") or []:
        import_status[str(module_name)] = importlib.util.find_spec(str(module_name)) is not None
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "import_status": import_status,
    }


def _isolated_probe(python_executable: str, required_imports: list[str]) -> dict[str, Any]:
    snippet = """
import importlib.util, json, sys
try:
    import numpy as np
    numpy_version = np.__version__
except Exception:
    numpy_version = None
modules = json.loads(sys.argv[1])
print(json.dumps({
    "python_version": sys.version.split()[0],
    "numpy_version": numpy_version,
    "import_status": {name: importlib.util.find_spec(name) is not None for name in modules},
}))
""".strip()
    completed = subprocess.run(
        [python_executable, "-c", snippet, json.dumps(required_imports)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(stderr or "runtime probe failed")
    return json.loads(completed.stdout)


def _numpy_major(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(str(value).split(".")[0])
    except Exception:
        return None


def _probe_failures(profile: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required_python_prefix = str(profile.get("required_python_prefix") or "").strip()
    python_version = str(payload.get("python_version") or "").strip()
    if required_python_prefix and not python_version.startswith(required_python_prefix):
        failures.append(
            f"requires Python {required_python_prefix}.x but the target runtime reports Python {python_version or 'unknown'}."
        )

    max_numpy_major = profile.get("max_numpy_major")
    numpy_version = str(payload.get("numpy_version") or "").strip() or None
    if isinstance(max_numpy_major, int):
        numpy_major = _numpy_major(numpy_version)
        if numpy_major is None:
            failures.append("could not confirm a NumPy installation in the target runtime.")
        elif numpy_major > max_numpy_major:
            failures.append(
                f"requires NumPy major <= {max_numpy_major}, but the target runtime reports NumPy {numpy_version}."
            )

    import_status = payload.get("import_status")
    if isinstance(import_status, dict):
        for module_name, available in import_status.items():
            if not bool(available):
                failures.append(f"missing required import '{module_name}' in the target runtime.")
    return failures


@lru_cache(maxsize=None)
def probe_runtime_profile(runtime_key: str) -> dict[str, Any]:
    profile = runtime_profile(runtime_key)
    expected_env_dir = profile.get("expected_env_dir")
    python_executable = runtime_python_executable(profile)

    probe: dict[str, Any] = {
        "profile_id": profile["profile_id"],
        "label": profile["label"],
        "mode": profile["mode"],
        "expected_env_dir": str(Path(str(expected_env_dir)).expanduser().resolve()) if expected_env_dir else None,
        "python_executable": python_executable,
        "required_python_prefix": profile.get("required_python_prefix"),
        "max_numpy_major": profile.get("max_numpy_major"),
        "required_imports": list(profile.get("required_imports") or []),
        "available": False,
        "python_version": None,
        "numpy_version": None,
        "import_status": {},
        "missing_reasons": [],
    }

    if python_executable is None:
        probe["missing_reasons"] = [
            f"Runtime profile '{profile['profile_id']}' is not configured. Expected interpreter under "
            f"{probe['expected_env_dir'] or 'an explicit configured env path'}."
        ]
        return probe

    try:
        payload = (
            _current_process_probe(profile)
            if profile["mode"] == "current_process"
            else _isolated_probe(python_executable, list(profile.get("required_imports") or []))
        )
    except Exception as exc:
        probe["missing_reasons"] = [
            f"Runtime profile '{profile['profile_id']}' could not be probed via {python_executable}: {exc}"
        ]
        return probe

    probe["python_version"] = payload.get("python_version")
    probe["numpy_version"] = payload.get("numpy_version")
    probe["import_status"] = payload.get("import_status") or {}
    probe["missing_reasons"] = _probe_failures(profile, payload)
    probe["available"] = len(probe["missing_reasons"]) == 0
    return probe


def runtime_unavailable_message(runtime_key: str) -> str:
    probe = probe_runtime_profile(runtime_key)
    if probe.get("available"):
        return ""
    details = " ".join(str(item).strip() for item in (probe.get("missing_reasons") or []) if str(item).strip())
    return (
        f"Runtime profile '{probe['profile_id']}' ({probe['label']}) is unavailable. {details}".strip()
    )
