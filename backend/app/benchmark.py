"""Benchmark Lab V2 -- multi-suite, capability-aware benchmarking."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from huggingface_hub import hf_hub_download

from app.benchmark_catalog import list_assets as benchmark_list_assets
from app.benchmark_catalog import list_recipes as benchmark_list_recipes
from app.benchmark_catalog import list_recipes
from app.benchmark_eval import run_suite_evaluation
from app.benchmark_eval.common import (
    BenchmarkEvaluationUnavailable,
    BenchmarkPredictionExportUnavailable,
    metric_value,
    na_metric,
)
from app.benchmark_eval.prediction_exports import prepare_prediction_exports
from app.benchmark_provenance import build_benchmark_provenance
from app.benchmark_suites import (
    build_suite_dataset_state,
    get_suite_definition,
    list_suite_dataset_states,
    list_suite_definitions,
)
from app.training_provenance import probe_dvc_runtime, resolve_dvc_tracking, utc_now_iso
from app.wide_angle import resolve_detector_spec, resolve_model_path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
CLIP_CACHE_DIR = BENCHMARKS_DIR / "_clip_cache"
IMPORTS_DIR = BENCHMARKS_DIR / "_imports"
RUNS_DIR = BASE_DIR / "runs"
BENCHMARK_CLIP_FILENAME = "benchmark_clip.mp4"
BENCHMARK_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1

BENCHMARK_RUNTIME_PROFILE: dict[str, Any] = {
    "pipeline": "classic",
    "keypoint_model": "soccana_keypoint",
    "tracker_mode": "hybrid_reid",
    "include_ball": True,
    "player_conf": 0.25,
    "ball_conf": 0.20,
    "iou": 0.50,
}

BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clip_status() -> dict[str, Any]:
    clip_path = CLIP_CACHE_DIR / BENCHMARK_CLIP_FILENAME
    exists = clip_path.exists() and clip_path.stat().st_size > 0
    size_mb = round(clip_path.stat().st_size / (1024 * 1024), 1) if exists else None
    return {
        "ready": exists,
        "path": str(clip_path) if exists else None,
        "size_mb": size_mb,
        "cache_dir": str(CLIP_CACHE_DIR),
        "expected_filename": BENCHMARK_CLIP_FILENAME,
        "dvc": resolve_dvc_tracking(str(clip_path)) if exists else None,
        "suite_id": "ops.clip_review_v1",
        "note": None if exists else (
            "Place a benchmark clip at "
            f"{clip_path} or use /api/benchmark/ensure-clip to provide one."
        ),
    }


def ensure_clip(source_path: str = "") -> dict[str, Any]:
    dest = CLIP_CACHE_DIR / BENCHMARK_CLIP_FILENAME
    source = Path(source_path.strip()).expanduser().resolve() if source_path.strip() else None
    if source and source.exists() and source.is_file():
        shutil.copy2(str(source), str(dest))
        return clip_status()
    if dest.exists() and dest.stat().st_size > 0:
        return clip_status()
    return {
        "ready": False,
        "path": None,
        "cache_dir": str(CLIP_CACHE_DIR),
        "expected_filename": BENCHMARK_CLIP_FILENAME,
        "error": (
            "No benchmark clip available. Provide a local path via "
            "'source_path' or manually place a clip at "
            f"{dest}"
        ),
    }


def _load_import_manifest() -> list[dict[str, Any]]:
    manifest_path = IMPORTS_DIR / "imports.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _save_import_manifest(records: list[dict[str, Any]]) -> None:
    manifest_path = IMPORTS_DIR / "imports.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def validate_checkpoint(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return {"valid": False, "error": f"File not found: {resolved}"}
    try:
        spec = resolve_detector_spec(str(resolved))
        return {
            "valid": True,
            "path": str(resolved),
            "class_names_source": spec.get("class_names_source", ""),
            "player_class_ids": spec.get("player_class_ids", []),
            "ball_class_ids": spec.get("ball_class_ids", []),
            "referee_class_ids": spec.get("referee_class_ids", []),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def import_local_checkpoint(checkpoint_path: str, label: str = "") -> dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Checkpoint not found: {resolved}")
    validation = validate_checkpoint(str(resolved))
    if not validation["valid"]:
        raise ValueError(f"Checkpoint validation failed: {validation.get('error')}")

    import_id = f"import_{uuid.uuid4().hex[:8]}"
    dest_dir = IMPORTS_DIR / import_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "best.pt"
    shutil.copy2(str(resolved), str(dest_path))

    records = _load_import_manifest()
    record = {
        "id": import_id,
        "label": label.strip() or import_id,
        "path": str(dest_path),
        "origin": str(resolved),
        "imported_at": utc_now_iso(),
        "origin_dvc": resolve_dvc_tracking(str(resolved)),
    }
    records.append(record)
    _save_import_manifest(records)

    assets = benchmark_list_assets()
    recipes = benchmark_list_recipes()
    asset_id = f"detector.{import_id}"
    return {
        "asset": next((asset for asset in assets if str(asset.get("asset_id")) == asset_id), None),
        "recipes": [recipe for recipe in recipes if asset_id in (recipe.get("source_asset_ids") or [])],
    }


def import_hf_checkpoint(repo_id: str, filename: str = "best.pt", label: str = "") -> dict[str, Any]:
    import_id = f"import_hf_{uuid.uuid4().hex[:8]}"
    dest_dir = IMPORTS_DIR / import_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))
    dest_path = Path(downloaded).resolve()
    validation = validate_checkpoint(str(dest_path))
    if not validation["valid"]:
        shutil.rmtree(str(dest_dir), ignore_errors=True)
        raise ValueError(f"HF checkpoint validation failed: {validation.get('error')}")
    records = _load_import_manifest()
    record = {
        "id": import_id,
        "label": label.strip() or f"{repo_id}/{filename}",
        "path": str(dest_path),
        "origin": f"hf://{repo_id}/{filename}",
        "imported_at": utc_now_iso(),
    }
    records.append(record)
    _save_import_manifest(records)
    assets = benchmark_list_assets()
    recipes = benchmark_list_recipes()
    asset_id = f"detector.{import_id}"
    return {
        "asset": next((asset for asset in assets if str(asset.get("asset_id")) == asset_id), None),
        "recipes": [recipe for recipe in recipes if asset_id in (recipe.get("source_asset_ids") or [])],
    }


def list_assets() -> list[dict[str, Any]]:
    return benchmark_list_assets()


def list_candidates() -> list[dict[str, Any]]:
    # Backward-compatible name for older callers.
    return list_assets()


def list_recipes_public() -> list[dict[str, Any]]:
    return list_recipes()


def list_suites() -> list[dict[str, Any]]:
    return list_suite_definitions()


def list_dataset_states() -> list[dict[str, Any]]:
    states = list_suite_dataset_states()
    operational_clip = clip_status()
    for state in states:
        if state.get("suite_id") == "ops.clip_review_v1":
            state["ready"] = bool(operational_clip.get("ready"))
            state["readiness_status"] = "ready" if operational_clip.get("ready") else "blocked"
            state["dataset_root"] = operational_clip.get("path")
            state["dataset_exists"] = bool(operational_clip.get("ready"))
            state["dataset_dvc"] = operational_clip.get("dvc")
            state["note"] = operational_clip.get("note")
            state["blockers"] = [] if operational_clip.get("ready") else [str(operational_clip.get("note") or "Benchmark clip is unavailable.")]
            state["manifest_summary"] = {
                "kind": "clip_manifest",
                "split": None,
                "selection": "single_clip",
                "item_count": 1 if operational_clip.get("ready") else 0,
                "class_count": None,
                "task_coverage": ["operational_review"],
            }
    return states


def benchmark_config_snapshot() -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suites": list_suites(),
        "dataset_states": list_dataset_states(),
        "assets": list_assets(),
        "recipes": list_recipes_public(),
        "dvc_runtime": probe_dvc_runtime(),
        "legacy_clip_status": clip_status(),
        "benchmarks_dir": str(BENCHMARKS_DIR),
    }


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            row[key] = value.get("value")
        else:
            row[key] = value
    return row


def _blocked_suite_status(suite_dataset_state: dict[str, Any]) -> str:
    status = str(suite_dataset_state.get("readiness_status") or "").strip().lower()
    if status:
        return status
    note = str(suite_dataset_state.get("note") or "").lower()
    if "still blocked" in note or "does not yet" in note:
        return "blocked"
    return "blocked"


def hydrate_legacy_benchmark(data: dict[str, Any], benchmark_dir: Path) -> dict[str, Any]:
    candidates = list(data.get("candidates") or [])
    leaderboard = list(data.get("leaderboard") or [])
    suite_id = "ops.clip_review_v1"
    suite_results: dict[str, dict[str, Any]] = {suite_id: {}}
    recipe_snapshots: list[dict[str, Any]] = []
    recipe_lookup = {recipe["id"]: recipe for recipe in list_recipes_public()}
    asset_lookup = {asset["asset_id"]: asset for asset in list_assets()}

    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        recipe_id = f"detector:{candidate_id}"
        if candidate.get("pipeline_override") == "soccermaster":
            recipe_id = "pipeline:soccermaster"
        row = next((entry for entry in leaderboard if str(entry.get("candidate_id") or "") == candidate_id), {})
        metrics = {
            "fps": metric_value(row.get("throughput"), label="FPS", precision=2),
            "track_stability": metric_value(row.get("track_stability"), label="Track stability", precision=2),
            "calibration": metric_value(row.get("calibration"), label="Calibration", precision=2),
            "coverage": metric_value(row.get("coverage"), label="Coverage", precision=2),
        }
        suite_results[suite_id][recipe_id] = {
            "suite_id": suite_id,
            "recipe_id": recipe_id,
            "status": row.get("status") or "completed",
            "error": row.get("error"),
            "metrics": metrics,
            "artifacts": {
                "run_id": row.get("run_id"),
            },
            "legacy_record": True,
        }
        if recipe_id in recipe_lookup:
            recipe_snapshots.append(recipe_lookup[recipe_id])
        elif recipe_id.startswith("detector:"):
            asset_id = f"detector.{candidate_id}"
            recipe_snapshots.append({
                "id": recipe_id,
                "label": candidate.get("label") or candidate_id,
                "kind": "legacy_detector_recipe",
                "asset_id": asset_id,
                "source_asset_ids": [asset_id],
                "pipeline": candidate.get("pipeline_override") or "classic",
                "available": True,
                "class_mapping": {},
                "capabilities": {"detection": True},
            })
            if asset_id not in asset_lookup:
                asset_lookup[asset_id] = {
                    "asset_id": asset_id,
                    "kind": "detector",
                    "provider": candidate.get("source", "legacy"),
                    "source": candidate.get("source", "legacy"),
                    "label": candidate.get("label") or candidate_id,
                    "version": "legacy",
                    "architecture": "legacy_detector",
                    "artifact_path": candidate.get("path"),
                    "bundle_mode": "separable",
                    "runtime_binding": "replace_component",
                    "available": True,
                    "capabilities": {"detection": True},
                }

    return {
        "benchmark_id": data.get("benchmark_id") or benchmark_dir.name,
        "schema_version": LEGACY_SCHEMA_VERSION,
        "legacy_record": True,
        "label": data.get("benchmark_id") or benchmark_dir.name,
        "status": data.get("status", "completed"),
        "created_at": data.get("created_at", ""),
        "primary_suite_id": suite_id,
        "suite_ids": [suite_id],
        "recipe_ids": [recipe["id"] for recipe in recipe_snapshots],
        "suite_results": suite_results,
        "assets": list(asset_lookup.values()),
        "recipes": recipe_snapshots,
        "progress": data.get("progress", 100.0),
        "logs": data.get("logs", []),
        "dvc_runtime": data.get("dvc_runtime") or probe_dvc_runtime(),
        "legacy_clip_status": {
            "ready": bool(data.get("clip_path")),
            "path": data.get("clip_path"),
            "dvc": data.get("clip_dvc"),
        },
    }


class BenchmarkOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_benchmarks: dict[str, dict[str, Any]] = {}
        self._restore_completed()

    def create_benchmark(
        self,
        *,
        suite_ids: list[str],
        recipe_ids: list[str],
        label: str = "",
    ) -> dict[str, Any]:
        suites = [get_suite_definition(suite_id) for suite_id in suite_ids]
        recipes = [recipe for recipe in list_recipes_public() if recipe["id"] in set(recipe_ids)]
        if not suites:
            raise RuntimeError("Select at least one benchmark suite.")
        if not recipes:
            raise RuntimeError("Select at least one benchmark recipe.")

        benchmark_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        benchmark_dir = BENCHMARKS_DIR / benchmark_id
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "benchmark_id": benchmark_id,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "legacy_record": False,
            "label": label.strip() or benchmark_id,
            "status": "queued",
            "created_at": utc_now_iso(),
            "primary_suite_id": str(suites[0]["id"]),
            "suite_ids": [suite["id"] for suite in suites],
            "recipe_ids": [recipe["id"] for recipe in recipes],
            "assets": list_assets(),
            "recipes": recipes,
            "suite_results": {str(suite["id"]): {} for suite in suites},
            "progress": 0.0,
            "logs": [],
            "error": None,
            "dvc_runtime": probe_dvc_runtime(),
            "legacy_clip_status": clip_status(),
        }
        environment = {
            "created_at": utc_now_iso(),
            "runtime_profile": dict(BENCHMARK_RUNTIME_PROFILE),
        }
        (benchmark_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        self._persist_state(benchmark_dir, state)
        with self._lock:
            self._active_benchmarks[benchmark_id] = state

        thread = threading.Thread(
            target=self._run_benchmark,
            args=(benchmark_id, benchmark_dir, suites, recipes),
            daemon=True,
        )
        thread.start()
        return state

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._active_benchmarks.get(benchmark_id)
            if state:
                return dict(state)
        benchmark_dir = BENCHMARKS_DIR / benchmark_id
        benchmark_json = benchmark_dir / "benchmark.json"
        legacy_json = benchmark_dir / "benchmark_summary.json"
        if benchmark_json.exists():
            try:
                return json.loads(benchmark_json.read_text(encoding="utf-8"))
            except Exception:
                return None
        if legacy_json.exists():
            try:
                payload = json.loads(legacy_json.read_text(encoding="utf-8"))
                return hydrate_legacy_benchmark(payload, benchmark_dir)
            except Exception:
                return None
        return None

    def list_benchmarks(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not BENCHMARKS_DIR.exists():
            return results
        for child in sorted(BENCHMARKS_DIR.iterdir(), reverse=True):
            if child.name.startswith("_") or not child.is_dir():
                continue
            benchmark_json = child / "benchmark.json"
            legacy_json = child / "benchmark_summary.json"
            data: dict[str, Any] | None = None
            try:
                if benchmark_json.exists():
                    data = json.loads(benchmark_json.read_text(encoding="utf-8"))
                elif legacy_json.exists():
                    data = hydrate_legacy_benchmark(json.loads(legacy_json.read_text(encoding="utf-8")), child)
            except Exception:
                data = None
            if data is None:
                continue
            results.append({
                "benchmark_id": data.get("benchmark_id", child.name),
                "label": data.get("label") or child.name,
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at", ""),
                "primary_suite_id": data.get("primary_suite_id"),
                "suite_ids": data.get("suite_ids", []),
                "recipe_count": len(data.get("recipe_ids") or []),
                "legacy_record": bool(data.get("legacy_record")),
            })
        return results

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.list_benchmarks()[:limit]

    def _run_benchmark(
        self,
        benchmark_id: str,
        benchmark_dir: Path,
        suites: list[dict[str, Any]],
        recipes: list[dict[str, Any]],
    ) -> None:
        total_tasks = max(len(suites) * len(recipes), 1)
        completed_tasks = 0
        self._update_state(benchmark_id, benchmark_dir, status="running")
        for suite in suites:
            suite_id = str(suite["id"])
            self._append_log(benchmark_id, benchmark_dir, f"Running suite {suite_id}")
            dataset_state = build_suite_dataset_state(suite)
            dataset_root = str(dataset_state.get("dataset_root") or "")
            if suite_id == "ops.clip_review_v1":
                clip = clip_status()
                dataset_root = str(clip.get("path") or "")
                dataset_state["ready"] = bool(clip.get("ready"))
            for recipe in recipes:
                recipe_id = str(recipe["id"])
                try:
                    result = self._run_suite_recipe(
                        benchmark_id=benchmark_id,
                        benchmark_dir=benchmark_dir,
                        suite=suite,
                        suite_dataset_state=dataset_state,
                        recipe=recipe,
                        dataset_root=dataset_root,
                    )
                except Exception as exc:  # pragma: no cover - safety net
                    result = {
                        "suite_id": suite_id,
                        "recipe_id": recipe_id,
                        "status": "failed",
                        "error": str(exc),
                        "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                        "artifacts": {},
                    }
                with self._lock:
                    state = self._active_benchmarks[benchmark_id]
                    state["suite_results"].setdefault(suite_id, {})[recipe_id] = result
                    self._persist_state(benchmark_dir, state)
                completed_tasks += 1
                self._update_state(
                    benchmark_id,
                    benchmark_dir,
                    progress=round((completed_tasks / total_tasks) * 100.0, 2),
                )
        self._update_state(benchmark_id, benchmark_dir, status="completed", progress=100.0)

    def _run_suite_recipe(
        self,
        *,
        benchmark_id: str,
        benchmark_dir: Path,
        suite: dict[str, Any],
        suite_dataset_state: dict[str, Any],
        recipe: dict[str, Any],
        dataset_root: str,
    ) -> dict[str, Any]:
        suite_id = str(suite["id"])
        recipe_id = str(recipe["id"])
        compatible_suite_ids = {str(item) for item in (recipe.get("compatible_suite_ids") or []) if str(item)}
        if compatible_suite_ids and suite_id not in compatible_suite_ids:
            return {
                "suite_id": suite_id,
                "recipe_id": recipe_id,
                "status": "not_supported",
                "error": "Recipe is outside this suite's declared compatibility set.",
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": {},
                "raw_result": {
                    "reason": "recipe_compatibility",
                    "compatible_suite_ids": sorted(compatible_suite_ids),
                },
            }
        required = [str(item) for item in (suite.get("required_capabilities") or [])]
        capabilities = dict(recipe.get("capabilities") or {})
        if not all(bool(capabilities.get(key)) for key in required):
            return {
                "suite_id": suite_id,
                "recipe_id": recipe_id,
                "status": "not_supported",
                "error": "Recipe does not satisfy suite capability requirements.",
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": {},
                "blockers": [],
                "runtime_context": {},
                "raw_result": {
                    "reason": "not_supported",
                    "required_capabilities": required,
                    "recipe_capabilities": capabilities,
                },
            }
        if not bool(recipe.get("available", False)):
            return {
                "suite_id": suite_id,
                "recipe_id": recipe_id,
                "status": "unavailable",
                "error": str(recipe.get("availability_error") or "Recipe assets are unavailable."),
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": {},
                "blockers": [],
                "runtime_context": {},
                "raw_result": {
                    "reason": "recipe_unavailable",
                    "availability_error": recipe.get("availability_error"),
                },
            }
        if not bool(suite_dataset_state.get("ready")):
            return {
                "suite_id": suite_id,
                "recipe_id": recipe_id,
                "status": _blocked_suite_status(suite_dataset_state),
                "error": str(suite_dataset_state.get("note") or "Benchmark dataset is unavailable."),
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": {},
                "blockers": list(suite_dataset_state.get("blockers") or []),
                "runtime_context": {},
                "raw_result": {
                    "reason": "suite_blocked",
                    "dataset_state": suite_dataset_state,
                },
            }

        artifacts_dir = benchmark_dir / "suite_results" / suite_id / recipe_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        provenance = build_benchmark_provenance(
            benchmark_id=benchmark_id,
            suite=suite,
            recipe=recipe,
            dataset_root=dataset_root,
            manifest_path=str(suite.get("manifest_path") or ""),
            benchmark_dir=benchmark_dir,
            artifacts_dir=artifacts_dir,
        )
        provenance_path = artifacts_dir / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        started_at = utc_now_iso()
        started_clock = perf_counter()
        prediction_export_artifacts: dict[str, Any] = {}
        prediction_export_raw: dict[str, Any] = {}
        try:
            prepared = prepare_prediction_exports(
                suite=suite,
                recipe=recipe,
                dataset_root=dataset_root,
                artifacts_dir=artifacts_dir,
                benchmark_id=benchmark_id,
            )
            prediction_export_artifacts = dict(prepared.get("artifacts") or {})
            prediction_export_raw = dict(prepared.get("raw_result") or {})
            evaluation = run_suite_evaluation(
                suite=suite,
                recipe=recipe,
                dataset_root=dataset_root,
                artifacts_dir=artifacts_dir,
                benchmark_id=benchmark_id,
            )
            status = "completed"
            error = None
            blockers: list[str] = []
        except BenchmarkPredictionExportUnavailable as exc:
            evaluation = {
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": dict(exc.artifacts or {}),
                "raw_result": dict(exc.raw_result or {}),
            }
            status = "blocked"
            error = str(exc)
            blockers = [str(exc)]
        except BenchmarkEvaluationUnavailable as exc:
            evaluation = {
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": {},
                "raw_result": {},
            }
            status = "blocked"
            error = str(exc)
            blockers = [str(exc)]
        except Exception as exc:
            evaluation = {
                "metrics": {metric: na_metric(label=metric) for metric in suite.get("metric_columns") or []},
                "artifacts": {},
                "raw_result": {},
            }
            status = "failed"
            error = str(exc)
            blockers = []
        finished_at = utc_now_iso()
        duration_seconds = round(perf_counter() - started_clock, 4)

        metrics = dict(evaluation.get("metrics") or {})
        for metric_name in suite.get("metric_columns") or []:
            metrics.setdefault(str(metric_name), na_metric(label=str(metric_name)))
        result = {
            "suite_id": suite_id,
            "recipe_id": recipe_id,
            "status": status,
            "error": error,
            "metrics": metrics,
            "flattened_metrics": _flatten_metrics(metrics),
            "primary_metric": suite.get("primary_metric"),
            "artifacts": {
                **prediction_export_artifacts,
                **dict(evaluation.get("artifacts") or {}),
                "provenance_json": str(provenance_path),
            },
            "blockers": blockers,
            "runtime_context": {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration_seconds,
            },
            "raw_result": {
                **prediction_export_raw,
                **dict(evaluation.get("raw_result") or {}),
            },
        }
        result_path = benchmark_dir / "suite_results" / suite_id / recipe_id / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _update_state(self, benchmark_id: str, benchmark_dir: Path, **kwargs: Any) -> None:
        with self._lock:
            state = self._active_benchmarks.get(benchmark_id)
            if state is None:
                return
            state.update(kwargs)
            self._persist_state(benchmark_dir, state)

    def _append_log(self, benchmark_id: str, benchmark_dir: Path, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            state = self._active_benchmarks.get(benchmark_id)
            if state is None:
                return
            state.setdefault("logs", []).append(line)
            self._persist_state(benchmark_dir, state)

    def _persist_state(self, benchmark_dir: Path, state: dict[str, Any]) -> None:
        benchmark_json = benchmark_dir / "benchmark.json"
        benchmark_json.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _restore_completed(self) -> None:
        if not BENCHMARKS_DIR.exists():
            return
        for child in BENCHMARKS_DIR.iterdir():
            if child.name.startswith("_") or not child.is_dir():
                continue
            benchmark_json = child / "benchmark.json"
            legacy_json = child / "benchmark_summary.json"
            try:
                if benchmark_json.exists():
                    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
                elif legacy_json.exists():
                    payload = hydrate_legacy_benchmark(json.loads(legacy_json.read_text(encoding="utf-8")), child)
                else:
                    continue
            except Exception:
                continue
            with self._lock:
                self._active_benchmarks[child.name] = payload


benchmark_orchestrator = BenchmarkOrchestrator()
