from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.benchmark_suites import list_suite_definitions
from app.training_registry import REGISTRY_PATH
from app.training_provenance import resolve_dvc_tracking
from app.wide_angle import resolve_detector_spec, resolve_model_path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
IMPORTS_DIR = BENCHMARKS_DIR / "_imports"
IMPORTS_MANIFEST_PATH = IMPORTS_DIR / "imports.json"
CATALOG_JSON_PATH = Path(__file__).resolve().with_name("benchmark_catalog.json")
SOCCERMASTER_MODELS_DIR = BASE_DIR / "models" / "soccermaster"
SOCCERNET_VENDOR_DIR = BASE_DIR / "third_party" / "soccernet"
SN_GAMESTATE_DIR = SOCCERNET_VENDOR_DIR / "sn-gamestate"
TRACKLAB_DIR = SOCCERNET_VENDOR_DIR / "tracklab"


def _resolve_repo_path(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    raw = str(raw_path)
    if raw.startswith("backend/"):
        return str((BASE_DIR.parent / raw).resolve())
    return raw


def _load_static_catalog() -> list[dict[str, Any]]:
    payload = json.loads(CATALOG_JSON_PATH.read_text(encoding="utf-8"))
    assets = payload.get("assets") or []
    result: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        normalized = dict(asset)
        normalized["artifact_path"] = _resolve_repo_path(str(asset.get("artifact_path") or ""))
        result.append(normalized)
    return result


def _capability_map(**overrides: bool) -> dict[str, bool]:
    base = {
        "detection": False,
        "tracking": False,
        "reid": False,
        "calibration": False,
        "team_id": False,
        "role_id": False,
        "jersey_ocr": False,
        "event_spotting": False,
    }
    base.update(overrides)
    return base


def _detector_asset_from_spec(
    *,
    asset_id: str,
    label: str,
    source: str,
    provider: str,
    artifact_path: str,
    version: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = resolve_detector_spec(str(artifact_path))
    class_mapping = {
        "player_class_ids": list(spec.get("player_class_ids") or []),
        "ball_class_ids": list(spec.get("ball_class_ids") or []),
        "referee_class_ids": list(spec.get("referee_class_ids") or []),
        "class_names_source": spec.get("class_names_source") or "",
        "class_names": {str(key): value for key, value in (spec.get("class_names") or {}).items()},
    }
    class_names = {str(value).strip().lower() for value in (spec.get("class_names") or {}).values()}
    return {
        "asset_id": asset_id,
        "kind": "detector",
        "provider": provider,
        "source": source,
        "label": label,
        "version": version,
        "architecture": "ultralytics_yolo",
        "artifact_path": str(Path(str(artifact_path)).resolve()),
        "bundle_mode": "separable",
        "runtime_binding": "replace_component",
        "available": Path(str(artifact_path)).exists(),
        "class_mapping": class_mapping,
        "capabilities": _capability_map(
            detection=True,
            role_id=bool(class_mapping["referee_class_ids"] or {"goalkeeper"} & class_names),
        ),
        "artifact_dvc": resolve_dvc_tracking(str(artifact_path)),
        **(metadata or {}),
    }


def _load_import_records() -> list[dict[str, Any]]:
    if not IMPORTS_MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(IMPORTS_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _dynamic_detector_assets() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if REGISTRY_PATH.exists():
        try:
            payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for entry in payload.get("detectors") or []:
            if not isinstance(entry, dict):
                continue
            detector_id = str(entry.get("id") or "").strip()
            if not detector_id or detector_id == "soccana":
                continue
            checkpoint_path = str(entry.get("path") or "").strip()
            if not checkpoint_path:
                continue
            try:
                result.append(
                    _detector_asset_from_spec(
                        asset_id=f"detector.{detector_id}",
                        label=str(entry.get("label") or detector_id),
                        source="registry",
                        provider="local",
                        artifact_path=checkpoint_path,
                        version=str(entry.get("training_run_id") or detector_id),
                        metadata={
                            "training_run_id": entry.get("training_run_id"),
                            "metrics": entry.get("metrics"),
                        },
                    )
                )
            except Exception as exc:
                result.append({
                    "asset_id": f"detector.{detector_id}",
                    "kind": "detector",
                    "provider": "local",
                    "source": "registry",
                    "label": str(entry.get("label") or detector_id),
                    "version": str(entry.get("training_run_id") or detector_id),
                    "architecture": "ultralytics_yolo",
                    "artifact_path": checkpoint_path,
                    "bundle_mode": "separable",
                    "runtime_binding": "replace_component",
                    "available": False,
                    "capabilities": _capability_map(detection=True),
                    "class_mapping": {},
                    "artifact_dvc": resolve_dvc_tracking(checkpoint_path),
                    "availability_error": str(exc),
                    "training_run_id": entry.get("training_run_id"),
                    "metrics": entry.get("metrics"),
                })

    for record in _load_import_records():
        if not isinstance(record, dict):
            continue
        import_id = str(record.get("id") or "").strip()
        checkpoint_path = str(record.get("path") or "").strip()
        if not import_id or not checkpoint_path:
            continue
        try:
            result.append(
                _detector_asset_from_spec(
                    asset_id=f"detector.{import_id}",
                    label=str(record.get("label") or import_id),
                    source="import",
                    provider="huggingface" if str(record.get("origin") or "").startswith("hf://") else "local",
                    artifact_path=checkpoint_path,
                    version=import_id,
                    metadata={
                        "import_origin": record.get("origin"),
                        "imported_at": record.get("imported_at"),
                    },
                )
            )
        except Exception as exc:
            result.append({
                "asset_id": f"detector.{import_id}",
                "kind": "detector",
                "provider": "local",
                "source": "import",
                "label": str(record.get("label") or import_id),
                "version": import_id,
                "architecture": "ultralytics_yolo",
                "artifact_path": checkpoint_path,
                "bundle_mode": "separable",
                "runtime_binding": "replace_component",
                "available": False,
                "capabilities": _capability_map(detection=True),
                "class_mapping": {},
                "artifact_dvc": resolve_dvc_tracking(checkpoint_path),
                "availability_error": str(exc),
                "import_origin": record.get("origin"),
                "imported_at": record.get("imported_at"),
            })
    return result


def list_assets() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset in _load_static_catalog():
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id or asset_id in seen:
            continue
        normalized = dict(asset)
        if asset_id == "detector.soccana":
            normalized = _detector_asset_from_spec(
                asset_id=asset_id,
                label=str(asset.get("label") or "soccana"),
                source=str(asset.get("source") or "pretrained"),
                provider=str(asset.get("provider") or "local"),
                artifact_path=resolve_model_path("soccana", "detector"),
                version=str(asset.get("version") or "pretrained"),
            )
        elif asset_id == "pipeline.soccermaster":
            backbone = SOCCERMASTER_MODELS_DIR / "backbone.pt"
            detection = SOCCERMASTER_MODELS_DIR / "SoccerNetGSR_Detection.pt"
            keypoints = SOCCERMASTER_MODELS_DIR / "KeypointsDetection.pt"
            normalized["available"] = backbone.exists() and detection.exists() and keypoints.exists()
            normalized["capabilities"] = _capability_map(
                detection=True,
                tracking=True,
                calibration=True,
                team_id=True,
            )
            normalized["artifact_dvc"] = resolve_dvc_tracking(str(SOCCERMASTER_MODELS_DIR))
        elif asset_id == "pipeline.tracklab_sn_gamestate":
            normalized["available"] = SN_GAMESTATE_DIR.exists() and TRACKLAB_DIR.exists()
            normalized["capabilities"] = _capability_map(
                detection=True,
                tracking=True,
                reid=True,
                calibration=True,
                team_id=True,
                role_id=True,
                jersey_ocr=True,
            )
            normalized["artifact_dvc"] = resolve_dvc_tracking(str(SN_GAMESTATE_DIR))
        elif asset_id.startswith("tracker."):
            normalized["available"] = True
            normalized["capabilities"] = _capability_map(
                tracking=True,
                reid=asset_id == "tracker.hybrid_reid",
            )
        elif asset_id.startswith("keypoint."):
            normalized["available"] = Path(str(normalized.get("artifact_path") or "")).exists()
            normalized["capabilities"] = _capability_map(calibration=True)
            normalized["artifact_dvc"] = resolve_dvc_tracking(str(normalized.get("artifact_path") or ""))
        result.append(normalized)
        seen.add(asset_id)

    for asset in _dynamic_detector_assets():
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id or asset_id in seen:
            continue
        result.append(asset)
        seen.add(asset_id)
    return result


def _asset_lookup(assets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(asset.get("asset_id") or ""): asset for asset in assets if asset.get("asset_id")}


def _supports_suite(recipe: dict[str, Any], suite: dict[str, Any]) -> bool:
    capabilities = dict(recipe.get("capabilities") or {})
    required = [str(item) for item in (suite.get("required_capabilities") or [])]
    return all(bool(capabilities.get(key)) for key in required)


def list_recipes() -> list[dict[str, Any]]:
    assets = list_assets()
    asset_by_id = _asset_lookup(assets)
    suites = list_suite_definitions()
    recipes: list[dict[str, Any]] = []

    detector_assets = [asset for asset in assets if asset.get("kind") == "detector"]
    for detector_asset in detector_assets:
        detector_asset_id = str(detector_asset.get("asset_id") or "")
        detector_slug = detector_asset_id.split(".", 1)[-1]
        class_mapping = dict(detector_asset.get("class_mapping") or {})
        detection_recipe = {
            "id": f"detector:{detector_slug}",
            "label": str(detector_asset.get("label") or detector_slug),
            "kind": "detector_recipe",
            "asset_id": detector_asset_id,
            "source_asset_ids": [detector_asset_id],
            "pipeline": "classic",
            "bundle_mode": "separable",
            "runtime_binding": "replace_component",
            "available": bool(detector_asset.get("available")),
            "artifact_path": detector_asset.get("artifact_path"),
            "class_mapping": class_mapping,
            "capabilities": _capability_map(
                detection=True,
                role_id=bool(detector_asset.get("capabilities", {}).get("role_id")),
            ),
        }
        detection_recipe["compatible_suite_ids"] = [
            str(suite.get("id"))
            for suite in suites
            if _supports_suite(detection_recipe, suite)
        ]
        recipes.append(detection_recipe)

        for tracker_asset_id, tracker_mode in (
            ("tracker.bytetrack", "bytetrack"),
            ("tracker.hybrid_reid", "hybrid_reid"),
        ):
            tracker_asset = asset_by_id.get(tracker_asset_id)
            recipe = {
                "id": f"tracker:{detector_slug}+{tracker_mode}+soccana_keypoint",
                "label": f"{detector_asset.get('label')} + {tracker_asset.get('label') if tracker_asset else tracker_mode}",
                "kind": "tracking_recipe",
                "asset_id": detector_asset_id,
                "source_asset_ids": [detector_asset_id, tracker_asset_id, "keypoint.soccana_keypoint"],
                "pipeline": "classic",
                "detector_asset_id": detector_asset_id,
                "tracker_asset_id": tracker_asset_id,
                "requested_tracker_mode": tracker_mode,
                "keypoint_model": "soccana_keypoint",
                "bundle_mode": "separable",
                "runtime_binding": "replace_component",
                "available": bool(detector_asset.get("available")) and bool((tracker_asset or {}).get("available", True)),
                "artifact_path": detector_asset.get("artifact_path"),
                "class_mapping": class_mapping,
                "capabilities": _capability_map(
                    detection=True,
                    tracking=True,
                    reid=tracker_mode == "hybrid_reid",
                    calibration=True,
                    team_id=True,
                    role_id=bool(detector_asset.get("capabilities", {}).get("role_id")),
                ),
            }
            recipe["compatible_suite_ids"] = [
                str(suite.get("id"))
                for suite in suites
                if _supports_suite(recipe, suite)
            ]
            recipes.append(recipe)

    pipeline_asset = asset_by_id.get("pipeline.soccermaster")
    if pipeline_asset is not None:
        recipe = {
            "id": "pipeline:soccermaster",
            "label": str(pipeline_asset.get("label") or "SoccerMaster"),
            "kind": "pipeline_recipe",
            "asset_id": "pipeline.soccermaster",
            "source_asset_ids": ["pipeline.soccermaster"],
            "pipeline": "soccermaster",
            "bundle_mode": "bundled",
            "runtime_binding": "full_pipeline",
            "available": bool(pipeline_asset.get("available")),
            "artifact_path": pipeline_asset.get("artifact_path"),
            "class_mapping": {},
            "capabilities": _capability_map(
                detection=True,
                tracking=True,
                calibration=True,
                team_id=True,
            ),
        }
        recipe["compatible_suite_ids"] = [
            str(suite.get("id"))
            for suite in suites
            if _supports_suite(recipe, suite) and str(suite.get("family") or "") in {"operational", "tracking", "game_state", "calibration"}
        ]
        recipes.append(recipe)

    tracklab_asset = asset_by_id.get("pipeline.tracklab_sn_gamestate")
    if tracklab_asset is not None:
        recipe = {
            "id": "pipeline:sn-gamestate-tracklab",
            "label": str(tracklab_asset.get("label") or "sn-gamestate / TrackLab"),
            "kind": "pipeline_recipe",
            "asset_id": "pipeline.tracklab_sn_gamestate",
            "source_asset_ids": ["pipeline.tracklab_sn_gamestate"],
            "pipeline": "tracklab_gamestate",
            "bundle_mode": "bundled",
            "runtime_binding": "full_pipeline",
            "available": bool(tracklab_asset.get("available")),
            "artifact_path": tracklab_asset.get("artifact_path"),
            "class_mapping": {},
            "capabilities": _capability_map(
                detection=True,
                tracking=True,
                reid=True,
                calibration=True,
                team_id=True,
                role_id=True,
                jersey_ocr=True,
            ),
        }
        recipe["compatible_suite_ids"] = [
            str(suite.get("id"))
            for suite in suites
            if _supports_suite(recipe, suite) and str(suite.get("family") or "") in {"tracking", "game_state"}
        ]
        recipes.append(recipe)

    recipes.sort(key=lambda item: (str(item.get("kind") or ""), str(item.get("label") or "")))
    return recipes
