from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.wide_angle import resolve_model_path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
DEFAULT_CLASS_IDS = {
    "player_class_id": 0,
    "ball_class_id": 1,
    "referee_class_id": 2,
}


class TrainingRegistry:
    def __init__(self, registry_path: Path = REGISTRY_PATH) -> None:
        self._registry_path = registry_path
        self._lock = threading.Lock()

    def init_if_absent(self) -> dict[str, Any]:
        with self._lock:
            payload = self._load_locked()
            self._save_locked(payload)
            return payload

    def get_registry(self) -> dict[str, Any]:
        with self._lock:
            payload = self._load_locked()
            self._save_locked(payload)
            return payload

    def snapshot(self) -> dict[str, Any]:
        return self.get_registry()

    def get_active_detector(self) -> str:
        with self._lock:
            payload = self._load_locked()
            return str(payload.get("active_detector") or "soccana")

    def get_active_detector_id(self) -> str:
        return self.get_active_detector()

    def get_active_path(self) -> str:
        with self._lock:
            payload = self._load_locked()
            active_id = str(payload.get("active_detector") or "soccana")
            detectors = payload.get("detectors") or []
            match = next((item for item in detectors if str(item.get("id")) == active_id), None)
            if not match:
                match = next((item for item in detectors if str(item.get("id")) == "soccana"), None)
            if not match:
                raise RuntimeError("No active detector is configured")
            return str(match.get("path") or resolve_model_path("soccana", "detector"))

    def get_active_entry(self) -> dict[str, Any]:
        with self._lock:
            payload = self._load_locked()
            active_id = str(payload.get("active_detector") or "soccana")
            detectors = payload.get("detectors") or []
            match = next((item for item in detectors if str(item.get("id")) == active_id), None)
            if match is None:
                match = next((item for item in detectors if str(item.get("id")) == "soccana"), None)
            return dict(match or {})

    def register_detector(
        self,
        *,
        run_id: str,
        checkpoint_path: str,
        run_name: str,
        base_weights: str,
        metrics: dict[str, Any] | None,
        created_at: str | None = None,
        activate: bool = False,
    ) -> dict[str, Any]:
        detector_id = f"custom_{run_id}"
        with self._lock:
            payload = self._load_locked()
            detectors = list(payload.get("detectors") or [])
            entry = {
                "id": detector_id,
                "label": run_name or detector_id,
                "path": str(Path(checkpoint_path).expanduser().resolve()),
                "is_pretrained": False,
                "base_weights": base_weights or "soccana",
                "created_at": created_at or datetime.utcnow().isoformat() + "Z",
                "metrics": metrics or None,
                "training_run_id": run_id,
                "class_ids": dict(DEFAULT_CLASS_IDS),
            }
            index = next((idx for idx, item in enumerate(detectors) if str(item.get("id")) == detector_id), None)
            if index is None:
                detectors.append(entry)
            else:
                existing = dict(detectors[index])
                existing.update({key: value for key, value in entry.items() if value is not None})
                if not existing.get("created_at"):
                    existing["created_at"] = datetime.utcnow().isoformat() + "Z"
                detectors[index] = existing
                entry = existing

            payload["detectors"] = detectors
            if activate:
                payload["active_detector"] = detector_id
            self._save_locked(payload)
            return entry

    def activate_detector(
        self,
        *,
        run_id: str,
        checkpoint_path: str,
        run_name: str,
        base_weights: str,
        metrics: dict[str, Any] | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        return self.register_detector(
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            run_name=run_name,
            base_weights=base_weights,
            metrics=metrics,
            created_at=created_at,
            activate=True,
        )

    def _load_locked(self) -> dict[str, Any]:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        if self._registry_path.exists():
            try:
                payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        return self._normalized_payload(payload if isinstance(payload, dict) else {})

    def _save_locked(self, payload: dict[str, Any]) -> None:
        normalized = self._normalized_payload(payload)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    def _normalized_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        pretrained_entry = {
            "id": "soccana",
            "label": "soccana (pretrained)",
            "path": resolve_model_path("soccana", "detector"),
            "is_pretrained": True,
            "created_at": payload.get("created_at") or datetime.utcnow().isoformat() + "Z",
            "metrics": None,
            "training_run_id": None,
            "class_ids": dict(DEFAULT_CLASS_IDS),
        }

        detectors: list[dict[str, Any]] = []
        raw_detectors = payload.get("detectors")
        if isinstance(raw_detectors, list):
            for item in raw_detectors:
                if not isinstance(item, dict):
                    continue
                detector = dict(item)
                detector_id = str(detector.get("id") or "")
                if not detector_id:
                    continue
                if detector_id == "soccana":
                    detector.update(pretrained_entry)
                else:
                    detector.setdefault("is_pretrained", False)
                    detector.setdefault("training_run_id", detector.get("training_run_id"))
                    detector.setdefault("base_weights", detector.get("base_weights") or "soccana")
                    detector.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
                    detector["class_ids"] = dict(detector.get("class_ids") or DEFAULT_CLASS_IDS)
                    path_value = detector.get("path")
                    if path_value:
                        detector["path"] = str(Path(str(path_value)).expanduser().resolve())
                detectors.append(detector)

        if not any(str(item.get("id")) == "soccana" for item in detectors):
            detectors.insert(0, pretrained_entry)

        active_detector = str(payload.get("active_detector") or "soccana")
        if not any(str(item.get("id")) == active_detector for item in detectors):
            active_detector = "soccana"

        detectors.sort(
            key=lambda item: (
                0 if str(item.get("id")) == active_detector else 1,
                0 if item.get("is_pretrained") else 1,
                str(item.get("created_at") or ""),
            ),
            reverse=False,
        )
        return {
            "active_detector": active_detector,
            "detectors": detectors,
        }
