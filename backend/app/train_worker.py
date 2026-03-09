from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION

from app.training import TRAINING_BACKEND

BASE_DIR = Path(__file__).resolve().parent.parent


def choose_training_device(config_value: str) -> str:
    requested = str(config_value or "").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_weights(base_weights: str) -> str:
    from app.wide_angle import resolve_model_path

    return resolve_model_path(base_weights or "soccana", "detector")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m app.train_worker /abs/path/to/run_dir")

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress_path = run_dir / "progress.json"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_path = resolve_weights(str(config.get("base_weights") or "soccana"))
    resolved_device = choose_training_device(str(config.get("device") or "auto"))
    data_arg = str(config.get("generated_dataset_yaml") or "")
    if not data_arg:
        raise RuntimeError("Training config is missing generated_dataset_yaml.")

    model = YOLO(weights_path)

    def write_progress(epoch: int, total_epochs: int, metrics: dict[str, float], done: bool) -> None:
        payload = {
            "epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "metrics": metrics,
            "done": bool(done),
            "resolved_device": resolved_device,
            "backend": TRAINING_BACKEND,
            "backend_version": ULTRALYTICS_VERSION,
            "generated_dataset_yaml": data_arg,
            "best_checkpoint": str((weights_dir / "best.pt").resolve()),
        }
        progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def on_epoch_end(trainer: Any) -> None:
        trainer_metrics = getattr(trainer, "metrics", None) or {}
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", config.get("epochs", 50)))
        metrics = {
            "mAP50": safe_float(trainer_metrics.get("metrics/mAP50(B)")),
            "mAP50_95": safe_float(trainer_metrics.get("metrics/mAP50-95(B)")),
        }
        write_progress(epoch, total_epochs, metrics, done=False)

    write_progress(0, int(config.get("epochs", 50)), {}, done=False)
    model.add_callback("on_train_epoch_end", on_epoch_end)

    train_kwargs: dict[str, Any] = {
        "data": data_arg,
        "epochs": int(config.get("epochs", 50)),
        "imgsz": int(config.get("imgsz", 640)),
        "batch": int(config.get("batch", 16)),
        "device": resolved_device,
        "workers": int(config.get("workers", 4)),
        "patience": int(config.get("patience", 20)),
        "project": str(run_dir / "yolo_output"),
        "name": "train",
        "exist_ok": True,
        "cache": bool(config.get("cache", False)),
    }
    freeze = config.get("freeze")
    if freeze is not None:
        train_kwargs["freeze"] = int(freeze)

    model.train(**train_kwargs)

    best_src = run_dir / "yolo_output" / "train" / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, weights_dir / "best.pt")

    results_dict = getattr(getattr(model, "metrics", None), "results_dict", {}) or {}
    final_metrics = {
        "mAP50": safe_float(results_dict.get("metrics/mAP50(B)")),
        "mAP50_95": safe_float(results_dict.get("metrics/mAP50-95(B)")),
        "precision": safe_float(results_dict.get("metrics/precision(B)")),
        "recall": safe_float(results_dict.get("metrics/recall(B)")),
    }
    write_progress(
        int(config.get("epochs", 50)),
        int(config.get("epochs", 50)),
        final_metrics,
        done=True,
    )


if __name__ == "__main__":
    main()
