from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION

from app.training import TRAINING_BACKEND

BASE_DIR = Path(__file__).resolve().parent.parent
MAX_CURVE_POINTS = 240
TARGET_SAMPLES_PER_EPOCH = 24


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


def trim_curve(points: list[dict[str, Any]], max_points: int = MAX_CURVE_POINTS) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    return points[-max_points:]


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
    training_curves: dict[str, list[dict[str, Any]]] = {"loss": [], "optimizer": []}
    curve_state = {
        "epoch_step": 0,
        "steps_per_epoch": 1,
        "sample_every": 1,
        "last_grad_norm": 0.0,
    }

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
            "training_curves": training_curves,
            "best_checkpoint": str((weights_dir / "best.pt").resolve()),
        }
        progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_curve_point(trainer: Any, *, force: bool = False) -> None:
        epoch_number = int(getattr(trainer, "epoch", 0)) + 1
        step_number = max(int(curve_state["epoch_step"]), 1)
        steps_per_epoch = max(int(curve_state["steps_per_epoch"]), 1)
        sample_every = max(int(curve_state["sample_every"]), 1)
        if not force and step_number % sample_every != 0 and step_number != steps_per_epoch:
            return

        epoch_progress = round((epoch_number - 1) + (step_number / steps_per_epoch), 4)
        loss_items = {}
        if hasattr(trainer, "label_loss_items") and getattr(trainer, "tloss", None) is not None:
            try:
                loss_items = trainer.label_loss_items(trainer.tloss)
            except Exception:
                loss_items = {}
        if not loss_items and getattr(trainer, "tloss", None) is not None:
            try:
                raw_losses = [safe_float(value) for value in trainer.tloss]
            except Exception:
                raw_losses = []
            if raw_losses:
                loss_items = {
                    "train/box_loss": raw_losses[0] if len(raw_losses) > 0 else 0.0,
                    "train/cls_loss": raw_losses[1] if len(raw_losses) > 1 else 0.0,
                    "train/dfl_loss": raw_losses[2] if len(raw_losses) > 2 else 0.0,
                }

        training_curves["loss"] = trim_curve([
            *training_curves["loss"],
            {
                "epoch": epoch_number,
                "step": step_number,
                "epoch_progress": epoch_progress,
                "box_loss": safe_float(loss_items.get("train/box_loss")),
                "cls_loss": safe_float(loss_items.get("train/cls_loss")),
                "dfl_loss": safe_float(loss_items.get("train/dfl_loss")),
            },
        ])

        optimizer_lr = 0.0
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer and getattr(optimizer, "param_groups", None):
            optimizer_lr = safe_float(optimizer.param_groups[0].get("lr"))

        training_curves["optimizer"] = trim_curve([
            *training_curves["optimizer"],
            {
                "epoch": epoch_number,
                "step": step_number,
                "epoch_progress": epoch_progress,
                "grad_norm": safe_float(curve_state["last_grad_norm"]),
                "lr": optimizer_lr,
            },
        ])

    def on_pretrain_routine_end(trainer: Any) -> None:
        import torch

        def instrumented_optimizer_step() -> None:
            trainer.scaler.unscale_(trainer.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
            curve_state["last_grad_norm"] = safe_float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad()
            if trainer.ema:
                trainer.ema.update(trainer.model)

        trainer.optimizer_step = instrumented_optimizer_step

    def on_epoch_start(trainer: Any) -> None:
        curve_state["epoch_step"] = 0
        try:
            steps_per_epoch = len(trainer.train_loader)
        except Exception:
            steps_per_epoch = 1
        curve_state["steps_per_epoch"] = max(int(steps_per_epoch), 1)
        curve_state["sample_every"] = max(1, int(curve_state["steps_per_epoch"] / TARGET_SAMPLES_PER_EPOCH))

    def on_train_batch_end(trainer: Any) -> None:
        curve_state["epoch_step"] = int(curve_state["epoch_step"]) + 1
        record_curve_point(trainer)

    def on_epoch_end(trainer: Any) -> None:
        trainer_metrics = getattr(trainer, "metrics", None) or {}
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", config.get("epochs", 50)))
        metrics = {
            "mAP50": safe_float(trainer_metrics.get("metrics/mAP50(B)")),
            "mAP50_95": safe_float(trainer_metrics.get("metrics/mAP50-95(B)")),
        }
        record_curve_point(trainer, force=True)
        write_progress(epoch, total_epochs, metrics, done=False)

    write_progress(0, int(config.get("epochs", 50)), {}, done=False)
    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
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
