from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.training import SUMMARY_FILENAME, collect_training_artifacts
from app.training_provenance import (
    PROVENANCE_FILENAME,
    build_training_provenance,
    read_training_provenance,
    write_training_provenance,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_RUNS_DIR = BASE_DIR / "training_runs"
TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingJobState:
    job_id: str
    run_id: str
    run_dir: str
    status: str = "queued"
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    logs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    started_at: str | None = None
    finished_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    dataset_scan: dict[str, Any] | None = None
    generated_dataset_yaml: str | None = None
    generated_split_lists: dict[str, str] = field(default_factory=dict)
    resolved_device: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    validation_strategy: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    training_curves: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    best_checkpoint: str | None = None
    summary_path: str | None = None
    training_provenance_path: str | None = None
    training_provenance: dict[str, Any] | None = None
    error: str | None = None
    pid: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "progress": round(self.progress, 2),
            "current_epoch": int(self.current_epoch or 0),
            "total_epochs": int(self.total_epochs or 0),
            "logs": self.logs[-250:],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "dataset_scan": self.dataset_scan,
            "generated_dataset_yaml": self.generated_dataset_yaml,
            "generated_split_lists": self.generated_split_lists,
            "resolved_device": self.resolved_device,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "validation_strategy": self.validation_strategy,
            "metrics": self.metrics or {},
            "training_curves": self.training_curves or {},
            "artifacts": self.artifacts or {},
            "best_checkpoint": self.best_checkpoint,
            "summary_path": self.summary_path,
            "training_provenance_path": self.training_provenance_path,
            "training_provenance": self.training_provenance or None,
            "error": self.error,
        }

    def persistence_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["pid"] = self.pid
        return payload


class TrainingManager:
    def __init__(self, runs_dir: Path = TRAINING_RUNS_DIR) -> None:
        self._runs_dir = runs_dir
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, TrainingJobState] = {}
        self._runs: dict[str, TrainingJobState] = {}
        self._restartable_job_ids: list[str] = []
        self._log_offsets: dict[str, int] = {}
        self._seen_epochs: dict[str, int] = {}
        self._lock = threading.Lock()
        self._restore()

    def create(self, config: dict[str, Any]) -> TrainingJobState:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        normalized_config = dict(config)
        normalized_config["run_name"] = str(normalized_config.get("run_name") or run_id)

        job = TrainingJobState(
            job_id=uuid.uuid4().hex[:12],
            run_id=run_id,
            run_dir=str(run_dir),
            config=normalized_config,
            total_epochs=int(normalized_config.get("epochs") or 0),
            backend=str(normalized_config.get("backend") or "") or None,
            backend_version=str(normalized_config.get("backend_version") or "") or None,
            summary_path=str((run_dir / SUMMARY_FILENAME).resolve()),
            training_provenance_path=str((run_dir / PROVENANCE_FILENAME).resolve()),
        )
        self._write_config(run_dir, normalized_config)
        with self._lock:
            self._jobs[job.job_id] = job
            self._runs[job.run_id] = job
            self._persist_locked(job)
        self.append_log(job.job_id, f"Queued detector training run {normalized_config.get('run_name') or run_id}")
        return job

    def get(self, job_id: str) -> TrainingJobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_by_run_id(self, run_id: str) -> TrainingJobState | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [job.as_dict() for job in jobs]

    def list_states(self) -> list[TrainingJobState]:
        with self._lock:
            return list(sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True))

    def list_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for job in self.list_states()[: max(int(limit), 1)]:
            runs.append(
                {
                    "run_id": job.run_id,
                    "run_name": str(job.config.get("run_name") or job.run_id),
                    "status": job.status,
                    "base_weights": str(job.config.get("base_weights") or "soccana"),
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "resolved_device": job.resolved_device,
                    "metrics": job.metrics or {},
                    "best_checkpoint": job.best_checkpoint,
                    "summary_path": job.summary_path,
                    "training_provenance_path": job.training_provenance_path,
                }
            )
        return runs

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)
            if "config" in kwargs and isinstance(job.config, dict):
                self._write_config(Path(job.run_dir), job.config)
            self._persist_locked(job)

    def refresh_training_provenance(self, job_id: str, activation: dict[str, Any] | None = None) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            run_dir = Path(job.run_dir)
            payload = build_training_provenance(
                run_id=job.run_id,
                run_dir=run_dir,
                status=job.status,
                config=job.config,
                dataset_path=str(job.config.get("dataset_path") or ""),
                dataset_scan_path=str((job.artifacts or {}).get("dataset_scan") or ""),
                generated_dataset_yaml=job.generated_dataset_yaml or str((job.artifacts or {}).get("generated_dataset_yaml") or ""),
                generated_split_lists=job.generated_split_lists,
                summary_path=job.summary_path,
                best_checkpoint=job.best_checkpoint,
                activation=activation,
            )
            provenance_path = write_training_provenance(run_dir / PROVENANCE_FILENAME, payload)
            job.training_provenance_path = provenance_path
            job.training_provenance = payload
            job.artifacts = self._collect_artifacts(run_dir)
            self._persist_locked(job)
            return {"path": provenance_path, "payload": payload}

    def append_log(self, job_id: str, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(f"[{stamp}] {message}")
            if len(job.logs) > 1000:
                job.logs = job.logs[-1000:]
            self._persist_locked(job)

    def launch(self, job_id: str, run_dir: Path, config: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == "stopped":
                return
        self._write_config(run_dir, config)
        progress_path = run_dir / "progress.json"
        progress_path.unlink(missing_ok=True)
        (run_dir / "weights" / "best.pt").unlink(missing_ok=True)
        log_path = run_dir / "train.log"
        current_offset = log_path.stat().st_size if log_path.exists() else 0
        logfile = log_path.open("ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.train_worker", str(run_dir)],
            cwd=str(BASE_DIR),
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.progress = max(float(job.progress or 0.0), 1.0)
            job.error = None
            job.pid = proc.pid
            job.started_at = datetime.utcnow().isoformat() + "Z"
            job.finished_at = None
            job.config = dict(config)
            job.total_epochs = int(config.get("epochs") or job.total_epochs or 0)
            job.generated_dataset_yaml = str(config.get("generated_dataset_yaml") or "") or None
            job.generated_split_lists = dict(config.get("generated_split_lists") or {})
            job.validation_strategy = str(config.get("validation_strategy") or "") or None
            job.dataset_scan = dict(config.get("dataset_scan") or {}) or None
            job.backend = str(config.get("backend") or job.backend or "") or None
            job.backend_version = str(config.get("backend_version") or job.backend_version or "") or None
            job.artifacts = self._collect_artifacts(run_dir)
            self._persist_locked(job)
            self._log_offsets[job_id] = current_offset
            self._seen_epochs[job_id] = int(job.current_epoch or 0)
        self.refresh_training_provenance(job_id)
        self.append_log(job_id, f"Training worker started (pid {proc.pid})")
        threading.Thread(
            target=self._poll_process,
            args=(job_id, run_dir, proc, logfile),
            daemon=True,
        ).start()

    def request_stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status == "queued" and job.pid is None:
                job.status = "stopped"
                job.error = None
                job.finished_at = datetime.utcnow().isoformat() + "Z"
                self._persist_locked(job)
                queued_stop = True
                pid = None
            else:
                if job.status not in {"queued", "running", "stopping"}:
                    return True
                job.status = "stopping"
                self._persist_locked(job)
                queued_stop = False
                pid = job.pid

        if queued_stop:
            self.refresh_training_provenance(job_id)
            self.append_log(job_id, "Stopped queued job before worker launch")
            return True

        self.append_log(job_id, "Stop requested")
        if pid:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        return True

    def consume_restartable_jobs(self) -> list[tuple[str, Path, dict[str, Any]]]:
        with self._lock:
            recovered: list[tuple[str, Path, dict[str, Any]]] = []
            for job_id in self._restartable_job_ids:
                job = self._jobs.get(job_id)
                if job is None or not job.config:
                    continue
                recovered.append((job.job_id, Path(job.run_dir), dict(job.config)))
            self._restartable_job_ids = []
            return recovered

    def _restore(self) -> None:
        if not self._runs_dir.exists():
            return

        restored_at = datetime.utcnow().strftime("%H:%M:%S")
        for run_dir in sorted(self._runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            state_path = run_dir / "job_state.json"
            if not state_path.exists():
                continue
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            job = TrainingJobState(
                job_id=str(payload.get("job_id") or ""),
                run_id=str(payload.get("run_id") or run_dir.name),
                run_dir=str(payload.get("run_dir") or run_dir),
                status=str(payload.get("status") or "queued"),
                progress=float(payload.get("progress") or 0.0),
                current_epoch=int(payload.get("current_epoch") or 0),
                total_epochs=int(payload.get("total_epochs") or 0),
                logs=list(payload.get("logs") or []),
                created_at=str(payload.get("created_at") or datetime.utcnow().isoformat() + "Z"),
                started_at=str(payload.get("started_at")) if payload.get("started_at") else None,
                finished_at=str(payload.get("finished_at")) if payload.get("finished_at") else None,
                config=dict(payload.get("config") or {}),
                dataset_scan=dict(payload.get("dataset_scan") or {}) or None,
                generated_dataset_yaml=str(payload.get("generated_dataset_yaml")) if payload.get("generated_dataset_yaml") else None,
                generated_split_lists=dict(payload.get("generated_split_lists") or {}),
                resolved_device=str(payload.get("resolved_device")) if payload.get("resolved_device") else None,
                backend=str(payload.get("backend")) if payload.get("backend") else None,
                backend_version=str(payload.get("backend_version")) if payload.get("backend_version") else None,
                validation_strategy=str(payload.get("validation_strategy")) if payload.get("validation_strategy") else None,
                metrics=dict(payload.get("metrics") or {}),
                training_curves=dict(payload.get("training_curves") or {}),
                artifacts=dict(payload.get("artifacts") or {}),
                best_checkpoint=str(payload.get("best_checkpoint")) if payload.get("best_checkpoint") else None,
                summary_path=str(payload.get("summary_path")) if payload.get("summary_path") else str((run_dir / SUMMARY_FILENAME).resolve()),
                training_provenance_path=(
                    str(payload.get("training_provenance_path"))
                    if payload.get("training_provenance_path")
                    else (str((run_dir / PROVENANCE_FILENAME).resolve()) if (run_dir / PROVENANCE_FILENAME).exists() else None)
                ),
                training_provenance=dict(payload.get("training_provenance") or {}) or None,
                error=str(payload.get("error")) if payload.get("error") else None,
                pid=int(payload.get("pid")) if payload.get("pid") else None,
            )
            if not job.job_id:
                continue

            if job.training_provenance is None and job.training_provenance_path:
                try:
                    job.training_provenance = read_training_provenance(job.training_provenance_path)
                except (ValueError, json.JSONDecodeError) as exc:
                    job.training_provenance = {
                        "load_error": f"Could not parse training provenance: {exc}",
                        "path": job.training_provenance_path,
                    }

            if job.status in {"queued", "running", "stopping"} and job.config:
                job.status = "queued"
                job.pid = None
                job.progress = 0.0
                job.error = None
                job.started_at = None
                job.finished_at = None
                job.logs.append(f"[{restored_at}] Backend restarted; re-queued detector training from epoch 0.")
                self._restartable_job_ids.append(job.job_id)

            self._jobs[job.job_id] = job
            self._runs[job.run_id] = job
            with self._lock:
                self._persist_locked(job)

    def _poll_process(self, job_id: str, run_dir: Path, proc: subprocess.Popen[bytes], logfile: Any) -> None:
        progress_path = run_dir / "progress.json"
        log_path = run_dir / "train.log"
        sigkill_after: float | None = None

        while True:
            time.sleep(3)
            self._ingest_progress(job_id, progress_path)
            self._ingest_log_lines(job_id, log_path)

            current = self.get(job_id)
            if current is None:
                break

            if current.status == "stopping":
                if sigkill_after is None:
                    sigkill_after = time.time() + 10.0
                elif current.pid and time.time() > sigkill_after:
                    try:
                        os.killpg(os.getpgid(current.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

            return_code = proc.poll()
            if return_code is None:
                continue

            self._ingest_progress(job_id, progress_path)
            self._ingest_log_lines(job_id, log_path)
            logfile.close()
            latest = self.get(job_id)
            if latest is None:
                break

            finished_at = datetime.utcnow().isoformat() + "Z"
            if latest.status in {"stopped", "stopping"}:
                self.update(job_id, status="stopped", pid=None, error=None, finished_at=finished_at)
                self.refresh_training_provenance(job_id)
                self.append_log(job_id, "Training stopped")
            elif return_code == 0:
                best_checkpoint = run_dir / "weights" / "best.pt"
                metrics = self._extract_final_metrics(run_dir) or latest.metrics or {}
                artifacts = self._collect_artifacts(run_dir)
                if best_checkpoint.exists():
                    self.update(
                        job_id,
                        status="completed",
                        progress=100.0,
                        current_epoch=max(int(latest.current_epoch or 0), int(latest.total_epochs or 0)),
                        metrics=metrics,
                        artifacts=artifacts,
                        best_checkpoint=str(best_checkpoint.resolve()),
                        pid=None,
                        error=None,
                        finished_at=finished_at,
                    )
                    self.refresh_training_provenance(job_id)
                    self.append_log(job_id, "Training completed")
                else:
                    self.update(
                        job_id,
                        status="failed",
                        pid=None,
                        artifacts=artifacts,
                        error="Training exited successfully but no best checkpoint was produced.",
                        finished_at=finished_at,
                    )
                    self.refresh_training_provenance(job_id)
            else:
                tail = " ".join(self._tail_log(log_path, limit=3))
                self.update(
                    job_id,
                    status="failed",
                    pid=None,
                    artifacts=self._collect_artifacts(run_dir),
                    error=f"Training process exited with code {return_code}. {tail}".strip(),
                    finished_at=finished_at,
                )
                self.refresh_training_provenance(job_id)
            with self._lock:
                self._log_offsets.pop(job_id, None)
                self._seen_epochs.pop(job_id, None)
            break

    def _job_state_path(self, run_dir: Path | str) -> Path:
        return Path(run_dir) / "job_state.json"

    def _summary_state_path(self, run_dir: Path | str) -> Path:
        return Path(run_dir) / SUMMARY_FILENAME

    def _write_config(self, run_dir: Path, config: dict[str, Any]) -> None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _collect_artifacts(self, run_dir: Path) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for key, value in collect_training_artifacts(run_dir).items():
            if value is None:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            filtered[key] = value
        return filtered

    def _summary_payload(self, job: TrainingJobState) -> dict[str, Any]:
        payload = job.as_dict()
        payload["logs"] = job.logs[-500:]
        payload["summary_path"] = job.summary_path or str(self._summary_state_path(job.run_dir).resolve())
        return payload

    def _persist_locked(self, job: TrainingJobState) -> None:
        run_dir = Path(job.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not job.summary_path:
            job.summary_path = str(self._summary_state_path(run_dir).resolve())
        if not job.training_provenance_path:
            job.training_provenance_path = str((run_dir / PROVENANCE_FILENAME).resolve())
        self._job_state_path(run_dir).write_text(json.dumps(job.persistence_dict(), indent=2), encoding="utf-8")
        self._summary_state_path(run_dir).write_text(json.dumps(self._summary_payload(job), indent=2), encoding="utf-8")

    def _ingest_progress(self, job_id: str, progress_path: Path) -> None:
        if not progress_path.exists():
            return
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        epoch = int(payload.get("epoch") or 0)
        total_epochs = max(int(payload.get("total_epochs") or 0), 1)
        metrics = dict(payload.get("metrics") or {})
        training_curves = dict(payload.get("training_curves") or {})
        done = bool(payload.get("done", False))
        progress = 100.0 if done else min((epoch / total_epochs) * 100.0, 99.0)
        resolved_device = str(payload.get("resolved_device") or "") or None
        backend = str(payload.get("backend") or "") or None
        backend_version = str(payload.get("backend_version") or "") or None

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.current_epoch = epoch
            job.total_epochs = total_epochs
            job.progress = max(float(job.progress or 0.0), float(progress))
            if metrics:
                job.metrics = metrics
            if training_curves:
                job.training_curves = training_curves
            if resolved_device:
                job.resolved_device = resolved_device
            if backend:
                job.backend = backend
            if backend_version:
                job.backend_version = backend_version
            job.artifacts = self._collect_artifacts(Path(job.run_dir))
            previous_epoch = self._seen_epochs.get(job_id, 0)
            self._seen_epochs[job_id] = max(previous_epoch, epoch)
            self._persist_locked(job)

        if epoch > previous_epoch:
            if metrics and "mAP50" in metrics:
                try:
                    self.append_log(job_id, f"Epoch {epoch}/{total_epochs} - mAP50 {float(metrics['mAP50']):.4f}")
                except Exception:
                    self.append_log(job_id, f"Epoch {epoch}/{total_epochs}")
            else:
                self.append_log(job_id, f"Epoch {epoch}/{total_epochs}")

    def _ingest_log_lines(self, job_id: str, log_path: Path) -> None:
        if not log_path.exists():
            return
        try:
            with self._lock:
                offset = self._log_offsets.get(job_id, 0)
            with log_path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                new_offset = handle.tell()
        except Exception:
            return

        if not chunk:
            return

        text = chunk.decode("utf-8", errors="replace")
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        with self._lock:
            self._log_offsets[job_id] = new_offset
        for line in lines[-120:]:
            self.append_log(job_id, line)

    def _tail_log(self, log_path: Path, limit: int = 3) -> list[str]:
        if not log_path.exists():
            return []
        try:
            return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        except Exception:
            return []

    def _extract_final_metrics(self, run_dir: Path) -> dict[str, Any]:
        results_csv_path = run_dir / "yolo_output" / "train" / "results.csv"
        if not results_csv_path.exists():
            return {}
        try:
            with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            return {}
        if not rows:
            return {}
        last_row = rows[-1]
        metrics: dict[str, Any] = {}
        field_map = {
            "metrics/mAP50(B)": "mAP50",
            "metrics/mAP50-95(B)": "mAP50_95",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        for source_key, target_key in field_map.items():
            raw_value = last_row.get(source_key)
            if raw_value in {None, ""}:
                continue
            try:
                metrics[target_key] = round(float(raw_value), 6)
            except Exception:
                continue
        return metrics
