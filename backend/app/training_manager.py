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
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    best_checkpoint: str | None = None
    error: str | None = None
    pid: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status,
            "progress": round(self.progress, 2),
            "current_epoch": int(self.current_epoch or 0),
            "total_epochs": int(self.total_epochs or 0),
            "logs": self.logs[-250:],
            "created_at": self.created_at,
            "config": self.config,
            "metrics": self.metrics or {},
            "best_checkpoint": self.best_checkpoint,
            "error": self.error,
        }
        for key in ("dataset_scan", "generated_dataset_yaml", "generated_split_lists", "validation_strategy", "backend", "backend_version", "artifacts"):
            value = getattr(self, key, None)
            if value is not None:
                payload[key] = value
        return payload

    def persistence_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["run_dir"] = self.run_dir
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
                    "metrics": job.metrics or {},
                    "best_checkpoint": job.best_checkpoint,
                }
            )
        return runs

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)
            self._persist_locked(job)

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
            job.config = dict(config)
            job.total_epochs = int(config.get("epochs") or job.total_epochs or 0)
            self._persist_locked(job)
            self._log_offsets[job_id] = current_offset
            self._seen_epochs[job_id] = int(job.current_epoch or 0)
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
                config=dict(payload.get("config") or {}),
                metrics=dict(payload.get("metrics") or {}),
                best_checkpoint=str(payload.get("best_checkpoint")) if payload.get("best_checkpoint") else None,
                error=str(payload.get("error")) if payload.get("error") else None,
                pid=int(payload.get("pid")) if payload.get("pid") else None,
            )
            if not job.job_id:
                continue

            for key in ("dataset_scan", "generated_dataset_yaml", "generated_split_lists", "validation_strategy", "backend", "backend_version", "artifacts"):
                if key in payload:
                    setattr(job, key, payload.get(key))

            if job.status in {"queued", "running", "stopping"} and job.config:
                job.status = "queued"
                job.pid = None
                job.progress = 0.0
                job.error = None
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

            if latest.status in {"stopped", "stopping"}:
                self.update(job_id, status="stopped", pid=None, error=None)
                self.append_log(job_id, "Training stopped")
            elif return_code == 0:
                best_checkpoint = run_dir / "weights" / "best.pt"
                metrics = self._extract_final_metrics(run_dir) or latest.metrics or {}
                if best_checkpoint.exists():
                    self.update(
                        job_id,
                        status="completed",
                        progress=100.0,
                        current_epoch=max(int(latest.current_epoch or 0), int(latest.total_epochs or 0)),
                        metrics=metrics,
                        best_checkpoint=str(best_checkpoint.resolve()),
                        pid=None,
                        error=None,
                    )
                    self.append_log(job_id, "Training completed")
                else:
                    self.update(
                        job_id,
                        status="failed",
                        pid=None,
                        error="Training exited successfully but no best checkpoint was produced.",
                    )
            else:
                tail = " ".join(self._tail_log(log_path, limit=3))
                self.update(
                    job_id,
                    status="failed",
                    pid=None,
                    error=f"Training process exited with code {return_code}. {tail}".strip(),
                )
            with self._lock:
                self._log_offsets.pop(job_id, None)
                self._seen_epochs.pop(job_id, None)
            break

    def _job_state_path(self, run_dir: Path | str) -> Path:
        return Path(run_dir) / "job_state.json"

    def _write_config(self, run_dir: Path, config: dict[str, Any]) -> None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _persist_locked(self, job: TrainingJobState) -> None:
        run_dir = Path(job.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._job_state_path(run_dir).write_text(json.dumps(job.persistence_dict(), indent=2), encoding="utf-8")

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
        done = bool(payload.get("done", False))
        progress = 100.0 if done else min((epoch / total_epochs) * 100.0, 99.0)

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.current_epoch = epoch
            job.total_epochs = total_epochs
            job.progress = max(float(job.progress or 0.0), float(progress))
            if metrics:
                job.metrics = metrics
            self._persist_locked(job)
            previous_epoch = self._seen_epochs.get(job_id, 0)
            self._seen_epochs[job_id] = max(previous_epoch, epoch)

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
