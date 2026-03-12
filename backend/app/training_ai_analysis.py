from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.ai_diagnostics import (
    MAX_RECENT_LOGS,
    ProviderConfig,
    _build_code_slice,
    _build_diagnostics_model,
    extract_json_object,
    fit_code_context_budget,
    fit_prompt_context_budget,
    resolve_provider_config,
    trim_recent_logs,
)

PROMPT_VERSION = "training-analysis-v2"
ARTIFACT_FILENAME = "training_analysis_ai.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEBACK_FILE_RE = re.compile(r'File "(?P<path>.+?)", line (?P<line>\d+), in (?P<function>.+)$')

ALLOWED_SECTION_IDS = [
    "run_outcome",
    "dataset_contract",
    "training_dynamics",
    "validation_signal",
    "artifact_readiness",
]
ALLOWED_SECTION_ID_SET = set(ALLOWED_SECTION_IDS)


def _build_good_output_example() -> str:
    return json.dumps(
        {
            "summary_line": "Failed before epoch 1 on MPS during the backward pass, so the run produced no usable optimization signal and no activation-worthy checkpoint.",
            "overall_status": "blocked",
            "activation_recommendation": "reject",
            "sections": [
                {
                    "id": "run_outcome",
                    "title": "Run outcome and failure point",
                    "status": "warn",
                    "summary": "The worker exited almost immediately with a backward-pass runtime error, so this is not a weak-model result. It is a runtime execution failure.",
                    "details": "The run finished at 1% progress with `0 / 50` epochs completed and the log tail shows `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`. That means optimization never settled into a real training loop, so every later artifact is partial at best.",
                    "evidence": [
                        "Status is `failed`.",
                        "Epoch progress is `0 / 50`.",
                        "The terminal error includes `does not require grad and does not have a grad_fn`.",
                    ],
                    "actions": [
                        "Treat this as a backend/runtime bug first, not a dataset-quality failure.",
                        "Reproduce on CPU after the code path is inspected so you can separate MPS-specific behavior from general callback logic.",
                    ],
                    "implementation_diagnosis": "The most likely failure is the custom optimizer-step instrumentation inside `backend/app/train_worker.py:152::on_pretrain_routine_end`, where the worker overrides `trainer.optimizer_step` and directly manipulates scaler and gradients. A no-grad tensor at backward time points at the training callback path or a backend-specific autograd interaction rather than the dataset manifest.",
                    "suggested_fix": "Guard the optimizer-step override behind explicit backend compatibility checks, log the resolved trainer/scaler state before the first backward pass, and fall back to the stock Ultralytics optimizer step when the callback path produces no-grad failures.",
                    "code_refs": [
                        "backend/app/train_worker.py:152::on_pretrain_routine_end",
                        "backend/app/train_worker.py:214::main",
                        "backend/app/training_manager.py::_poll_process",
                    ],
                    "evidence_keys": [
                        "status",
                        "progress",
                        "current_epoch",
                        "total_epochs",
                        "error",
                    ],
                    "artifact_refs": [
                        "train_log",
                        "progress",
                    ],
                },
                {
                    "id": "dataset_contract",
                    "title": "Dataset contract and football mapping",
                    "status": "neutral",
                    "summary": "The dataset contract appears usable enough to start, so it is not the primary explanation for this specific crash.",
                    "details": "The scan tier is usable, the runtime manifest was written, and the player or ball class mapping exists. That still deserves review before the next run, but the current evidence does not point to label parsing or split discovery as the immediate blocker.",
                    "evidence": [
                        "Dataset scan allowed training to start.",
                        "A generated `dataset_runtime.yaml` exists for the run.",
                        "Football class IDs were derived before the worker launched.",
                    ],
                    "actions": [
                        "Keep the dataset contract attached to the bug report so runtime debugging happens with the exact manifest that launched the run.",
                    ],
                    "implementation_diagnosis": "",
                    "suggested_fix": "",
                    "code_refs": [],
                    "evidence_keys": [
                        "dataset_scan.tier",
                        "validation_strategy",
                    ],
                    "artifact_refs": [
                        "dataset_scan",
                        "generated_dataset_yaml",
                    ],
                },
                {
                    "id": "training_dynamics",
                    "title": "Optimization dynamics",
                    "status": "warn",
                    "summary": "There are no trustworthy training dynamics to interpret because the run failed before a real epoch completed.",
                    "details": "No stable loss trajectory or metric trend exists. Any curve samples captured before the crash only prove the process started, not that the model learned anything.",
                    "evidence": [
                        "Loss samples are absent or nearly empty.",
                        "No completed epoch metrics were recorded.",
                    ],
                    "actions": [
                        "Do not compare this run against successful runs by curve shape or early `mAP50`.",
                        "Focus on making epoch 1 complete cleanly before tuning epochs, freeze, or batch.",
                    ],
                    "implementation_diagnosis": "The current worker writes curve points from callback hooks, but those hooks can still fire before the training loop is healthy. Without a completed epoch, the right diagnosis surface is the callback and autograd path, not the curve chart.",
                    "suggested_fix": "Add an explicit `first_successful_backward` or `first_completed_epoch` marker to `progress.json` so the UI can distinguish `training started` from `training became valid`.",
                    "code_refs": [
                        "backend/app/train_worker.py:97::record_curve_point",
                        "backend/app/train_worker.py:180::on_epoch_end",
                    ],
                    "evidence_keys": [
                        "training_curves",
                        "current_epoch",
                    ],
                    "artifact_refs": [
                        "progress",
                        "results_csv",
                    ],
                },
                {
                    "id": "validation_signal",
                    "title": "Validation signal and result quality",
                    "status": "warn",
                    "summary": "There is no meaningful validation result to trust from this run.",
                    "details": "Without completed epoch metrics, `mAP50`, `mAP50-95`, precision, and recall are missing or defaulted. The absence of metrics here is expected after the runtime failure, but it means this run contributes no evidence about detector quality.",
                    "evidence": [
                        "Validation metrics are missing or `n/a`.",
                        "No `results.csv` signal is available to compare against prior runs.",
                    ],
                    "actions": [
                        "Exclude this run from checkpoint comparison or activation decisions.",
                    ],
                    "implementation_diagnosis": "Metric extraction in `backend/app/training_manager.py::_extract_final_metrics` depends on the Ultralytics results export. When the worker exits before that export is written, the missing metrics are downstream of the runtime crash, not an independent reporting bug.",
                    "suggested_fix": "Keep the current extraction path, but add an explicit `metrics_unavailable_reason` field to the training summary when `results.csv` never materializes.",
                    "code_refs": [
                        "backend/app/training_manager.py::_extract_final_metrics",
                        "backend/app/train_worker.py:214::main",
                    ],
                    "evidence_keys": [
                        "metrics",
                        "status",
                    ],
                    "artifact_refs": [
                        "results_csv",
                        "args_yaml",
                    ],
                },
                {
                    "id": "artifact_readiness",
                    "title": "Artifacts, durability, and activation readiness",
                    "status": "warn",
                    "summary": "This run is not activation-ready because it failed and did not produce a trustworthy best checkpoint.",
                    "details": "A provenance file may still exist because the studio records lineage early, but provenance alone does not make a failed run promotable. Activation should remain blocked until a completed run writes `weights/best.pt` and clean validation evidence.",
                    "evidence": [
                        "No trustworthy best checkpoint was produced.",
                        "The run ended in `failed` status.",
                    ],
                    "actions": [
                        "Do not promote or activate this run.",
                        "Keep the provenance artifact for debugging, but treat it as failure lineage only.",
                    ],
                    "implementation_diagnosis": "The registry path is correctly stricter than the training summary. Activation in `backend/app/main.py::activate_training_run` requires `status == completed` and a real checkpoint path, so the product already blocks accidental promotion here.",
                    "suggested_fix": "Surface the activation block directly in the AI review with an explicit recommendation pill so the operator does not need to infer it from status plus missing checkpoint.",
                    "code_refs": [
                        "backend/app/main.py::activate_training_run",
                        "backend/app/training_registry.py::activate_detector",
                    ],
                    "evidence_keys": [
                        "best_checkpoint",
                        "training_provenance_path",
                    ],
                    "artifact_refs": [
                        "best_checkpoint",
                        "training_provenance",
                    ],
                },
            ],
        },
        indent=2,
    )


GOOD_OUTPUT_EXAMPLE = _build_good_output_example()


class TrainingAnalysisSection(BaseModel):
    id: str
    title: str
    status: str
    summary: str
    details: str
    evidence: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    implementation_diagnosis: str = ""
    suggested_fix: str = ""
    code_refs: list[str] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class TrainingAnalysisOutput(BaseModel):
    summary_line: str
    overall_status: str
    activation_recommendation: str
    sections: list[TrainingAnalysisSection] = Field(default_factory=list)


@dataclass
class RunSnapshot:
    run_id: str
    run_name: str
    run_dir: Path
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    logs: list[str]
    error: str | None
    config: dict[str, Any]
    dataset_scan: dict[str, Any] | None
    generated_dataset_yaml: str | None
    generated_split_lists: dict[str, str]
    validation_strategy: str | None
    resolved_device: str | None
    backend: str | None
    backend_version: str | None
    metrics: dict[str, Any]
    training_curves: dict[str, list[dict[str, Any]]]
    artifacts: dict[str, Any]
    best_checkpoint: str | None
    training_provenance_path: str | None
    training_provenance: dict[str, Any] | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None


@dataclass
class TracebackFrame:
    path: str
    line: int
    function: str
    is_local: bool

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.line}::{self.function}"


def artifact_path_for_run(run_dir: Path | str) -> Path:
    return Path(run_dir) / ARTIFACT_FILENAME


def build_system_prompt() -> str:
    return f"""
You are generating an implementation-aware AI review for one Training Studio detector fine-tuning run.

Stable repository facts:
- This repository is a browser-first football analysis and detector-fine-tuning workbench.
- Training Studio is detector-only in V1. It fine-tunes from football-pretrained `soccana`, writes a run-local dataset manifest, records training provenance, and can later activate a checkpoint for analysis.
- A completed training run is only valuable if it yields a checkpoint that is trustworthy for downstream football analysis. Good detector metrics are not enough by themselves.
- The supplied code excerpts are current implementation truth. Prefer them over generic YOLO advice.

What to optimize for:
- Explain what actually happened in this specific run, not generic model-training lore.
- Treat dataset contract, runtime failure mode, optimization dynamics, validation signal, and activation readiness as separate review categories.
- Use the logs, config, metrics, dataset scan, artifact paths, provenance, and code excerpts together.
- Warn sections must name the exact function, condition, callback, fallback, or artifact gate that most likely explains the problem.
- When traceback frames are available, anchor the diagnosis to the deepest relevant local frame and use line-specific refs like `backend/app/train_worker.py:214::main`.
- If the crash is in third-party code, still identify the nearest local entrypoint and the local hook, override, device resolution rule, or config path most likely contributing.
- Give the operator and the engineer concrete next actions. Avoid vague verbs like "inspect" unless you also say exactly what to inspect and why.

Output contract:
- Return valid JSON only.
- Use this exact top-level schema:
  {{"summary_line":"string","overall_status":"good|mixed|blocked","activation_recommendation":"activate|hold|reject","sections":[{{"id":"run_outcome|dataset_contract|training_dynamics|validation_signal|artifact_readiness","title":"string","status":"good|neutral|warn","summary":"string","details":"string","evidence":["string"],"actions":["string"],"implementation_diagnosis":"string","suggested_fix":"string","code_refs":["path::symbol"],"evidence_keys":["metric_or_field"],"artifact_refs":["artifact_key"]}}]}}
- `code_refs` may use either `path::symbol` or `path:line::function`, but prefer line-specific refs whenever the traceback or local code context makes that possible.
- Produce exactly five sections, one for each required `id`, in this order:
  1. `run_outcome`
  2. `dataset_contract`
  3. `training_dynamics`
  4. `validation_signal`
  5. `artifact_readiness`
- Titles should be short enough to fit inside collapsible UI headers, but the body text should be verbose and educational.
- Each section must include at least one `evidence` item and one `actions` item.
- Each `warn` section must include `implementation_diagnosis`, `suggested_fix`, and at least one `code_refs` entry.
- For failed runs, the `run_outcome` section must go beyond the status string. It must attempt a root-cause diagnosis grounded in the failure context and point to local code that could change the outcome.
- `artifact_refs` should use keys like `train_log`, `progress`, `results_csv`, `args_yaml`, `best_checkpoint`, `generated_dataset_yaml`, `dataset_scan`, or `training_provenance`.
- Do not wrap the JSON in markdown fences.
- Use the example below as the quality bar for depth and specificity. Match its detail level, not its literal outcome.

Exact output example:
{GOOD_OUTPUT_EXAMPLE}

Writing rules:
- Do not invent evidence that is not present in the supplied context.
- Do not default to cheerful "training best practices" filler.
- If the run failed before a real epoch completed, say that plainly and refuse to over-interpret the metrics.
- Do not stop at \"the run failed\". Failed runs are the highest-priority case for root-cause analysis.
- If the run completed but the validation strategy is weak, explain why the checkpoint may still be a hold instead of an activation candidate.
- If the code context is insufficient to justify a concrete patch, say that explicitly in `implementation_diagnosis` instead of faking certainty.
""".strip()


def _status_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_metric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if numeric == numeric else None


def _tail_path(value: str | None) -> str:
    if not value:
        return "missing"
    parts = [part for part in re.split(r"[\\/]+", str(value)) if part]
    if len(parts) <= 4:
        return str(value)
    return ".../" + "/".join(parts[-4:])


def _safe_curve_points(training_curves: dict[str, list[dict[str, Any]]] | None, key: str) -> list[dict[str, Any]]:
    values = training_curves.get(key) if isinstance(training_curves, dict) else None
    return values if isinstance(values, list) else []


def _interesting_log_lines(logs: list[str]) -> list[str]:
    keywords = (
        "error",
        "exception",
        "runtimeerror",
        "traceback",
        "warning",
        "failed",
        "backward",
        "nan",
        "cuda",
        "mps",
        "oom",
        "epoch ",
        "map50",
        "precision",
        "recall",
        "completed",
        "stopped",
        "queued",
        "manifest",
    )
    selected: list[str] = []
    seen: set[str] = set()
    for line in reversed(logs):
        lowered = line.lower()
        if any(token in lowered for token in keywords):
            if line in seen:
                continue
            seen.add(line)
            selected.append(line)
        if len(selected) >= 24:
            break
    return list(reversed(selected))


def _normalize_trace_path(raw_path: str) -> tuple[str, bool]:
    candidate = Path(str(raw_path).strip()).expanduser()
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        path_text = str(resolved)
        if "site-packages/" in path_text:
            return path_text.split("site-packages/", 1)[1], False
        if ".venv/" in path_text:
            return path_text.split(".venv/", 1)[1], False
        return str(resolved), False
    return relative.as_posix(), True


def _extract_traceback_frames(logs: list[str]) -> list[TracebackFrame]:
    frames: list[TracebackFrame] = []
    for raw_line in logs:
        line = str(raw_line).strip()
        match = TRACEBACK_FILE_RE.search(line)
        if not match:
            continue
        path, is_local = _normalize_trace_path(match.group("path"))
        try:
            line_number = int(match.group("line"))
        except Exception:
            continue
        function_name = str(match.group("function") or "").strip() or "<unknown>"
        frames.append(
            TracebackFrame(
                path=path,
                line=line_number,
                function=function_name,
                is_local=is_local,
            )
        )
    return frames


def _find_anchor_line(path_relative: str, anchor: str) -> int | None:
    path = REPO_ROOT / path_relative
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    needle = anchor.strip()
    for index, line in enumerate(lines, start=1):
        if line.strip().startswith(needle):
            return index
    return None


def _build_line_ref(path_relative: str, line_number: int | None, function_name: str) -> str:
    if line_number is None:
        return f"{path_relative}::{function_name}"
    return f"{path_relative}:{line_number}::{function_name}"


def _build_anchor_ref(path_relative: str, anchor: str, function_name: str) -> str:
    return _build_line_ref(path_relative, _find_anchor_line(path_relative, anchor), function_name)


def _build_failure_context(snapshot: RunSnapshot, recent_logs: list[str]) -> dict[str, Any]:
    combined_logs = [*snapshot.logs, *recent_logs]
    traceback_frames = _extract_traceback_frames(combined_logs)
    local_frames = [frame for frame in traceback_frames if frame.is_local]
    external_frames = [frame for frame in traceback_frames if not frame.is_local]
    error_text = " ".join([snapshot.error or "", *combined_logs[-60:]]).lower()
    signal_flags = {
        "mentions_mps": "mps" in error_text or _status_text(snapshot.resolved_device) == "mps",
        "mentions_backward": "backward" in error_text,
        "mentions_no_grad": "does not require grad" in error_text or "grad_fn" in error_text,
        "mentions_pin_memory_warning": "pin_memory" in error_text,
        "mentions_accelerator_error": "acceleratorerror" in error_text,
        "mentions_out_of_bounds": "out of bounds" in error_text,
    }
    candidate_local_code_refs: list[str] = []
    if local_frames:
        candidate_local_code_refs.extend(frame.ref for frame in local_frames[-4:])
    if signal_flags["mentions_mps"]:
        candidate_local_code_refs.append(_build_anchor_ref("backend/app/train_worker.py", "def choose_training_device(", "choose_training_device"))
    if signal_flags["mentions_no_grad"] or signal_flags["mentions_backward"]:
        candidate_local_code_refs.extend(
            [
                _build_anchor_ref("backend/app/train_worker.py", "def on_pretrain_routine_end(", "on_pretrain_routine_end"),
                _build_anchor_ref("backend/app/train_worker.py", "def main()", "main"),
            ]
        )

    seen_refs: set[str] = set()
    unique_candidate_refs: list[str] = []
    for ref in candidate_local_code_refs:
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        unique_candidate_refs.append(ref)

    return {
        "local_frames": [frame.__dict__ | {"ref": frame.ref} for frame in local_frames[:8]],
        "external_frames": [frame.__dict__ | {"ref": frame.ref} for frame in external_frames[:8]],
        "candidate_local_code_refs": unique_candidate_refs[:8],
        "signal_flags": signal_flags,
    }


def _load_recent_training_logs(snapshot: RunSnapshot) -> list[str]:
    if snapshot.logs:
        return trim_recent_logs(snapshot.logs[-MAX_RECENT_LOGS:])
    log_path = snapshot.run_dir / "train.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return trim_recent_logs(lines[-MAX_RECENT_LOGS:])


def _infer_issue_categories(snapshot: RunSnapshot) -> list[str]:
    categories: list[str] = []
    status = _status_text(snapshot.status)
    error_text = " ".join([snapshot.error or "", *snapshot.logs[-40:]]).lower()
    dataset_tier = _status_text((snapshot.dataset_scan or {}).get("tier"))
    validation_strategy = _status_text(snapshot.validation_strategy or (snapshot.dataset_scan or {}).get("suggested_validation_strategy"))
    best_checkpoint = Path(str(snapshot.best_checkpoint)).expanduser() if snapshot.best_checkpoint else None
    metrics = snapshot.metrics or {}

    if status in {"failed", "stopped"} or snapshot.error:
        categories.append("runtime_failure")
    if "mps" in error_text or _status_text(snapshot.resolved_device) == "mps":
        categories.append("device_runtime")
    if dataset_tier in {"invalid", "usable_with_warnings"} or (snapshot.dataset_scan or {}).get("warnings"):
        categories.append("dataset_contract")
    if validation_strategy in {"generate_from_train", "reuse_train_single_image"}:
        categories.append("validation_strategy")
    if not metrics or _normalize_metric(metrics.get("mAP50")) is None:
        categories.append("metrics_reporting")
    if best_checkpoint is None or not best_checkpoint.exists():
        categories.append("artifact_readiness")
    if (snapshot.training_provenance or {}).get("load_error"):
        categories.append("durability")
    return categories


def build_code_context(snapshot: RunSnapshot) -> list[dict[str, Any]]:
    categories = _infer_issue_categories(snapshot)
    slices: list[dict[str, Any]] = []

    if "dataset_contract" in categories or "validation_strategy" in categories:
        slices.extend(
            [
                _build_code_slice(
                    "backend/app/training.py",
                    label="dataset_inspection",
                    reason="Dataset scan, class mapping, and validation-strategy inference for Training Studio.",
                    anchor="def inspect_training_dataset(",
                    before=0,
                    after=120,
                ),
                _build_code_slice(
                    "backend/app/training.py",
                    label="runtime_dataset_manifest",
                    reason="Run-local manifest generation and split list materialization before worker launch.",
                    anchor="def prepare_training_run_inputs(",
                    before=0,
                    after=90,
                ),
            ]
        )

    if "runtime_failure" in categories or "device_runtime" in categories:
        slices.extend(
            [
                _build_code_slice(
                    "backend/app/train_worker.py",
                    label="training_device_resolution",
                    reason="Resolved training device selection for auto, MPS, CUDA, and CPU.",
                    anchor="def choose_training_device(",
                    before=0,
                    after=28,
                ),
                _build_code_slice(
                    "backend/app/train_worker.py",
                    label="training_optimizer_override",
                    reason="Custom optimizer-step instrumentation and grad-norm capture added on top of the Ultralytics trainer.",
                    anchor="def on_pretrain_routine_end(",
                    before=0,
                    after=22,
                ),
                _build_code_slice(
                    "backend/app/train_worker.py",
                    label="training_worker_main",
                    reason="Ultralytics training worker setup, callback wiring, and artifact writing.",
                    anchor="def main()",
                    before=0,
                    after=210,
                ),
                _build_code_slice(
                    "backend/app/training_manager.py",
                    label="training_process_poll",
                    reason="Training subprocess completion handling, checkpoint checks, and final status persistence.",
                    anchor="def _poll_process(",
                    before=0,
                    after=110,
                ),
            ]
        )

    if "metrics_reporting" in categories:
        slices.append(
            _build_code_slice(
                "backend/app/training_manager.py",
                label="final_metric_extraction",
                reason="How final detector metrics are extracted from Ultralytics results exports.",
                anchor="def _extract_final_metrics(",
                before=0,
                after=40,
            )
        )

    if "artifact_readiness" in categories or "durability" in categories:
        slices.extend(
            [
                _build_code_slice(
                    "backend/app/main.py",
                    label="training_activation_gate",
                    reason="Activation and promoted-checkpoint requirements for completed detector runs.",
                    anchor="def activate_training_run(",
                    before=0,
                    after=90,
                ),
                _build_code_slice(
                    "backend/app/training_registry.py",
                    label="registry_activation",
                    reason="Registry validation before a custom detector becomes active for analysis.",
                    anchor="def activate_detector(",
                    before=0,
                    after=70,
                ),
            ]
        )

    unique: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for item in slices:
        label = str(item.get("label") or "")
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        unique.append(item)
    return fit_code_context_budget(unique)


def build_run_context(snapshot: RunSnapshot, recent_logs: list[str], code_context: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_scan = snapshot.dataset_scan or {}
    training_provenance = snapshot.training_provenance or {}
    metrics = snapshot.metrics or {}
    loss_points = _safe_curve_points(snapshot.training_curves, "loss")
    optimizer_points = _safe_curve_points(snapshot.training_curves, "optimizer")
    best_checkpoint_exists = bool(snapshot.best_checkpoint and Path(snapshot.best_checkpoint).expanduser().exists())
    failure_context = _build_failure_context(snapshot, recent_logs)

    return {
        "prompt_version": PROMPT_VERSION,
        "analysis_goal": "Produce an implementation-aware training run review with clear activation guidance and code-level debugging steps when the run is weak.",
        "run_identity": {
            "run_id": snapshot.run_id,
            "run_name": snapshot.run_name,
            "status": snapshot.status,
            "created_at": snapshot.created_at,
            "started_at": snapshot.started_at,
            "finished_at": snapshot.finished_at,
        },
        "operator_config": {
            "base_weights": snapshot.config.get("base_weights"),
            "dataset_path": snapshot.config.get("dataset_path"),
            "run_name": snapshot.config.get("run_name"),
            "epochs": snapshot.config.get("epochs"),
            "imgsz": snapshot.config.get("imgsz"),
            "batch": snapshot.config.get("batch"),
            "requested_device": snapshot.config.get("device"),
            "resolved_device": snapshot.resolved_device,
            "workers": snapshot.config.get("workers"),
            "patience": snapshot.config.get("patience"),
            "freeze": snapshot.config.get("freeze"),
            "cache": snapshot.config.get("cache"),
            "backend": snapshot.backend,
            "backend_version": snapshot.backend_version,
        },
        "run_state": {
            "progress": snapshot.progress,
            "current_epoch": snapshot.current_epoch,
            "total_epochs": snapshot.total_epochs,
            "error": snapshot.error,
            "best_checkpoint_exists": best_checkpoint_exists,
        },
        "dataset_contract": {
            "tier": dataset_scan.get("tier"),
            "yaml_path": dataset_scan.get("yaml_path"),
            "classes_source": dataset_scan.get("classes_source"),
            "classes": dataset_scan.get("classes") or [],
            "class_mapping": dataset_scan.get("class_mapping") or {},
            "split_paths": dataset_scan.get("split_paths") or {},
            "splits": dataset_scan.get("splits") or {},
            "warnings": list(dataset_scan.get("warnings") or [])[:12],
            "errors": list(dataset_scan.get("errors") or [])[:12],
            "validation_strategy": snapshot.validation_strategy or dataset_scan.get("suggested_validation_strategy"),
            "generated_dataset_yaml": snapshot.generated_dataset_yaml,
            "generated_split_lists": snapshot.generated_split_lists,
        },
        "training_metrics": {
            "mAP50": metrics.get("mAP50"),
            "mAP50_95": metrics.get("mAP50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
        },
        "curve_summary": {
            "loss_points": len(loss_points),
            "optimizer_points": len(optimizer_points),
            "last_loss_point": loss_points[-1] if loss_points else None,
            "last_optimizer_point": optimizer_points[-1] if optimizer_points else None,
        },
        "artifact_state": {
            "config": snapshot.artifacts.get("config"),
            "dataset_scan": snapshot.artifacts.get("dataset_scan"),
            "generated_dataset_yaml": snapshot.artifacts.get("generated_dataset_yaml"),
            "train_log": snapshot.artifacts.get("train_log"),
            "progress": snapshot.artifacts.get("progress"),
            "results_csv": snapshot.artifacts.get("results_csv"),
            "args_yaml": snapshot.artifacts.get("args_yaml"),
            "best_checkpoint": snapshot.artifacts.get("best_checkpoint") or snapshot.best_checkpoint,
            "training_provenance": snapshot.artifacts.get("training_provenance") or snapshot.training_provenance_path,
            "promoted_checkpoint": snapshot.artifacts.get("promoted_checkpoint"),
            "plots": list(snapshot.artifacts.get("plots") or [])[:8],
        },
        "durability": {
            "training_provenance": training_provenance,
        },
        "failure_context": failure_context,
        "log_highlights": _interesting_log_lines(recent_logs),
        "recent_logs": recent_logs,
        "code_context": code_context,
    }


def render_context_for_provider(context: dict[str, Any]) -> str:
    code_blocks = [
        f"FILE: {item.get('path')}\nANCHOR: {item.get('anchor')}\n{item.get('excerpt')}"
        for item in (context.get("code_context") or [])
    ]
    sections = [
        ("ANALYSIS GOAL", context.get("analysis_goal")),
        ("RUN IDENTITY", json.dumps(context.get("run_identity") or {}, indent=2)),
        ("OPERATOR CONFIG", json.dumps(context.get("operator_config") or {}, indent=2)),
        ("RUN STATE", json.dumps(context.get("run_state") or {}, indent=2)),
        ("DATASET CONTRACT", json.dumps(context.get("dataset_contract") or {}, indent=2)),
        ("TRAINING METRICS", json.dumps(context.get("training_metrics") or {}, indent=2)),
        ("CURVE SUMMARY", json.dumps(context.get("curve_summary") or {}, indent=2)),
        ("ARTIFACT STATE", json.dumps(context.get("artifact_state") or {}, indent=2)),
        ("DURABILITY", json.dumps(context.get("durability") or {}, indent=2)),
        ("FAILURE CONTEXT", json.dumps(context.get("failure_context") or {}, indent=2)),
    ]
    rendered = [f"{title}\n{body}" for title, body in sections if body]
    if context.get("log_highlights"):
        rendered.append("LOG HIGHLIGHTS\n" + "\n".join(str(line) for line in context["log_highlights"]))
    if context.get("recent_logs"):
        rendered.append("RECENT LOGS\n" + "\n".join(str(line) for line in context["recent_logs"]))
    if code_blocks:
        rendered.append("LIVE IMPLEMENTATION CODE\n" + "\n\n".join(code_blocks))
    return "\n\n".join(rendered).strip()


def _normalize_section(item: TrainingAnalysisSection) -> TrainingAnalysisSection:
    section_id = item.id.strip()
    if section_id not in ALLOWED_SECTION_ID_SET:
        raise ValueError(f"Unsupported training analysis section id: {item.id}")
    status = item.status.strip().lower()
    if status not in {"good", "neutral", "warn"}:
        raise ValueError(f"Unsupported training analysis section status: {item.status}")

    normalized = TrainingAnalysisSection(
        id=section_id,
        title=item.title.strip(),
        status=status,
        summary=item.summary.strip(),
        details=item.details.strip(),
        evidence=[str(entry).strip() for entry in item.evidence if str(entry).strip()][:6],
        actions=[str(entry).strip() for entry in item.actions if str(entry).strip()][:6],
        implementation_diagnosis=item.implementation_diagnosis.strip(),
        suggested_fix=item.suggested_fix.strip(),
        code_refs=[str(ref).strip() for ref in item.code_refs if str(ref).strip()][:8],
        evidence_keys=[str(key).strip() for key in item.evidence_keys if str(key).strip()][:12],
        artifact_refs=[str(ref).strip() for ref in item.artifact_refs if str(ref).strip()][:12],
    )
    if not normalized.title or not normalized.summary or not normalized.details:
        raise ValueError(f"Training analysis section {section_id} must include title, summary, and details")
    if not normalized.evidence:
        raise ValueError(f"Training analysis section {section_id} must include at least one evidence item")
    if not normalized.actions:
        raise ValueError(f"Training analysis section {section_id} must include at least one action item")
    if normalized.status == "warn":
        missing: list[str] = []
        if not normalized.implementation_diagnosis:
            missing.append("implementation_diagnosis")
        if not normalized.suggested_fix:
            missing.append("suggested_fix")
        if not normalized.code_refs:
            missing.append("code_refs")
        if missing:
            raise ValueError(f"Warn section '{normalized.id}' is missing {', '.join(missing)}")
    return normalized


def _normalize_output(output: TrainingAnalysisOutput) -> TrainingAnalysisOutput:
    summary_line = output.summary_line.strip()
    if not summary_line:
        raise ValueError("summary_line must not be empty")
    overall_status = output.overall_status.strip().lower()
    if overall_status not in {"good", "mixed", "blocked"}:
        raise ValueError(f"Unsupported overall_status: {output.overall_status}")
    activation_recommendation = output.activation_recommendation.strip().lower()
    if activation_recommendation not in {"activate", "hold", "reject"}:
        raise ValueError(f"Unsupported activation_recommendation: {output.activation_recommendation}")

    normalized_sections = [_normalize_section(section) for section in output.sections]
    if len(normalized_sections) != len(ALLOWED_SECTION_IDS):
        raise ValueError(f"Training analysis must include exactly {len(ALLOWED_SECTION_IDS)} sections")

    ids = [section.id for section in normalized_sections]
    if ids != ALLOWED_SECTION_IDS:
        raise ValueError(f"Training analysis sections must be ordered as {', '.join(ALLOWED_SECTION_IDS)}")

    return TrainingAnalysisOutput(
        summary_line=summary_line,
        overall_status=overall_status,
        activation_recommendation=activation_recommendation,
        sections=normalized_sections,
    )


def build_training_analysis_agent(model: Any, system_prompt: str, *, max_output_tokens: int, timeout_seconds: float) -> Any:
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent = Agent(
        model=model,
        output_type=TrainingAnalysisOutput,
        instructions=system_prompt,
        model_settings={
            "temperature": 0.1,
            "max_tokens": max_output_tokens,
            "timeout": timeout_seconds,
        },
        output_retries=2,
    )

    @agent.output_validator
    def validate_output(_ctx: RunContext[None], output: TrainingAnalysisOutput) -> TrainingAnalysisOutput:
        try:
            return _normalize_output(output)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    return agent


def call_provider_via_pydantic_ai(config: ProviderConfig, system_prompt: str, context: dict[str, Any]) -> str:
    model = _build_diagnostics_model(config)
    agent = build_training_analysis_agent(
        model,
        system_prompt,
        max_output_tokens=config.max_output_tokens,
        timeout_seconds=config.timeout_seconds,
    )
    result = agent.run_sync(render_context_for_provider(context))
    normalized_output = _normalize_output(result.output)
    return json.dumps(normalized_output.model_dump(mode="json"), ensure_ascii=True)


def _heuristic_section(
    *,
    section_id: str,
    title: str,
    status: str,
    summary: str,
    details: str,
    evidence: list[str],
    actions: list[str],
    implementation_diagnosis: str = "",
    suggested_fix: str = "",
    code_refs: list[str] | None = None,
    evidence_keys: list[str] | None = None,
    artifact_refs: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": section_id,
        "title": title,
        "status": status,
        "summary": summary,
        "details": details,
        "evidence": evidence,
        "actions": actions,
        "evidence_keys": evidence_keys or [],
        "artifact_refs": artifact_refs or [],
    }
    if implementation_diagnosis:
        payload["implementation_diagnosis"] = implementation_diagnosis
    if suggested_fix:
        payload["suggested_fix"] = suggested_fix
    if code_refs:
        payload["code_refs"] = code_refs
    return payload


def build_heuristic_analysis(snapshot: RunSnapshot) -> dict[str, Any]:
    dataset_scan = snapshot.dataset_scan or {}
    training_provenance = snapshot.training_provenance or {}
    metrics = snapshot.metrics or {}
    recent_logs = _load_recent_training_logs(snapshot)
    failure_context = _build_failure_context(snapshot, recent_logs)
    mAP50 = _normalize_metric(metrics.get("mAP50"))
    mAP50_95 = _normalize_metric(metrics.get("mAP50_95"))
    precision = _normalize_metric(metrics.get("precision"))
    recall = _normalize_metric(metrics.get("recall"))
    loss_points = _safe_curve_points(snapshot.training_curves, "loss")
    optimizer_points = _safe_curve_points(snapshot.training_curves, "optimizer")
    best_checkpoint_exists = bool(snapshot.best_checkpoint and Path(snapshot.best_checkpoint).expanduser().exists())
    dataset_tier = _status_text(dataset_scan.get("tier"))
    validation_strategy = str(snapshot.validation_strategy or dataset_scan.get("suggested_validation_strategy") or "unknown")
    warnings = list(dataset_scan.get("warnings") or [])
    errors = list(dataset_scan.get("errors") or [])
    dvc_runtime = (training_provenance.get("dvc_runtime") if isinstance(training_provenance, dict) else None) or {}
    overall_status = "mixed"
    activation_recommendation = "hold"

    if _status_text(snapshot.status) in {"failed", "stopped"} or not best_checkpoint_exists:
        overall_status = "blocked"
        activation_recommendation = "reject"
    elif mAP50 is not None and mAP50 >= 0.65 and validation_strategy == "existing_split" and dataset_tier == "valid":
        overall_status = "good"
        activation_recommendation = "activate"

    outcome_warn = _status_text(snapshot.status) in {"failed", "stopped"} or bool(snapshot.error)
    local_trace_refs = list(failure_context.get("candidate_local_code_refs") or [])
    nearest_local_frame = next(iter(failure_context.get("local_frames") or []), None)
    no_grad_failure = bool((failure_context.get("signal_flags") or {}).get("mentions_no_grad"))
    backward_failure = bool((failure_context.get("signal_flags") or {}).get("mentions_backward"))
    mps_failure = bool((failure_context.get("signal_flags") or {}).get("mentions_mps"))
    runtime_diagnosis = (
        "The nearest local traceback frame is "
        f"`{nearest_local_frame['ref']}` and the failure signature includes a no-grad backward error on the MPS training path. "
        "That makes the local suspects `choose_training_device`, the custom optimizer-step override in `on_pretrain_routine_end`, "
        "and the `model.train(**train_kwargs)` call site more relevant than dataset parsing."
        if nearest_local_frame and no_grad_failure and mps_failure
        else (
            "The nearest local traceback frame is "
            f"`{nearest_local_frame['ref']}`. The worker failed before a completed epoch, so the first code to inspect is the local training entrypoint and any local callbacks or overrides used before or during backward."
            if nearest_local_frame
            else "The final status is assigned in `backend/app/training_manager.py::_poll_process`, which reads the worker exit code, checks for a best checkpoint, and persists the terminal summary state."
        )
    )
    runtime_fix = (
        "Reproduce with `device=cpu`, log `trainer.loss.requires_grad` and AMP/scaler state immediately before backward, and temporarily disable the custom `trainer.optimizer_step` override to isolate whether `backend/app/train_worker.py:152::on_pretrain_routine_end` is involved. "
        "If CPU succeeds and MPS fails on the same manifest, stop auto-selecting MPS for detector training until a verified compatible path exists."
        if no_grad_failure and (mps_failure or backward_failure)
        else "Capture the first local traceback frame, then patch the corresponding worker entrypoint or callback path before rerunning."
    )
    outcome_section = _heuristic_section(
        section_id="run_outcome",
        title="Run outcome and failure point",
        status="warn" if outcome_warn else "good",
        summary=(
            (
                f"The run failed before epoch 1 and the nearest local frame is `{nearest_local_frame['ref']}`."
                if outcome_warn and nearest_local_frame
                else f"The run ended as `{snapshot.status}` at {snapshot.current_epoch}/{snapshot.total_epochs} epochs."
            )
            if outcome_warn
            else f"The run completed at {snapshot.current_epoch}/{snapshot.total_epochs} epochs with a persisted checkpoint."
        ),
        details=(
            (
                f"Progress reached {snapshot.progress:.1f}% and the terminal error was `{snapshot.error}`. "
                f"Failure flags: no-grad={no_grad_failure}, backward={backward_failure}, mps={mps_failure}."
            )
            if outcome_warn and snapshot.error
            else f"Progress reached {snapshot.progress:.1f}% and Training Studio recorded the run as `{snapshot.status}`."
        ),
        evidence=[
            f"Status: {snapshot.status}",
            f"Epoch progress: {snapshot.current_epoch}/{snapshot.total_epochs}",
            f"Error: {snapshot.error}" if snapshot.error else "No terminal error was recorded.",
            f"Nearest local frame: {nearest_local_frame['ref']}" if nearest_local_frame else "No local traceback frame was parsed from the logs.",
        ],
        actions=[
            "Treat failed or stopped runs as execution evidence first and model-quality evidence second.",
            "Use the log tail and persisted progress artifact together before changing hyperparameters.",
            "Prefer changing the first local failure path over retuning data or epochs when the traceback never reaches a completed epoch.",
        ],
        implementation_diagnosis=(runtime_diagnosis if outcome_warn else ""),
        suggested_fix=(runtime_fix if outcome_warn else ""),
        code_refs=(local_trace_refs or ["backend/app/training_manager.py::_poll_process"] if outcome_warn else []),
        evidence_keys=["status", "progress", "current_epoch", "total_epochs", "error"],
        artifact_refs=["train_log", "progress"],
    )

    dataset_warn = dataset_tier in {"invalid", "usable_with_warnings"} or bool(errors)
    dataset_section = _heuristic_section(
        section_id="dataset_contract",
        title="Dataset contract and football mapping",
        status="warn" if dataset_warn else ("neutral" if warnings else "good"),
        summary=(
            f"Dataset intake tier is `{dataset_scan.get('tier') or 'unknown'}` with validation strategy `{validation_strategy}`."
        ),
        details=(
            f"The scanner found player IDs {dataset_scan.get('class_mapping', {}).get('player_class_ids', [])}, "
            f"ball IDs {dataset_scan.get('class_mapping', {}).get('ball_class_ids', [])}, "
            f"and referee IDs {dataset_scan.get('class_mapping', {}).get('referee_class_ids', [])}. "
            f"Warnings: {len(warnings)}. Errors: {len(errors)}."
        ),
        evidence=[
            f"Tier: {dataset_scan.get('tier') or 'unknown'}",
            f"Validation strategy: {validation_strategy}",
            f"Warnings: {len(warnings)}",
            f"Errors: {len(errors)}",
        ],
        actions=[
            "Keep the exact dataset scan snapshot with the run when comparing checkpoints later.",
            "If class mapping or split quality is weak, fix that contract before trusting better-looking metrics.",
        ],
        implementation_diagnosis=(
            "Training Studio derives football class IDs and validation strategy in `backend/app/training.py::inspect_training_dataset`, then materializes a run-local manifest in `prepare_training_run_inputs`."
            if dataset_warn
            else ""
        ),
        suggested_fix=(
            "Correct the dataset YAML names or split structure so the generated runtime manifest does not rely on warning-heavy fallback behavior."
            if dataset_warn
            else ""
        ),
        code_refs=(
            [
                "backend/app/training.py::inspect_training_dataset",
                "backend/app/training.py::prepare_training_run_inputs",
            ]
            if dataset_warn
            else []
        ),
        evidence_keys=["dataset_scan", "validation_strategy"],
        artifact_refs=["dataset_scan", "generated_dataset_yaml"],
    )

    dynamics_warn = outcome_warn or snapshot.current_epoch == 0 or not loss_points
    dynamics_section = _heuristic_section(
        section_id="training_dynamics",
        title="Optimization dynamics",
        status="warn" if dynamics_warn else ("neutral" if len(loss_points) < 8 else "good"),
        summary=(
            "The run never produced enough stable optimization signal to interpret."
            if dynamics_warn
            else "The stored loss and optimizer traces are usable as a coarse sanity signal."
        ),
        details=(
            f"Loss points: {len(loss_points)}. Optimizer points: {len(optimizer_points)}. "
            f"Last loss point: {loss_points[-1] if loss_points else 'none'}."
        ),
        evidence=[
            f"Loss points recorded: {len(loss_points)}",
            f"Optimizer points recorded: {len(optimizer_points)}",
            f"Current epoch: {snapshot.current_epoch}",
        ],
        actions=[
            "Do not over-read curve shapes from runs that failed before a completed epoch.",
            "If curves are sparse, stabilize the worker first, then compare freeze, batch, or image size.",
        ],
        implementation_diagnosis=(
            "Curve samples are emitted by callback hooks in `backend/app/train_worker.py`, so sparse curves usually mean the worker exited before the training loop reached a stable cadence."
            if dynamics_warn
            else ""
        ),
        suggested_fix=(
            "Persist an explicit first-completed-epoch marker in `progress.json` so the UI can distinguish `worker booted` from `training dynamics are real`."
            if dynamics_warn
            else ""
        ),
        code_refs=(
            [
                _build_anchor_ref("backend/app/train_worker.py", "def record_curve_point(", "record_curve_point"),
                _build_anchor_ref("backend/app/train_worker.py", "def on_epoch_end(", "on_epoch_end"),
            ]
            if dynamics_warn
            else []
        ),
        evidence_keys=["training_curves", "current_epoch"],
        artifact_refs=["progress", "results_csv"],
    )

    weak_validation = (
        mAP50 is None
        or validation_strategy in {"generate_from_train", "reuse_train_single_image"}
        or (mAP50 is not None and mAP50 < 0.45)
    )
    validation_section = _heuristic_section(
        section_id="validation_signal",
        title="Validation signal and result quality",
        status="warn" if weak_validation else "good",
        summary=(
            "The validation evidence is incomplete or too weak to justify activation."
            if weak_validation
            else "The run produced a usable validation signal for checkpoint comparison."
        ),
        details=(
            f"mAP50={mAP50 if mAP50 is not None else 'n/a'}, "
            f"mAP50-95={mAP50_95 if mAP50_95 is not None else 'n/a'}, "
            f"precision={precision if precision is not None else 'n/a'}, "
            f"recall={recall if recall is not None else 'n/a'}, "
            f"validation strategy={validation_strategy}."
        ),
        evidence=[
            f"mAP50: {mAP50 if mAP50 is not None else 'n/a'}",
            f"mAP50-95: {mAP50_95 if mAP50_95 is not None else 'n/a'}",
            f"Validation strategy: {validation_strategy}",
        ],
        actions=[
            "Read detector metrics together with validation strategy instead of treating the score alone as truth.",
            "If metrics are missing, debug the worker or artifact export before comparing this run against successful checkpoints.",
        ],
        implementation_diagnosis=(
            "Training metrics are extracted from Ultralytics exports in `backend/app/training_manager.py::_extract_final_metrics`, so missing or weak values can reflect missing results files or an improvised validation split rather than a purely bad model."
            if weak_validation
            else ""
        ),
        suggested_fix=(
            "Add an explicit metrics-availability reason to the summary and prioritize a real validation split for promotion-grade runs."
            if weak_validation
            else ""
        ),
        code_refs=(
            [
                "backend/app/training_manager.py::_extract_final_metrics",
                _build_anchor_ref("backend/app/training.py", "def prepare_training_run_inputs(", "prepare_training_run_inputs"),
            ]
            if weak_validation
            else []
        ),
        evidence_keys=["metrics", "validation_strategy"],
        artifact_refs=["results_csv", "args_yaml"],
    )

    readiness_warn = (
        _status_text(snapshot.status) != "completed"
        or not best_checkpoint_exists
        or validation_section["status"] == "warn"
    )
    readiness_section = _heuristic_section(
        section_id="artifact_readiness",
        title="Artifacts, durability, and activation readiness",
        status="warn" if readiness_warn else "good",
        summary=(
            "This run is not ready for detector activation."
            if readiness_warn
            else "Artifacts and provenance are strong enough for activation review."
        ),
        details=(
            f"Best checkpoint: {_tail_path(snapshot.best_checkpoint)}. "
            f"Provenance file: {_tail_path(snapshot.training_provenance_path)}. "
            f"DVC runtime status: {dvc_runtime.get('status') or 'unknown'}."
        ),
        evidence=[
            f"Best checkpoint exists: {'yes' if best_checkpoint_exists else 'no'}",
            f"Run status: {snapshot.status}",
            f"Provenance path: {snapshot.training_provenance_path or 'missing'}",
        ],
        actions=[
            "Only promote runs that are completed, checkpoint-backed, and supported by believable validation evidence.",
            "Keep provenance even for rejected runs so the failure lineage stays auditable.",
        ],
        implementation_diagnosis=(
            "Activation is intentionally stricter than mere run completion. `backend/app/main.py::activate_training_run` and the registry layer both require a completed run and a real checkpoint path before analysis can use it."
            if readiness_warn
            else ""
        ),
        suggested_fix=(
            "Surface activation blockers directly from the summary so the UI can explain whether the block came from status, checkpoint absence, or weak validation evidence."
            if readiness_warn
            else ""
        ),
        code_refs=(
            [
                "backend/app/main.py::activate_training_run",
                "backend/app/training_registry.py::activate_detector",
            ]
            if readiness_warn
            else []
        ),
        evidence_keys=["best_checkpoint", "training_provenance_path", "status"],
        artifact_refs=["best_checkpoint", "training_provenance"],
    )

    if outcome_warn and snapshot.error:
        summary_line = (
            f"{snapshot.status.capitalize()} at {snapshot.current_epoch}/{snapshot.total_epochs} epochs on "
            f"{snapshot.resolved_device or snapshot.config.get('device') or 'unknown device'} with `{snapshot.error}`. "
            f"Start with {local_trace_refs[0] if local_trace_refs else 'backend/app/train_worker.py::main'} before changing dataset or hyperparameters."
        )
    elif overall_status == "good":
        summary_line = (
            f"Completed {snapshot.current_epoch}/{snapshot.total_epochs} epochs from {snapshot.config.get('base_weights') or 'soccana'} "
            f"with mAP50 {mAP50:.3f} on an existing validation split; this looks like a real activation candidate."
            if mAP50 is not None
            else f"Completed cleanly from {snapshot.config.get('base_weights') or 'soccana'} with a checkpoint and stable artifacts."
        )
    else:
        summary_line = (
            f"Completed training from {snapshot.config.get('base_weights') or 'soccana'}, but the run should stay on hold because "
            f"validation strategy is `{validation_strategy}` and the artifact evidence is not strong enough for blind activation."
        )

    return {
        "summary_line": summary_line,
        "overall_status": overall_status,
        "activation_recommendation": activation_recommendation,
        "sections": [
            outcome_section,
            dataset_section,
            dynamics_section,
            validation_section,
            readiness_section,
        ],
    }


def sanitize_training_analysis(candidate: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return fallback
    try:
        normalized = _normalize_output(TrainingAnalysisOutput.model_validate(candidate))
    except Exception:
        return fallback
    return normalized.model_dump(mode="json")


def build_snapshot(payload: dict[str, Any]) -> RunSnapshot:
    return RunSnapshot(
        run_id=str(payload.get("run_id") or ""),
        run_name=str((payload.get("config") or {}).get("run_name") or payload.get("run_id") or ""),
        run_dir=Path(str(payload.get("run_dir") or "")),
        status=str(payload.get("status") or "unknown"),
        progress=float(payload.get("progress") or 0.0),
        current_epoch=int(payload.get("current_epoch") or 0),
        total_epochs=int(payload.get("total_epochs") or 0),
        logs=list(payload.get("logs") or []),
        error=str(payload.get("error")) if payload.get("error") else None,
        config=dict(payload.get("config") or {}),
        dataset_scan=dict(payload.get("dataset_scan") or {}) or None,
        generated_dataset_yaml=str(payload.get("generated_dataset_yaml")) if payload.get("generated_dataset_yaml") else None,
        generated_split_lists=dict(payload.get("generated_split_lists") or {}),
        validation_strategy=str(payload.get("validation_strategy")) if payload.get("validation_strategy") else None,
        resolved_device=str(payload.get("resolved_device")) if payload.get("resolved_device") else None,
        backend=str(payload.get("backend")) if payload.get("backend") else None,
        backend_version=str(payload.get("backend_version")) if payload.get("backend_version") else None,
        metrics=dict(payload.get("metrics") or {}),
        training_curves=dict(payload.get("training_curves") or {}),
        artifacts=dict(payload.get("artifacts") or {}),
        best_checkpoint=str(payload.get("best_checkpoint")) if payload.get("best_checkpoint") else None,
        training_provenance_path=str(payload.get("training_provenance_path")) if payload.get("training_provenance_path") else None,
        training_provenance=dict(payload.get("training_provenance") or {}) or None,
        created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
        started_at=str(payload.get("started_at")) if payload.get("started_at") else None,
        finished_at=str(payload.get("finished_at")) if payload.get("finished_at") else None,
    )


def generate_training_run_analysis(
    payload: dict[str, Any],
    *,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = build_snapshot(payload)
    fallback = build_heuristic_analysis(snapshot)
    config = resolve_provider_config()
    artifact_path = artifact_path_for_run(snapshot.run_dir)
    artifact: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "disabled",
        "provider": None,
        "model": None,
        "summary_line": fallback["summary_line"],
        "overall_status": fallback["overall_status"],
        "activation_recommendation": fallback["activation_recommendation"],
        "error": "",
        "raw_text": "",
        "prompt_context": {
            "log_highlights": [],
            "recent_logs": [],
            "failure_context": {},
            "code_context": [],
            "budget": {
                "max_output_tokens": None,
                "context_json_chars": 0,
                "code_slice_count": 0,
                "recent_log_count": 0,
            },
        },
        "sections": fallback["sections"],
    }

    if config is None:
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return fallback, artifact

    if log_callback is not None:
        log_callback(f"Generating AI training review via {config.provider}:{config.model}")

    recent_logs = _load_recent_training_logs(snapshot)
    code_context = build_code_context(snapshot)
    context = build_run_context(snapshot, recent_logs, code_context)
    context = fit_prompt_context_budget(context)
    prompt_context = {
        "log_highlights": context.get("log_highlights", []),
        "recent_logs": context.get("recent_logs", []),
        "failure_context": context.get("failure_context", {}),
        "code_context": context.get("code_context", []),
        "budget": {
            "max_output_tokens": config.max_output_tokens,
            "context_json_chars": len(json.dumps(context)),
            "code_slice_count": len(context.get("code_context", [])),
            "recent_log_count": len(context.get("recent_logs", [])),
        },
    }

    try:
        raw_text = call_provider_via_pydantic_ai(config, build_system_prompt(), context)
        parsed = extract_json_object(raw_text)
        analysis = sanitize_training_analysis(parsed, fallback)
        artifact = {
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "completed",
            "provider": config.provider,
            "model": config.model,
            "summary_line": analysis["summary_line"],
            "overall_status": analysis["overall_status"],
            "activation_recommendation": analysis["activation_recommendation"],
            "error": "",
            "raw_text": raw_text,
            "prompt_context": prompt_context,
            "sections": analysis["sections"],
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return analysis, artifact
    except Exception as exc:
        artifact = {
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "failed",
            "provider": config.provider,
            "model": config.model,
            "summary_line": fallback["summary_line"],
            "overall_status": fallback["overall_status"],
            "activation_recommendation": fallback["activation_recommendation"],
            "error": str(exc),
            "raw_text": "",
            "prompt_context": prompt_context,
            "sections": fallback["sections"],
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        if log_callback is not None:
            log_callback(f"AI training review failed; using heuristic fallback. {exc}")
        return fallback, artifact


def build_summary_updates(analysis: dict[str, Any], artifact: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    source = "ai" if artifact.get("status") == "completed" else "heuristic"
    return {
        "training_analysis_source": source,
        "training_analysis_provider": artifact.get("provider"),
        "training_analysis_model": artifact.get("model"),
        "training_analysis_status": artifact.get("status"),
        "training_analysis_summary_line": artifact.get("summary_line", ""),
        "training_analysis_error": artifact.get("error", ""),
        "training_analysis_json": str(artifact_path_for_run(run_dir).resolve()),
        "training_analysis_prompt_version": artifact.get("prompt_version"),
        "training_analysis_current_prompt_version": PROMPT_VERSION,
        "training_analysis_stale": False,
        "training_analysis_stale_reason": "",
        "training_analysis_overall_status": analysis.get("overall_status", "mixed"),
        "training_analysis_activation_recommendation": analysis.get("activation_recommendation", "hold"),
        "training_analysis_sections": list(analysis.get("sections") or []),
    }


def normalize_training_analysis_fields(payload: dict[str, Any], *, run_dir: Path | None = None) -> dict[str, Any]:
    normalized = dict(payload)
    raw_run_dir = str(run_dir) if run_dir is not None else str(normalized.get("run_dir") or "")
    effective_run_dir = Path(raw_run_dir) if raw_run_dir.strip() else None
    artifact_path = artifact_path_for_run(effective_run_dir) if effective_run_dir is not None else None
    prompt_version = normalized.get("training_analysis_prompt_version")
    if not prompt_version and artifact_path and artifact_path.exists():
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            prompt_version = artifact.get("prompt_version")
        except Exception:
            prompt_version = None

    normalized.setdefault("training_analysis_source", "heuristic")
    normalized.setdefault("training_analysis_provider", None)
    normalized.setdefault("training_analysis_model", None)
    normalized.setdefault("training_analysis_status", "unknown")
    normalized.setdefault("training_analysis_summary_line", "")
    normalized.setdefault("training_analysis_error", "")
    if artifact_path:
        normalized.setdefault("training_analysis_json", str(artifact_path.resolve()))
    else:
        normalized.setdefault("training_analysis_json", None)
    normalized["training_analysis_prompt_version"] = prompt_version
    normalized["training_analysis_current_prompt_version"] = PROMPT_VERSION
    normalized["training_analysis_stale"] = bool(
        normalized.get("training_analysis_source") == "ai"
        and prompt_version != PROMPT_VERSION
    )
    normalized["training_analysis_stale_reason"] = (
        "Stored AI training analysis is from an older Training Studio prompt build and may not match the current runtime behavior."
        if normalized["training_analysis_stale"]
        else ""
    )
    normalized.setdefault("training_analysis_overall_status", "mixed")
    normalized.setdefault("training_analysis_activation_recommendation", "hold")
    normalized["training_analysis_sections"] = list(normalized.get("training_analysis_sections") or [])
    return normalized
