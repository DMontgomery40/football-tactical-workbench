from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from pydantic import BaseModel, Field


PROMPT_VERSION = "run-diagnostics-v8"
DEFAULT_TIMEOUT_SECONDS = 75.0
DEFAULT_MAX_OUTPUT_TOKENS = 3000

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_CODE_SLICE_LINES = 220
MAX_CODE_SLICE_CHARS = 6000
MAX_TOTAL_CODE_CONTEXT_CHARS = 45000
MAX_CODE_SLICES = 6
MAX_RECENT_LOGS = 120
MAX_RECENT_LOG_CHARS = 4000
MAX_CONTEXT_JSON_CHARS = 70000


def _build_good_output_example() -> str:
    return json.dumps(
        {
            "summary_line": "300-frame run where detector filtering likely collapsed upstream of tracking, leaving calibration alive but player, ball, team, and projection outputs empty.",
            "diagnostics": [
                {
                    "level": "warn",
                    "title": "Detector filtering is likely removing valid classes",
                    "message": (
                        "The run produced 0 player rows and 0 ball rows even though the detector checkpoint loaded and the pipeline continued through overlay rendering. "
                        "The most likely code-level cause is the detector class-resolution layer in `backend/app/wide_angle.py::resolve_detector_spec`: "
                        "if the resolved `player_detector_class_ids` or `ball_detector_class_ids` do not match the checkpoint's emitted labels, "
                        "the `classes=[...]` filter can zero both streams before tracking starts."
                    ),
                    "next_step": (
                        "Compare `player_detector_class_ids`, `ball_detector_class_ids`, `raw_detector_boxes_sampled`, and `raw_detector_class_histogram_sample`. "
                        "If sampled raw classes show detector output outside the resolved ids, fix the metadata mapping or dataset YAML before touching tracker settings."
                    ),
                    "implementation_diagnosis": (
                        "Detection emptiness occurs upstream of tracking. The class-resolution result from `resolve_detector_spec` is consumed directly by `detect_players_for_frame` "
                        "and the ball branch through the `classes=[...]` filter, so any mismatch between resolved ids and actual checkpoint labels erases detections before ByteTrack or ReID can help."
                    ),
                    "suggested_fix": (
                        "Resolve player/ball/referee ids from checkpoint metadata or adjacent dataset YAML, log the resolved ids at startup, and sample raw class histograms on the first few frames. "
                        "If metadata is missing, fail fast instead of guessing."
                    ),
                    "code_refs": [
                        "backend/app/wide_angle.py::resolve_detector_spec",
                        "backend/app/wide_angle.py::detect_players_for_frame",
                        "backend/app/wide_angle.py::analyze_video",
                    ],
                    "evidence_keys": [
                        "player_rows",
                        "ball_rows",
                        "player_model",
                        "ball_model",
                    ],
                },
                {
                    "level": "warn",
                    "title": "Calibration is rejecting on gates tighter than visibility",
                    "message": (
                        "Calibration attempts are not failing just because pitch keypoints disappear. `detect_pitch_homography` can form a candidate with 4 visible points, but the runtime acceptance gate additionally requires "
                        "`visible_count >= MIN_CALIBRATION_VISIBLE_KEYPOINTS`, `inlier_count >= MIN_CALIBRATION_INLIERS`, `reprojection_error <= MAX_CALIBRATION_REPROJECTION_ERROR_CM`, and "
                        "`temporal_drift <= MAX_CALIBRATION_TEMPORAL_DRIFT_CM`. When visible keypoints remain above the minimum but success rate is still poor, the failing logic is the acceptance gate, not the model loader."
                    ),
                    "next_step": (
                        "Capture and log the exact rejecting term per frame. The code already computes `visible_count`, `inlier_count`, `reprojection_error`, and `temporal_drift`; persist those values and report which gate failed instead of emitting one aggregate refresh-rate number."
                    ),
                    "implementation_diagnosis": (
                        "The rejection logic lives in the `candidate_is_usable` condition inside the main analysis loop. Without per-gate logging, diagnostics cannot distinguish 'no candidate homography' from 'candidate rejected by drift/inliers/error', so the user gets a weak aggregate warning."
                    ),
                    "suggested_fix": (
                        "Add per-frame calibration rejection reasons to the run summary or diagnostics context. Record one of: `no_candidate`, `low_visible_count`, `low_inliers`, `high_reprojection_error`, or `high_temporal_drift`, and preserve whether stale recovery mode was active."
                    ),
                    "code_refs": [
                        "backend/app/wide_angle.py::detect_pitch_homography",
                        "backend/app/wide_angle.py::analyze_video",
                    ],
                    "evidence_keys": [
                        "field_calibration_refresh_attempts",
                        "field_calibration_refresh_successes",
                        "field_calibration_refresh_rejections",
                        "average_visible_pitch_keypoints",
                    ],
                },
                {
                    "level": "warn",
                    "title": "Projection exporter is downstream-empty, not independently broken",
                    "message": (
                        "If `projected_player_points=0`, `projected_ball_points=0`, `field_registered_frames=0`, and `projection_csv` is null while detection rows are also zero, projection is not the primary failure. "
                        "The projection path depends on both a non-stale homography and actual player/ball anchors, so upstream detector collapse makes the exporter look dead even when the projection code itself is unchanged."
                    ),
                    "next_step": (
                        "Do not patch the projection exporter first. Fix the detector/class-id failure, rerun, and only patch projection if projected points stay at zero after player rows recover."
                    ),
                    "implementation_diagnosis": (
                        "The exporter is operating on empty upstream inputs. In the active loop, player detections are projected only after detection and calibration state exist; with zero detections, the projection CSV will naturally remain empty."
                    ),
                    "suggested_fix": (
                        "Keep the exporter as-is for now, but add a diagnostic note or status flag that explicitly distinguishes `projection unavailable because upstream anchors were empty` from `projection code failed`."
                    ),
                    "code_refs": [
                        "backend/app/wide_angle.py::detect_players_for_frame",
                        "backend/app/wide_angle.py::analyze_video",
                    ],
                    "evidence_keys": [
                        "projected_player_points",
                        "projected_ball_points",
                        "field_registered_frames",
                        "projection_csv",
                    ],
                },
            ],
        },
        indent=2,
    )


GOOD_OUTPUT_EXAMPLE = _build_good_output_example()


def load_project_env() -> None:
    for env_path in (REPO_ROOT / ".env", REPO_ROOT / ".env.local"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


load_project_env()


@dataclass
class ProviderConfig:
    provider: str
    model: str
    endpoint: str
    base_url: str
    api_key: str
    timeout_seconds: float
    extra_headers: dict[str, str]
    max_output_tokens: int


class DiagnosticsAgentItem(BaseModel):
    level: str
    title: str
    message: str = ""
    next_step: str = ""
    implementation_diagnosis: str = ""
    suggested_fix: str = ""
    code_refs: list[str] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)


class DiagnosticsAgentOutput(BaseModel):
    summary_line: str
    diagnostics: list[DiagnosticsAgentItem] = Field(default_factory=list)


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _normalize_openai_compatible_base_url(raw_value: str) -> str:
    normalized = str(raw_value or "").rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    return normalized


def resolve_provider_config() -> ProviderConfig | None:
    provider_pref = _env("AI_DIAGNOSTICS_PROVIDER", "auto").lower()
    timeout_seconds = float(_env("AI_DIAGNOSTICS_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)) or DEFAULT_TIMEOUT_SECONDS)
    max_output_tokens = int(_env("AI_DIAGNOSTICS_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)) or DEFAULT_MAX_OUTPUT_TOKENS)
    shared_model = _env("AI_DIAGNOSTICS_MODEL")
    local_base_url = _env("AI_DIAGNOSTICS_BASE_URL") or _env("OPENAI_COMPAT_BASE_URL") or _env("LOCAL_LLM_BASE_URL")
    local_api_key = _env("AI_DIAGNOSTICS_API_KEY") or _env("OPENAI_COMPAT_API_KEY") or _env("LOCAL_LLM_API_KEY")

    if provider_pref in {"off", "none", "disabled"}:
        return None

    def build_openai() -> ProviderConfig | None:
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            return None
        return ProviderConfig(
            provider="openai",
            model=shared_model or _env("OPENAI_MODEL") or "gpt-5.4",
            endpoint="https://api.openai.com/v1/responses",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers={},
            max_output_tokens=max_output_tokens,
        )

    def build_openrouter() -> ProviderConfig | None:
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            return None
        headers = {}
        if _env("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = _env("OPENROUTER_HTTP_REFERER")
        if _env("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = _env("OPENROUTER_APP_TITLE")
        return ProviderConfig(
            provider="openrouter",
            model=shared_model or _env("OPENROUTER_MODEL") or "openai/gpt-4.1-mini",
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers=headers,
            max_output_tokens=max_output_tokens,
        )

    def build_anthropic() -> ProviderConfig | None:
        api_key = _env("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return ProviderConfig(
            provider="anthropic",
            model=shared_model or _env("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
            endpoint="https://api.anthropic.com/v1/messages",
            base_url="https://api.anthropic.com",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers={"anthropic-version": _env("ANTHROPIC_VERSION") or "2023-06-01"},
            max_output_tokens=max_output_tokens,
        )

    def build_local() -> ProviderConfig | None:
        if not local_base_url:
            return None
        normalized_base_url = _normalize_openai_compatible_base_url(local_base_url)
        endpoint = f"{normalized_base_url}/chat/completions"
        return ProviderConfig(
            provider="local",
            model=shared_model or _env("LOCAL_LLM_MODEL") or "gpt-oss-20b",
            endpoint=endpoint,
            base_url=normalized_base_url,
            api_key=local_api_key or "local",
            timeout_seconds=timeout_seconds,
            extra_headers={},
            max_output_tokens=max_output_tokens,
        )

    builders = {
        "openai": build_openai,
        "openrouter": build_openrouter,
        "anthropic": build_anthropic,
        "local": build_local,
    }

    if provider_pref in builders:
        return builders[provider_pref]()

    for provider_name in ("openai", "openrouter", "anthropic", "local"):
        config = builders[provider_name]()
        if config is not None:
            return config
    return None


def _read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find_anchor_index(lines: list[str], anchor: str) -> int:
    needle = anchor.strip()
    for index, line in enumerate(lines):
        if line.strip().startswith(needle):
            return index
    for index, line in enumerate(lines):
        if needle in line:
            return index
    return 0


def _extract_top_level_block(lines: list[str], anchor_index: int) -> list[str]:
    if not (0 <= anchor_index < len(lines)):
        return []
    line = lines[anchor_index]
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if not stripped.startswith(("def ", "async def ", "class ")):
        return []

    end_index = anchor_index + 1
    while end_index < len(lines):
        candidate = lines[end_index]
        candidate_stripped = candidate.lstrip()
        candidate_indent = len(candidate) - len(candidate_stripped)
        if candidate_stripped and candidate_indent == indent and candidate_stripped.startswith(("def ", "async def ", "class ")):
            break
        end_index += 1
    return lines[anchor_index:end_index]


def _excerpt_lines(lines: list[str], anchor: str, before: int, after: int) -> str:
    anchor_index = _find_anchor_index(lines, anchor)
    block_lines = _extract_top_level_block(lines, anchor_index)
    if block_lines:
        excerpt = "\n".join(block_lines).rstrip()
        if len(excerpt) > MAX_CODE_SLICE_CHARS:
            return excerpt[: MAX_CODE_SLICE_CHARS].rstrip() + "\n# ...[truncated]"
        return excerpt

    start_index = max(0, anchor_index - before)
    end_index = min(len(lines), anchor_index + after + 1, start_index + MAX_CODE_SLICE_LINES)
    excerpt = "\n".join(lines[start_index:end_index]).rstrip()
    if len(excerpt) > MAX_CODE_SLICE_CHARS:
        return excerpt[: MAX_CODE_SLICE_CHARS].rstrip() + "\n# ...[truncated]"
    return excerpt


def _build_code_slice(path_relative: str, label: str, reason: str, anchor: str, before: int = 4, after: int = 60) -> dict[str, Any]:
    path = REPO_ROOT / path_relative
    lines = _read_text_lines(path)
    return {
        "label": label,
        "reason": reason,
        "path": path_relative,
        "anchor": anchor,
        "excerpt": _excerpt_lines(lines, anchor, before=before, after=after),
    }


def trim_recent_logs(logs: list[str]) -> list[str]:
    trimmed: list[str] = []
    total_chars = 0
    for line in reversed(logs):
        line_chars = len(line) + 1
        if total_chars + line_chars > MAX_RECENT_LOG_CHARS:
            break
        trimmed.append(line)
        total_chars += line_chars
    return list(reversed(trimmed))


def fit_code_context_budget(code_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total_chars = 0
    for item in code_context:
        if len(selected) >= MAX_CODE_SLICES:
            break
        excerpt = str(item.get("excerpt") or "")
        item_chars = len(excerpt)
        if selected and (total_chars + item_chars) > MAX_TOTAL_CODE_CONTEXT_CHARS:
            continue
        selected.append(item)
        total_chars += item_chars
    return selected


def fit_prompt_context_budget(context: dict[str, Any]) -> dict[str, Any]:
    bounded = dict(context)
    serialized = json.dumps(bounded, indent=2)
    if len(serialized) <= MAX_CONTEXT_JSON_CHARS:
        return bounded

    code_context = list(bounded.get("code_context") or [])
    while code_context and len(serialized) > MAX_CONTEXT_JSON_CHARS:
        code_context.pop()
        bounded["code_context"] = code_context
        serialized = json.dumps(bounded, indent=2)

    recent_logs = list(bounded.get("recent_logs") or [])
    while recent_logs and len(serialized) > MAX_CONTEXT_JSON_CHARS:
        recent_logs = recent_logs[len(recent_logs) // 2 :]
        bounded["recent_logs"] = recent_logs
        serialized = json.dumps(bounded, indent=2)
    return bounded


def infer_issue_categories(summary: dict[str, Any], heuristic_diagnostics: list[dict[str, str]]) -> list[str]:
    categories: list[str] = []
    combined_text = " ".join(
        str(item.get(key, ""))
        for item in heuristic_diagnostics
        for key in ("title", "message", "next_step")
    ).lower()

    if (
        int(summary.get("player_rows") or 0) == 0
        or int(summary.get("ball_rows") or 0) == 0
        or float(summary.get("average_player_detections_per_frame") or 0.0) <= 0.05
    ):
        categories.append("detection")
    if (
        "field" in combined_text
        or "calibration" in combined_text
        or "homography" in combined_text
        or float(summary.get("field_registered_ratio") or 0.0) < 0.98
        or float(summary.get("field_calibration_refresh_successes") or 0.0) < float(summary.get("field_calibration_refresh_attempts") or 0.0)
    ):
        categories.append("calibration")
    if (
        "track" in combined_text
        or "fragment" in combined_text
        or "identity" in combined_text
        or "merge" in combined_text
        or summary.get("tracklet_merges_applied") is not None
    ):
        categories.append("tracking")
    if "ball" in combined_text or float(summary.get("average_ball_detections_per_frame") or 0.0) < 0.9:
        categories.append("ball")
    if "goal" in combined_text or int(summary.get("goal_events_count") or 0) == 0:
        categories.append("experiments")
    if "diagnostic" in combined_text or "prompt" in combined_text or "ui" in combined_text:
        categories.append("diagnostics")
    return categories


def build_code_context(summary: dict[str, Any], heuristic_diagnostics: list[dict[str, str]]) -> list[dict[str, Any]]:
    categories = infer_issue_categories(summary, heuristic_diagnostics)
    code_slices: list[dict[str, Any]] = []

    if "detection" in categories:
        code_slices.extend(
            [
                _build_code_slice(
                    "backend/app/wide_angle.py",
                    label="detector_spec_resolution",
                    reason="Custom detector class-id resolution and fallback mapping.",
                    anchor="def resolve_detector_spec(",
                    before=0,
                    after=40,
                ),
                _build_code_slice(
                    "backend/app/wide_angle.py",
                    label="player_detection_and_tracking",
                    reason="Player detection path and tracker-mode wiring in the active pipeline.",
                    anchor="def detect_players_for_frame(",
                    before=0,
                    after=60,
                ),
            ]
        )

    if "calibration" in categories:
        code_slices.extend(
            [
                _build_code_slice(
                    "backend/app/wide_angle.py",
                    label="calibration_constants",
                    reason="Field-calibration constants and acceptance thresholds.",
                    anchor="CALIBRATION_REFRESH_FRAMES =",
                    before=0,
                    after=18,
                ),
                _build_code_slice(
                    "backend/app/wide_angle.py",
                    label="homography_detection",
                    reason="Pitch homography detection and reprojection error calculation.",
                    anchor="def detect_pitch_homography(",
                    before=0,
                    after=55,
                ),
                _build_code_slice(
                    "backend/app/wide_angle.py",
                    label="calibration_runtime_gate",
                    reason="Runtime calibration acceptance, stale handling, and projection wiring in the active analysis loop.",
                    anchor="candidate_is_usable = (",
                    before=18,
                    after=48,
                ),
            ]
        )

    if "tracking" in categories:
        code_slices.extend(
            [
                _build_code_slice(
                    "backend/app/reid_tracker.py",
                    label="appearance_embedder",
                    reason="Sparse appearance embedder initialization and feature extraction fallback rules.",
                    anchor="class SparseAppearanceEmbedder:",
                    before=0,
                    after=70,
                ),
                _build_code_slice(
                    "backend/app/reid_tracker.py",
                    label="tracklet_stitcher",
                    reason="Current post-run track stitching logic and merge gates.",
                    anchor="def build_stitched_track_map(",
                    before=0,
                    after=75,
                ),
            ]
        )

    if "ball" in categories:
        code_slices.append(
            _build_code_slice(
                "backend/app/wide_angle.py",
                label="ball_tracking_branch",
                reason="Ball tracking branch inside the active analysis loop.",
                anchor="if include_ball and ball_model is not None:",
                before=8,
                after=55,
            )
        )

    if "diagnostics" in categories:
        code_slices.extend(
            [
                _build_code_slice(
                    "backend/app/ai_diagnostics.py",
                    label="diagnostics_prompt_contract",
                    reason="Current diagnostics system prompt and JSON contract.",
                    anchor="def build_system_prompt()",
                    before=0,
                    after=90,
                ),
                _build_code_slice(
                    "backend/app/ai_diagnostics.py",
                    label="diagnostics_context_builder",
                    reason="Current run-context assembly for the AI diagnostics call.",
                    anchor="def build_run_context(",
                    before=0,
                    after=110,
                ),
                _build_code_slice(
                    "backend/app/ai_diagnostics.py",
                    label="diagnostics_sanitizer",
                    reason="Current post-processing and clipping rules applied to AI output.",
                    anchor="def sanitize_diagnostics(",
                    before=0,
                    after=45,
                ),
            ]
        )

    unique_slices: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for item in code_slices:
        if item["label"] in seen_labels:
            continue
        seen_labels.add(item["label"])
        unique_slices.append(item)
    return fit_code_context_budget(unique_slices)


def compact_context_for_provider(context: dict[str, Any]) -> dict[str, Any]:
    code_blocks = [
        f"FILE: {item.get('path')}\nANCHOR: {item.get('anchor')}\n{item.get('excerpt')}"
        for item in (context.get("code_context") or [])
    ]
    return {
        "prompt_version": context.get("prompt_version"),
        "diagnostics_goal": context.get("diagnostics_goal"),
        "input_video": context.get("input_video"),
        "active_config": context.get("active_config"),
        "run_metrics": context.get("run_metrics"),
        "derived_metrics": context.get("derived_metrics"),
        "heuristic_diagnostics": context.get("heuristic_diagnostics"),
        "recent_logs": context.get("recent_logs"),
        "debug_artifacts": context.get("debug_artifacts"),
        "code_blocks": code_blocks,
        "experiments": context.get("experiments"),
        "top_tracks": context.get("top_tracks"),
    }


def render_context_for_provider(context: dict[str, Any]) -> str:
    compact = compact_context_for_provider(context)
    sections: list[str] = []

    sections.append(f"DIAGNOSTICS GOAL\n{compact.get('diagnostics_goal')}")
    sections.append(f"INPUT VIDEO\n{compact.get('input_video')}")
    sections.append(f"ACTIVE CONFIG\n{json.dumps(compact.get('active_config') or {}, indent=2)}")
    sections.append(f"RUN METRICS\n{json.dumps(compact.get('run_metrics') or {}, indent=2)}")
    sections.append(f"DERIVED METRICS\n{json.dumps(compact.get('derived_metrics') or {}, indent=2)}")
    sections.append(f"HEURISTIC DIAGNOSTICS\n{json.dumps(compact.get('heuristic_diagnostics') or [], indent=2)}")

    recent_logs = compact.get("recent_logs") or []
    if recent_logs:
        sections.append("RECENT LOGS\n" + "\n".join(str(line) for line in recent_logs))

    code_blocks = compact.get("code_blocks") or []
    if code_blocks:
        sections.append("LIVE IMPLEMENTATION CODE\n" + "\n\n".join(str(block) for block in code_blocks))

    experiments = compact.get("experiments") or []
    if experiments:
        sections.append(f"EXPERIMENTS\n{json.dumps(experiments, indent=2)}")

    top_tracks = compact.get("top_tracks") or []
    if top_tracks:
        sections.append(f"TOP TRACKS\n{json.dumps(top_tracks, indent=2)}")

    debug_artifacts = compact.get("debug_artifacts") or {}
    sections.append(f"DEBUG ARTIFACTS\n{json.dumps(debug_artifacts, indent=2)}")

    return "\n\n".join(section for section in sections if section).strip()


def load_recent_logs(summary: dict[str, Any], job_id: str, job_manager: Any | None) -> list[str]:
    if job_manager is not None and hasattr(job_manager, "get"):
        try:
            job_state = job_manager.get(job_id)
            if job_state is not None and getattr(job_state, "logs", None):
                return trim_recent_logs(list(job_state.logs)[-MAX_RECENT_LOGS:])
        except Exception:
            pass

    run_dir = summary.get("run_dir")
    if not run_dir:
        return []
    state_path = Path(str(run_dir)) / "job_state.json"
    if not state_path.exists():
        return []
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    logs = payload.get("logs")
    return trim_recent_logs(list(logs)[-MAX_RECENT_LOGS:]) if isinstance(logs, list) else []


def build_system_prompt() -> str:
    return f"""
You are generating debugging diagnostics for one completed football video analysis run.

Stable repository facts:
- This is a browser-first football analysis tool. The overlay video and minimap are the primary debugging artifacts.
- The active football pipeline is detector -> player tracking / identity handling -> team clustering -> field registration -> overlay / exports / diagnostics.
- The detector is football-specific (`soccana`), ball detection shares the football detector by default, and field registration uses `soccana_keypoint`.
- Player identity may run in multiple tracker modes including legacy ByteTrack and hybrid ReID / stitch paths.
- Field calibration is a live part of the pipeline, not a post-hoc note. Calibration failures can invalidate the minimap even when detection looks healthy.
- The supplied code excerpts are current implementation truth. Prefer them over generic computer-vision assumptions.

What to optimize for:
- Diagnose the run like an engineer debugging the actual implementation.
- Explain the boring implementation reasons behind failures: thresholds, gates, fallbacks, stale-state behavior, merge rules, refresh cadence, config precedence, and code paths.
- Use the supplied metrics, logs, heuristics, and code excerpts together.
- Treat `next_step` as a detailed operator / engineer action plan, not as a slogan.
- Multi-step, educational, deeply actionable guidance is required when the run is weak or mixed.

Output contract:
- Return valid JSON only.
- Use this exact top-level schema:
  {{"summary_line":"string","diagnostics":[{{"level":"good|warn","title":"string","message":"string","next_step":"string","implementation_diagnosis":"string","suggested_fix":"string","code_refs":["path::symbol"],"evidence_keys":["metric_key"]}}]}}
- Produce 3 to 5 diagnostics.
- Keep titles short enough for the UI, but messages and next_step may be long and detailed.
- Each message must cite actual numeric evidence when possible.
- Each `warn` diagnostic must name the exact function, condition, or fallback that is most likely responsible and propose a concrete code change to try.
- `next_step` is the operator action. `implementation_diagnosis` is the exact code-level explanation. `suggested_fix` is the change to make.
- Do not stop at "inspect", "review", "check", or "verify". If the code in context is enough to name the likely broken logic, do that and propose the fix directly.
- If the code is insufficient to justify a concrete patch, say that explicitly in `implementation_diagnosis` instead of faking certainty.
- Do not wrap the JSON in markdown fences.
- Use the example below as the quality target for level of detail, specificity, and actionability. Match its depth, not its literal numbers.

Exact output example:
{GOOD_OUTPUT_EXAMPLE}

Writing rules:
- Do not invent observations.
- Do not use generic QA, marketing, or coaching language.
- Prefer direct, implementation-aware language over abstract advice.
- If the supplied code reveals a likely failure mode or fallback, say so explicitly.
- If the run is missing evidence, say that plainly instead of pretending certainty.
""".strip()


def build_run_context(
    summary: dict[str, Any],
    heuristic_diagnostics: list[dict[str, str]],
    recent_logs: list[str],
    code_context: list[dict[str, Any]],
) -> dict[str, Any]:
    experiments = summary.get("experiments") or []
    top_tracks = summary.get("top_tracks") or []
    frames_processed = float(summary.get("frames_processed") or 0.0)
    unique_player_track_ids = float(summary.get("unique_player_track_ids") or 0.0)
    refresh_attempts = float(summary.get("field_calibration_refresh_attempts") or 0.0)
    refresh_successes = float(summary.get("field_calibration_refresh_successes") or 0.0)
    goal_events_count = float(summary.get("goal_events_count") or 0.0)
    return {
        "prompt_version": PROMPT_VERSION,
        "diagnostics_goal": "Produce implementation-aware diagnostics and detailed next actions grounded in the current run, current code, and current logs.",
        "input_video": Path(str(summary.get("input_video", ""))).name,
        "active_config": {
            "detector": summary.get("player_model"),
            "ball": summary.get("ball_model"),
            "field_calibration": summary.get("field_calibration_model"),
            "requested_player_tracker_mode": summary.get("requested_player_tracker_mode"),
            "player_tracker_mode": summary.get("player_tracker_mode"),
            "resolved_player_tracker_mode": summary.get("resolved_player_tracker_mode"),
            "player_tracker_runtime": summary.get("player_tracker_runtime"),
            "player_tracker_backend": summary.get("player_tracker_backend"),
            "device": summary.get("device"),
            "field_calibration_device": summary.get("field_calibration_device"),
            "player_conf": summary.get("player_conf"),
            "ball_conf": summary.get("ball_conf"),
            "iou": summary.get("iou"),
            "field_calibration_refresh_frames": summary.get("field_calibration_refresh_frames"),
            "field_keypoint_confidence_threshold": summary.get("field_keypoint_confidence_threshold"),
            "field_calibration_min_visible_keypoints": summary.get("field_calibration_min_visible_keypoints"),
            "field_calibration_stale_recovery_min_visible_keypoints": summary.get("field_calibration_stale_recovery_min_visible_keypoints"),
        },
        "run_metrics": {
            "frames_processed": summary.get("frames_processed"),
            "fps": summary.get("fps"),
            "player_rows": summary.get("player_rows"),
            "ball_rows": summary.get("ball_rows"),
            "unique_player_track_ids": summary.get("unique_player_track_ids"),
            "raw_unique_player_track_ids": summary.get("raw_unique_player_track_ids"),
            "unique_ball_track_ids": summary.get("unique_ball_track_ids"),
            "home_tracks": summary.get("home_tracks"),
            "away_tracks": summary.get("away_tracks"),
            "unassigned_tracks": summary.get("unassigned_tracks"),
            "average_player_detections_per_frame": summary.get("average_player_detections_per_frame"),
            "average_ball_detections_per_frame": summary.get("average_ball_detections_per_frame"),
            "longest_track_length": summary.get("longest_track_length"),
            "average_track_length": summary.get("average_track_length"),
            "raw_longest_track_length": summary.get("raw_longest_track_length"),
            "raw_average_track_length": summary.get("raw_average_track_length"),
            "tracklet_merges_applied": summary.get("tracklet_merges_applied"),
            "stitched_track_id_reduction": summary.get("stitched_track_id_reduction"),
            "projected_player_points": summary.get("projected_player_points"),
            "projected_ball_points": summary.get("projected_ball_points"),
            "field_registered_frames": summary.get("field_registered_frames"),
            "field_registered_ratio": summary.get("field_registered_ratio"),
            "field_calibration_refresh_frames": summary.get("field_calibration_refresh_frames"),
            "field_calibration_refresh_attempts": summary.get("field_calibration_refresh_attempts"),
            "field_calibration_refresh_successes": summary.get("field_calibration_refresh_successes"),
            "field_calibration_refresh_rejections": summary.get("field_calibration_refresh_rejections"),
            "field_calibration_stale_recovery_attempts": summary.get("field_calibration_stale_recovery_attempts"),
            "field_calibration_stale_recovery_successes": summary.get("field_calibration_stale_recovery_successes"),
            "field_calibration_stale_recovery_rejections": summary.get("field_calibration_stale_recovery_rejections"),
            "field_calibration_rejections_no_candidate": summary.get("field_calibration_rejections_no_candidate"),
            "field_calibration_rejections_low_visible_count": summary.get("field_calibration_rejections_low_visible_count"),
            "field_calibration_rejections_low_inliers": summary.get("field_calibration_rejections_low_inliers"),
            "field_calibration_rejections_high_reprojection_error": summary.get("field_calibration_rejections_high_reprojection_error"),
            "field_calibration_rejections_high_temporal_drift": summary.get("field_calibration_rejections_high_temporal_drift"),
            "field_calibration_rejections_invalid_candidate": summary.get("field_calibration_rejections_invalid_candidate"),
            "field_calibration_primary_rejections_no_candidate": summary.get("field_calibration_primary_rejections_no_candidate"),
            "field_calibration_primary_rejections_low_visible_count": summary.get("field_calibration_primary_rejections_low_visible_count"),
            "field_calibration_primary_rejections_low_inliers": summary.get("field_calibration_primary_rejections_low_inliers"),
            "field_calibration_primary_rejections_high_reprojection_error": summary.get("field_calibration_primary_rejections_high_reprojection_error"),
            "field_calibration_primary_rejections_high_temporal_drift": summary.get("field_calibration_primary_rejections_high_temporal_drift"),
            "field_calibration_primary_rejections_invalid_candidate": summary.get("field_calibration_primary_rejections_invalid_candidate"),
            "average_visible_pitch_keypoints": summary.get("average_visible_pitch_keypoints"),
            "last_good_calibration_frame": summary.get("last_good_calibration_frame"),
            "goal_events_count": summary.get("goal_events_count"),
            "team_cluster_distance": summary.get("team_cluster_distance"),
            "jersey_crops_used": summary.get("jersey_crops_used"),
            "identity_embedding_updates": summary.get("identity_embedding_updates"),
            "identity_embedding_interval_frames": summary.get("identity_embedding_interval_frames"),
            "player_tracker_stitching_enabled": summary.get("player_tracker_stitching_enabled"),
        },
        "derived_metrics": {
            "player_track_churn_ratio": round(unique_player_track_ids / frames_processed, 6) if frames_processed > 0 else None,
            "field_calibration_success_rate": round(refresh_successes / refresh_attempts, 6) if refresh_attempts > 0 else None,
            "goal_aligned_experiment": bool(goal_events_count > 0),
            "has_experiments": bool(experiments),
        },
        "heuristic_diagnostics": heuristic_diagnostics[:5],
        "recent_logs": recent_logs,
        "debug_artifacts": {
            "overlay_video": summary.get("overlay_video"),
            "detections_csv": summary.get("detections_csv"),
            "track_summary_csv": summary.get("track_summary_csv"),
            "projection_csv": summary.get("projection_csv"),
            "calibration_debug_csv": summary.get("calibration_debug_csv"),
            "summary_json": summary.get("summary_json"),
        },
        "code_context": code_context,
        "experiments": experiments,
        "top_tracks": top_tracks[:8],
    }


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)
    ssl_context = None
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = ssl.create_default_context()
    try:
        with request.urlopen(req, timeout=timeout_seconds, context=ssl_context) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from provider: {detail[:400]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Provider request failed: {exc.reason}") from exc
    if not raw.strip():
        raise RuntimeError("Provider returned an empty response body.")
    return json.loads(raw)


def _extract_text_from_openai_compatible(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Provider returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("Provider returned no message content.")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise RuntimeError("Provider returned unsupported message content.")


def _extract_text_from_openai_responses(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload.get("output_text", "").strip():
        return str(payload["output_text"])

    output = payload.get("output")
    if not isinstance(output, list):
        error_payload = payload.get("error")
        if error_payload:
            raise RuntimeError(f"OpenAI Responses API error: {error_payload}")
        raise RuntimeError("Responses API payload had no output array.")

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text" and block.get("text"):
                parts.append(str(block.get("text")))
    if parts:
        return "\n".join(parts)

    error_payload = payload.get("error")
    if error_payload:
        raise RuntimeError(f"OpenAI Responses API error: {error_payload}")
    raise RuntimeError("Responses API returned no text output.")


def _extract_text_from_anthropic(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        raise RuntimeError("Anthropic response had no content list.")
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    if not parts:
        raise RuntimeError("Anthropic response had no text parts.")
    return "\n".join(parts)


def _normalize_diagnostics_agent_output(output: DiagnosticsAgentOutput) -> DiagnosticsAgentOutput:
    summary_line = output.summary_line.strip()
    if not summary_line:
        raise ValueError("summary_line must not be empty")

    normalized_items: list[DiagnosticsAgentItem] = []
    for item in output.diagnostics:
        level = item.level.strip().lower()
        if level not in {"good", "warn"}:
            raise ValueError(f"Unsupported diagnostics level: {item.level}")

        normalized = DiagnosticsAgentItem(
            level=level,
            title=item.title.strip(),
            message=item.message.strip(),
            next_step=item.next_step.strip(),
            implementation_diagnosis=item.implementation_diagnosis.strip(),
            suggested_fix=item.suggested_fix.strip(),
            code_refs=[str(ref).strip() for ref in item.code_refs if str(ref).strip()][:8],
            evidence_keys=[str(key).strip() for key in item.evidence_keys if str(key).strip()][:12],
        )
        if not normalized.title or not normalized.message or not normalized.next_step:
            raise ValueError("Each diagnostic must include title, message, and next_step")
        if normalized.level == "warn":
            missing: list[str] = []
            if not normalized.implementation_diagnosis:
                missing.append("implementation_diagnosis")
            if not normalized.suggested_fix:
                missing.append("suggested_fix")
            if not normalized.code_refs:
                missing.append("code_refs")
            if missing:
                raise ValueError(f"Warn diagnostic '{normalized.title}' is missing {', '.join(missing)}")
        normalized_items.append(normalized)

    if not 3 <= len(normalized_items) <= 5:
        raise ValueError(f"Diagnostics output must include 3 to 5 diagnostics, got {len(normalized_items)}")

    return DiagnosticsAgentOutput(summary_line=summary_line, diagnostics=normalized_items)


def _build_openai_client(config: ProviderConfig) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=config.extra_headers or None,
    )


def call_provider_via_pydantic_ai(config: ProviderConfig, system_prompt: str, context: dict[str, Any]) -> str:
    from pydantic_ai import Agent, ModelRetry, RunContext
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    if config.provider == "openai":
        model = OpenAIResponsesModel(
            config.model,
            provider=OpenAIProvider(api_key=config.api_key, base_url=config.base_url),
        )
    elif config.provider in {"openrouter", "local"}:
        model = OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(openai_client=_build_openai_client(config)),
        )
    elif config.provider == "anthropic":
        model = AnthropicModel(
            config.model,
            provider=AnthropicProvider(
                api_key=config.api_key,
                base_url=config.base_url,
            ),
        )
    else:
        raise RuntimeError(f"Unsupported diagnostics provider: {config.provider}")

    agent = Agent(
        model=model,
        output_type=DiagnosticsAgentOutput,
        instructions=system_prompt,
        model_settings={
            "temperature": 0.1,
            "max_tokens": config.max_output_tokens,
            "timeout": config.timeout_seconds,
        },
        output_retries=2,
    )

    @agent.output_validator
    def validate_output(_ctx: RunContext[None], output: DiagnosticsAgentOutput) -> DiagnosticsAgentOutput:
        try:
            return _normalize_diagnostics_agent_output(output)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    result = agent.run_sync(render_context_for_provider(context))
    normalized_output = _normalize_diagnostics_agent_output(result.output)
    return json.dumps(normalized_output.model_dump(mode="json"), ensure_ascii=True)


def call_provider_legacy(config: ProviderConfig, system_prompt: str, context: dict[str, Any]) -> str:
    user_payload = render_context_for_provider(context)
    if config.provider == "openai":
        response_payload = _post_json(
            url=config.endpoint,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                **config.extra_headers,
            },
            payload={
                "model": config.model,
                "instructions": system_prompt,
                "input": user_payload,
                "temperature": 0.1,
                "max_output_tokens": config.max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_object",
                    }
                },
            },
            timeout_seconds=config.timeout_seconds,
        )
        return _extract_text_from_openai_responses(response_payload)

    if config.provider in {"openrouter", "local"}:
        payload = {
            "model": config.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": config.max_output_tokens,
        }
        response_payload = _post_json(
            url=config.endpoint,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                **config.extra_headers,
            },
            payload=payload,
            timeout_seconds=config.timeout_seconds,
        )
        return _extract_text_from_openai_compatible(response_payload)

    if config.provider == "anthropic":
        response_payload = _post_json(
            url=config.endpoint,
            headers={
                "x-api-key": config.api_key,
                **config.extra_headers,
            },
            payload={
                "model": config.model,
                "temperature": 0.1,
                "max_tokens": config.max_output_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_payload},
                ],
            },
            timeout_seconds=config.timeout_seconds,
        )
        return _extract_text_from_anthropic(response_payload)

    raise RuntimeError(f"Unsupported diagnostics provider: {config.provider}")


def call_provider(config: ProviderConfig, system_prompt: str, context: dict[str, Any]) -> tuple[str, str, str | None]:
    try:
        return call_provider_via_pydantic_ai(config, system_prompt, context), "pydantic_ai", None
    except Exception as exc:
        fallback_reason = f"PydanticAI diagnostics path failed for {config.provider}:{config.model}; falling back to legacy provider call. {exc}"
        raw_text = call_provider_legacy(config, system_prompt, context)
        return raw_text, "legacy", fallback_reason


def extract_json_object(raw_text: str) -> dict[str, Any]:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(candidate[start:end + 1])


def sanitize_diagnostics(candidate: Any, fallback: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(candidate, list):
        return fallback
    sanitized: list[dict[str, str]] = []
    for item in candidate:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level", "warn")).lower()
        if level not in {"good", "warn"}:
            level = "warn"
        title = str(item.get("title", "")).strip()
        message = str(item.get("message", "")).strip()
        next_step = str(item.get("next_step", "")).strip()
        implementation_diagnosis = str(item.get("implementation_diagnosis", "")).strip()
        suggested_fix = str(item.get("suggested_fix", "")).strip()
        code_refs = item.get("code_refs") or []
        evidence_keys = item.get("evidence_keys") or []
        if not title or not message or not next_step:
            continue
        payload = {
            "level": level,
            "title": title,
            "message": message,
            "next_step": next_step,
            "evidence_keys": [str(key) for key in evidence_keys[:12]],
        }
        if implementation_diagnosis:
            payload["implementation_diagnosis"] = implementation_diagnosis
        if suggested_fix:
            payload["suggested_fix"] = suggested_fix
        if code_refs:
            payload["code_refs"] = [str(ref) for ref in code_refs[:8]]
        sanitized.append(payload)
    return sanitized[:5] if sanitized else fallback


def build_heuristic_summary_line(summary: dict[str, Any]) -> str:
    frames_processed = int(summary.get("frames_processed") or 0)
    unique_player_track_ids = int(summary.get("unique_player_track_ids") or 0)
    raw_unique_player_track_ids = int(summary.get("raw_unique_player_track_ids") or unique_player_track_ids)
    field_registered_ratio = float(summary.get("field_registered_ratio") or 0.0)
    average_ball_detections_per_frame = float(summary.get("average_ball_detections_per_frame") or 0.0)
    experiment_count = len(summary.get("experiments") or [])

    parts = [
        f"{frames_processed}-frame run",
        f"{unique_player_track_ids} player IDs",
        f"{field_registered_ratio * 100:.1f}% field registration",
        f"{average_ball_detections_per_frame:.2f} ball detections/frame",
    ]
    if raw_unique_player_track_ids > unique_player_track_ids:
        parts.insert(2, f"{raw_unique_player_track_ids}->{unique_player_track_ids} raw-to-stitched IDs")
    if experiment_count:
        parts.append(f"{experiment_count} experiment card{'s' if experiment_count != 1 else ''}")
    return ", ".join(parts) + "."


def build_summary_heuristic_diagnostics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    frames_processed = int(summary.get("frames_processed") or 0)
    player_rows = int(summary.get("player_rows") or 0)
    ball_rows = int(summary.get("ball_rows") or 0)
    unique_player_track_ids = int(summary.get("unique_player_track_ids") or 0)
    raw_unique_player_track_ids = int(summary.get("raw_unique_player_track_ids") or unique_player_track_ids)
    avg_player = float(summary.get("average_player_detections_per_frame") or 0.0)
    avg_ball = float(summary.get("average_ball_detections_per_frame") or 0.0)
    average_track_length = float(summary.get("average_track_length") or 0.0)
    projected_player_points = int(summary.get("projected_player_points") or 0)
    field_registered_ratio = float(summary.get("field_registered_ratio") or 0.0)
    calibration_attempts = int(summary.get("field_calibration_refresh_attempts") or 0)
    calibration_successes = int(summary.get("field_calibration_refresh_successes") or 0)
    visible_keypoints = float(summary.get("average_visible_pitch_keypoints") or 0.0)
    home_tracks = int(summary.get("home_tracks") or 0)
    away_tracks = int(summary.get("away_tracks") or 0)
    team_cluster_distance = float(summary.get("team_cluster_distance") or 0.0)
    goal_events_count = int(summary.get("goal_events_count") or 0)
    experiments = list(summary.get("experiments") or [])

    diagnostics: list[dict[str, Any]] = []

    if player_rows == 0:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Detector produced no objects",
                "message": (
                    f"Across {frames_processed} frames, the run emitted 0 player rows and 0 ball rows, with average player detections/frame {avg_player:.2f} "
                    f"and average ball detections/frame {avg_ball:.2f}. That means the pipeline failed before tracking, team labeling, and projection."
                ),
                "next_step": (
                    "Inspect the active detector checkpoint, class mapping, and confidence thresholds first. "
                    "If this was a custom detector, verify the player and ball class ids used by the runtime match the checkpoint output order before touching tracker or calibration settings."
                ),
                "implementation_diagnosis": (
                    "The most likely code-level failure is the class-id resolution path for custom detector checkpoints. "
                    "`resolve_detector_spec` falls back to hardcoded class ids, and `detect_players_for_frame` / the ball branch pass those ids directly into the YOLO `classes=[...]` filter."
                ),
                "suggested_fix": (
                    "Resolve class ids from checkpoint or dataset metadata instead of defaulting to 0/1/2. "
                    "If metadata is unavailable, fail fast and print discovered class names rather than silently filtering on guessed ids."
                ),
                "code_refs": [
                    "backend/app/wide_angle.py::resolve_detector_spec",
                    "backend/app/wide_angle.py::detect_players_for_frame",
                    "backend/app/wide_angle.py::analyze_video",
                ],
                "evidence_keys": ["player_rows", "ball_rows", "average_player_detections_per_frame", "average_ball_detections_per_frame"],
            }
        )
    elif unique_player_track_ids / max(frames_processed, 1) > 0.1:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Player tracking is fragmented",
                "message": (
                    f"{unique_player_track_ids} player IDs were produced across {frames_processed} frames, with average track length {average_track_length:.1f}. "
                    f"That is high identity churn for football continuity."
                ),
                "next_step": (
                    "Inspect the overlay around dense phases and cutaways. If using stitched identity, compare raw and canonical ID counts before changing detector thresholds."
                ),
                "evidence_keys": ["unique_player_track_ids", "frames_processed", "average_track_length"],
            }
        )
    else:
        diagnostics.append(
            {
                "level": "good",
                "title": "Player tracking is stable enough",
                "message": (
                    f"{unique_player_track_ids} player IDs across {frames_processed} frames with average track length {average_track_length:.1f} "
                    f"looks acceptable for broad tactical review."
                ),
                "next_step": "Spot-check identity continuity through camera motion before using the run for player-level conclusions.",
                "evidence_keys": ["unique_player_track_ids", "frames_processed", "average_track_length"],
            }
        )

    if projected_player_points == 0 or field_registered_ratio <= 0.05:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Calibration produced no usable projection",
                "message": (
                    f"Calibration refreshes succeeded {calibration_successes}/{calibration_attempts} times with average visible keypoints {visible_keypoints:.1f}, "
                    f"but projected player points are {projected_player_points} and registered ratio is {field_registered_ratio:.3f}."
                ),
                "next_step": (
                    "Do not trust the minimap. Verify the homography acceptance path, visible keypoint quality, and anchor projection before debugging tracker behavior."
                ),
                "implementation_diagnosis": (
                    "The projection path is downstream of both the homography gate and anchor generation. "
                    "If projected points stay at zero, the likely failing code is either the `candidate_is_usable` acceptance condition or an upstream empty-detection path that leaves nothing to project."
                ),
                "suggested_fix": (
                    "Log the calibration rejection reason per frame and distinguish `no detections to project` from `homography rejected` in the summary."
                ),
                "code_refs": [
                    "backend/app/wide_angle.py::detect_pitch_homography",
                    "backend/app/wide_angle.py::analyze_video",
                ],
                "evidence_keys": ["field_calibration_refresh_successes", "field_calibration_refresh_attempts", "average_visible_pitch_keypoints", "projected_player_points", "field_registered_ratio"],
            }
        )
    elif calibration_attempts > 0 and (calibration_successes / calibration_attempts) < 0.7:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Calibration refresh is unstable",
                "message": (
                    f"Only {calibration_successes}/{calibration_attempts} calibration refreshes succeeded, with average visible pitch keypoints {visible_keypoints:.1f} "
                    f"and field registered ratio {field_registered_ratio:.3f}."
                ),
                "next_step": (
                    "Review frames around rejected refreshes and confirm the field keypoint model still sees enough pitch structure before relying on minimap movement."
                ),
                "evidence_keys": ["field_calibration_refresh_successes", "field_calibration_refresh_attempts", "average_visible_pitch_keypoints", "field_registered_ratio"],
            }
        )
    else:
        diagnostics.append(
            {
                "level": "good",
                "title": "Field mapping mostly present",
                "message": (
                    f"Calibration refreshes succeeded {calibration_successes}/{calibration_attempts} times and field registered ratio is {field_registered_ratio:.3f}."
                ),
                "next_step": "Use the minimap for review, but still spot-check late-match sequences and camera transitions for drift.",
                "evidence_keys": ["field_calibration_refresh_successes", "field_calibration_refresh_attempts", "field_registered_ratio"],
            }
        )

    if ball_rows == 0 or avg_ball < 0.15:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Ball signal is sparse",
                "message": f"Ball detections/frame is {avg_ball:.2f} with {ball_rows} total ball rows, so ball continuity is weak.",
                "next_step": "Avoid ball-led interpretation until detector coverage improves; inspect detector confidence and missed ball phases first.",
                "implementation_diagnosis": (
                    "The ball stream uses the same football detector class map as the player stream. "
                    "When both player and ball rows collapse together, the shared detector class filter is a stronger suspect than the ball tracker itself."
                ),
                "suggested_fix": (
                    "Verify the ball class id used in the YOLO `classes=[...]` filter and stop assuming the fallback custom-detector mapping is correct."
                ),
                "code_refs": [
                    "backend/app/wide_angle.py::resolve_detector_spec",
                    "backend/app/wide_angle.py::analyze_video",
                ],
                "evidence_keys": ["average_ball_detections_per_frame", "ball_rows"],
            }
        )
    elif home_tracks > 0 and away_tracks > 0 and team_cluster_distance > 0.08:
        diagnostics.append(
            {
                "level": "good",
                "title": "Team split is usable",
                "message": f"Home/away track counts are {home_tracks}/{away_tracks} with cluster distance {team_cluster_distance:.3f}.",
                "next_step": "Validate borderline team assignments on long tracks before using team-level aggregates.",
                "evidence_keys": ["home_tracks", "away_tracks", "team_cluster_distance"],
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Team split is weak",
                "message": f"Home/away track counts are {home_tracks}/{away_tracks} with cluster distance {team_cluster_distance:.3f}.",
                "next_step": "Inspect jersey crops and track-level vote stability before using home/away tactical summaries.",
                "evidence_keys": ["home_tracks", "away_tracks", "team_cluster_distance"],
            }
        )

    diagnostics.append(
        {
            "level": "warn" if goal_events_count == 0 or not experiments else "good",
            "title": "Experiment context"
            if goal_events_count > 0 and experiments
            else "Experiment is not goal-ready",
            "message": (
                f"Goal events count is {goal_events_count} and experiment cards attached: {len(experiments)}."
            ),
            "next_step": (
                "Use the experiment output only as exploratory context unless the run is goal-aligned and the metrics actually separate the target window."
            ),
            "evidence_keys": ["goal_events_count"],
        }
    )
    return diagnostics[:5]


def generate_run_diagnostics(
    summary: dict[str, Any],
    heuristic_diagnostics: list[dict[str, str]],
    outputs_dir: Path,
    job_id: str,
    job_manager: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fallback_diagnostics = build_summary_heuristic_diagnostics(summary)
    config = resolve_provider_config()
    artifact_path = outputs_dir / "diagnostics_ai.json"

    artifact: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "disabled",
        "orchestrator": "disabled",
        "provider": None,
        "model": None,
        "summary_line": build_heuristic_summary_line(summary),
        "error": "",
        "raw_text": "",
        "prompt_context": {
            "recent_logs": [],
            "code_context": [],
            "budget": {
                "max_output_tokens": None,
                "context_json_chars": 0,
                "code_slice_count": 0,
                "recent_log_count": 0,
            },
        },
        "diagnostics": fallback_diagnostics,
    }

    if config is None:
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return fallback_diagnostics, artifact

    if job_manager is not None:
        job_manager.log(job_id, f"Generating AI diagnostics via {config.provider}:{config.model}")

    recent_logs = load_recent_logs(summary, job_id, job_manager)
    code_context = build_code_context(summary, fallback_diagnostics)
    system_prompt = build_system_prompt()
    context = build_run_context(summary, fallback_diagnostics, recent_logs, code_context)
    context = fit_prompt_context_budget(context)
    prompt_context = {
        "recent_logs": context.get("recent_logs", []),
        "code_context": context.get("code_context", []),
        "budget": {
            "max_output_tokens": config.max_output_tokens,
            "context_json_chars": len(json.dumps(context)),
            "code_slice_count": len(context.get("code_context", [])),
            "recent_log_count": len(context.get("recent_logs", [])),
        },
    }

    try:
        raw_text, orchestrator, fallback_note = call_provider(config, system_prompt, context)
        if fallback_note and job_manager is not None:
            job_manager.log(job_id, fallback_note)
        parsed = extract_json_object(raw_text)
        diagnostics = sanitize_diagnostics(parsed.get("diagnostics"), fallback_diagnostics)
        summary_line = str(parsed.get("summary_line", "")).strip()
        artifact = {
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "completed",
            "orchestrator": orchestrator,
            "provider": config.provider,
            "model": config.model,
            "summary_line": summary_line,
            "error": "",
            "raw_text": raw_text,
            "prompt_context": prompt_context,
            "diagnostics": diagnostics,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return diagnostics, artifact
    except Exception as exc:
        artifact = {
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "failed",
            "orchestrator": "pydantic_ai" if "PydanticAI" in str(exc) else "legacy",
            "provider": config.provider,
            "model": config.model,
            "summary_line": build_heuristic_summary_line(summary),
            "error": str(exc),
            "raw_text": "",
            "prompt_context": prompt_context,
            "diagnostics": fallback_diagnostics,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        if job_manager is not None:
            job_manager.log(job_id, f"AI diagnostics failed; using heuristic fallback. {exc}")
        return fallback_diagnostics, artifact
