from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request


PROMPT_VERSION = "run-diagnostics-v2"
DEFAULT_TIMEOUT_SECONDS = 45.0

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    api_key: str
    timeout_seconds: float
    extra_headers: dict[str, str]


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def resolve_provider_config() -> ProviderConfig | None:
    provider_pref = _env("AI_DIAGNOSTICS_PROVIDER", "auto").lower()
    timeout_seconds = float(_env("AI_DIAGNOSTICS_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)) or DEFAULT_TIMEOUT_SECONDS)
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
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers={},
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
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers=headers,
        )

    def build_anthropic() -> ProviderConfig | None:
        api_key = _env("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return ProviderConfig(
            provider="anthropic",
            model=shared_model or _env("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
            endpoint="https://api.anthropic.com/v1/messages",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers={"anthropic-version": _env("ANTHROPIC_VERSION") or "2023-06-01"},
        )

    def build_local() -> ProviderConfig | None:
        if not local_base_url:
            return None
        normalized = local_base_url.rstrip("/")
        endpoint = normalized if normalized.endswith("/chat/completions") else f"{normalized}/chat/completions"
        return ProviderConfig(
            provider="local",
            model=shared_model or _env("LOCAL_LLM_MODEL") or "gpt-oss-20b",
            endpoint=endpoint,
            api_key=local_api_key or "local",
            timeout_seconds=timeout_seconds,
            extra_headers={},
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


def build_system_prompt() -> str:
    return (
        "You are generating UI diagnostics for one completed football video analysis run. "
        "Analyze only the supplied run JSON. Do not invent observations. "
        "Do not copy generic QA language, marketing language, or coaching cliches. "
        "Avoid words such as strong, solid, reliable, effective, meaningful, usable, workable, supports, provides context, and believable unless they are numerically justified. "
        "Prefer direct operator language tied to actual metrics. "
        "Call out weaknesses or uncertainty when the metrics are mixed. "
        "If experimental signals are missing, sparse, or not goal-aligned, say so directly. "
        "Do not mention that you are an AI. "
        "Return valid JSON only with this exact schema: "
        '{"summary_line":"string","diagnostics":[{"level":"good|warn","title":"string","message":"string","next_step":"string","evidence_keys":["metric_key"]}]}. '
        "Produce 3 to 5 diagnostics. "
        "Use short UI-ready titles. "
        "Each message must cite actual numeric evidence when possible. "
        "Each next_step must be a concrete operator action. "
        "Do not wrap the JSON in markdown fences."
    )


def build_run_context(summary: dict[str, Any], heuristic_diagnostics: list[dict[str, str]]) -> dict[str, Any]:
    experiments = summary.get("experiments") or []
    top_tracks = summary.get("top_tracks") or []
    frames_processed = float(summary.get("frames_processed") or 0.0)
    unique_player_track_ids = float(summary.get("unique_player_track_ids") or 0.0)
    refresh_attempts = float(summary.get("field_calibration_refresh_attempts") or 0.0)
    refresh_successes = float(summary.get("field_calibration_refresh_successes") or 0.0)
    goal_events_count = float(summary.get("goal_events_count") or 0.0)
    return {
        "prompt_version": PROMPT_VERSION,
        "input_video": Path(str(summary.get("input_video", ""))).name,
        "models": {
            "detector": summary.get("player_model"),
            "ball": summary.get("ball_model"),
            "field_calibration": summary.get("field_calibration_model"),
            "player_tracker_mode": summary.get("player_tracker_mode"),
            "player_tracker_backend": summary.get("player_tracker_backend"),
            "device": summary.get("device"),
            "field_calibration_device": summary.get("field_calibration_device"),
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
            "average_visible_pitch_keypoints": summary.get("average_visible_pitch_keypoints"),
            "last_good_calibration_frame": summary.get("last_good_calibration_frame"),
            "goal_events_count": summary.get("goal_events_count"),
            "team_cluster_distance": summary.get("team_cluster_distance"),
            "jersey_crops_used": summary.get("jersey_crops_used"),
        },
        "derived_metrics": {
            "player_track_churn_ratio": round(unique_player_track_ids / frames_processed, 6) if frames_processed > 0 else None,
            "field_calibration_success_rate": round(refresh_successes / refresh_attempts, 6) if refresh_attempts > 0 else None,
            "goal_aligned_experiment": bool(goal_events_count > 0),
            "has_experiments": bool(experiments),
        },
        "heuristic_diagnostics": heuristic_diagnostics[:5],
        "experiments": experiments,
        "top_tracks": top_tracks[:8],
    }


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
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


def call_provider(config: ProviderConfig, system_prompt: str, context: dict[str, Any]) -> str:
    user_payload = json.dumps(context, indent=2)
    if config.provider in {"openai", "openrouter", "local"}:
        payload = {
            "model": config.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "response_format": {"type": "json_object"},
        }
        if config.provider == "openai" and config.model.startswith("gpt-5"):
            payload["max_completion_tokens"] = 900
        else:
            payload["max_tokens"] = 900
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
                "max_tokens": 900,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_payload},
                ],
            },
            timeout_seconds=config.timeout_seconds,
        )
        return _extract_text_from_anthropic(response_payload)

    raise RuntimeError(f"Unsupported diagnostics provider: {config.provider}")


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
        evidence_keys = item.get("evidence_keys") or []
        if not title or not message or not next_step:
            continue
        sanitized.append(
            {
                "level": level,
                "title": title[:96],
                "message": message[:320],
                "next_step": next_step[:220],
                "evidence_keys": [str(key) for key in evidence_keys[:8]],
            }
        )
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


def generate_run_diagnostics(
    summary: dict[str, Any],
    heuristic_diagnostics: list[dict[str, str]],
    outputs_dir: Path,
    job_id: str,
    job_manager: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = resolve_provider_config()
    artifact_path = outputs_dir / "diagnostics_ai.json"

    artifact: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "disabled",
        "provider": None,
        "model": None,
        "summary_line": build_heuristic_summary_line(summary),
        "error": "",
        "raw_text": "",
        "diagnostics": heuristic_diagnostics,
    }

    if config is None:
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return heuristic_diagnostics, artifact

    if job_manager is not None:
        job_manager.log(job_id, f"Generating AI diagnostics via {config.provider}:{config.model}")

    system_prompt = build_system_prompt()
    context = build_run_context(summary, heuristic_diagnostics)

    try:
        raw_text = call_provider(config, system_prompt, context)
        parsed = extract_json_object(raw_text)
        diagnostics = sanitize_diagnostics(parsed.get("diagnostics"), heuristic_diagnostics)
        summary_line = str(parsed.get("summary_line", "")).strip()
        artifact = {
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "completed",
            "provider": config.provider,
            "model": config.model,
            "summary_line": summary_line[:240],
            "error": "",
            "raw_text": raw_text,
            "diagnostics": diagnostics,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return diagnostics, artifact
    except Exception as exc:
        artifact = {
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "failed",
            "provider": config.provider,
            "model": config.model,
            "summary_line": build_heuristic_summary_line(summary),
            "error": str(exc),
            "raw_text": "",
            "diagnostics": heuristic_diagnostics,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        if job_manager is not None:
            job_manager.log(job_id, f"AI diagnostics failed; using heuristic fallback. {exc}")
        return heuristic_diagnostics, artifact
