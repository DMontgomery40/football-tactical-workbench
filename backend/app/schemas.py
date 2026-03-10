from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_SCHEMA_VERSION = "2026.03.10"
SUMMARY_SCHEMA_VERSION = CONTRACT_SCHEMA_VERSION
JOB_STATE_SCHEMA_VERSION = CONTRACT_SCHEMA_VERSION
SOURCE_SCHEMA_VERSION = CONTRACT_SCHEMA_VERSION


class BoundaryModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class LearnCard(BoundaryModel):
    title: str
    what_it_does: str
    what_breaks: str
    what_to_try_next: str


class DiagnosticItem(BoundaryModel):
    level: str
    title: str
    message: str = ""
    next_step: str = ""
    implementation_diagnosis: str = ""
    suggested_fix: str = ""
    code_refs: list[str] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)


class ExperimentMetric(BoundaryModel):
    label: str
    value: Any = None
    hint: str = ""


class ExperimentCard(BoundaryModel):
    id: str = ""
    status: str = ""
    title: str
    summary: str = ""
    interpretation: str = ""
    metrics: list[ExperimentMetric] = Field(default_factory=list)


class TopTrackRow(BoundaryModel):
    track_id: int
    team_label: str = "unassigned"
    team_vote_ratio: float = 0.0
    frames: int = 0
    first_frame: int = 0
    last_frame: int = 0
    average_confidence: float = 0.0
    average_bbox_area: float = 0.0
    projected_points: int = 0


class RunSummary(BoundaryModel):
    summary_version: str = SUMMARY_SCHEMA_VERSION
    job_id: str = ""
    run_dir: str = ""
    input_video: str = ""
    overlay_video: str | None = None
    detections_csv: str | None = None
    track_summary_csv: str | None = None
    projection_csv: str | None = None
    calibration_debug_csv: str | None = None
    entropy_timeseries_csv: str | None = None
    goal_events_csv: str | None = None
    summary_json: str | None = None
    all_outputs_zip: str | None = None
    device: str = ""
    field_calibration_device: str = ""
    player_model: str = ""
    requested_player_tracker_mode: str | None = None
    player_tracker_mode: str = ""
    resolved_player_tracker_mode: str = ""
    player_tracker_runtime: str | None = None
    player_tracker_backend: str | None = None
    player_tracker_embedding_error: str | None = None
    player_tracker_stitching_enabled: bool | None = None
    detector_class_names_source: str = ""
    player_detector_class_ids: list[int] = Field(default_factory=list)
    ball_detector_class_ids: list[int] = Field(default_factory=list)
    referee_detector_class_ids: list[int] = Field(default_factory=list)
    ball_model: str = ""
    ball_tracker_mode: str = ""
    field_calibration_model: str = ""
    include_ball: bool = True
    player_conf: float = 0.0
    ball_conf: float = 0.0
    iou: float = 0.0
    frames_processed: int = 0
    fps: float = 0.0
    player_rows: int = 0
    ball_rows: int = 0
    unique_player_track_ids: int = 0
    raw_unique_player_track_ids: int | None = None
    unique_ball_track_ids: int = 0
    home_tracks: int = 0
    away_tracks: int = 0
    unassigned_tracks: int = 0
    average_player_detections_per_frame: float = 0.0
    average_ball_detections_per_frame: float = 0.0
    longest_track_length: int = 0
    average_track_length: float = 0.0
    raw_longest_track_length: int = 0
    raw_average_track_length: float = 0.0
    player_track_churn_ratio: float = 0.0
    raw_player_track_churn_ratio: float = 0.0
    tracklet_merges_applied: int | None = None
    stitched_track_id_reduction: float | None = None
    identity_embedding_updates: int | None = None
    identity_embedding_interval_frames: int | None = None
    projected_player_points: int = 0
    projected_ball_points: int = 0
    projected_player_points_fresh: int = 0
    projected_player_points_stale: int = 0
    projected_ball_points_fresh: int = 0
    projected_ball_points_stale: int = 0
    player_rows_while_calibration_fresh: int = 0
    player_rows_while_calibration_stale: int = 0
    field_registered_frames: int = 0
    field_registered_ratio: float = 0.0
    frames_with_field_homography: int = 0
    frames_with_usable_homography: int = 0
    frames_with_nonstale_homography: int = 0
    frames_with_stale_homography: int = 0
    frames_projection_blocked_by_stale: int = 0
    frames_projected_with_last_known_homography: int = 0
    frames_with_player_anchors: int = 0
    frames_with_projected_points: int = 0
    frames_with_homography_but_no_player_anchors: int = 0
    homography_enabled: bool = False
    field_calibration_refresh_frames: int = 0
    field_calibration_refresh_attempts: int = 0
    field_calibration_refresh_successes: int = 0
    field_calibration_success_rate: float = 0.0
    field_calibration_refresh_rejections: int = 0
    field_keypoint_confidence_threshold: float = 0.0
    field_calibration_min_visible_keypoints: int = 0
    field_calibration_stale_recovery_min_visible_keypoints: int = 0
    field_calibration_stale_recovery_attempts: int = 0
    field_calibration_stale_recovery_successes: int = 0
    field_calibration_stale_recovery_rejections: int = 0
    field_calibration_rejections_no_candidate: int = 0
    field_calibration_rejections_low_visible_count: int = 0
    field_calibration_rejections_low_visible_keypoints: int = 0
    field_calibration_rejections_low_inliers: int = 0
    field_calibration_rejections_high_reprojection_error: int = 0
    field_calibration_rejections_high_temporal_drift: int = 0
    field_calibration_rejections_invalid_candidate: int = 0
    field_calibration_primary_rejections_no_candidate: int = 0
    field_calibration_primary_rejections_low_visible_count: int = 0
    field_calibration_primary_rejections_low_visible_keypoints: int = 0
    field_calibration_primary_rejections_low_inliers: int = 0
    field_calibration_primary_rejections_high_reprojection_error: int = 0
    field_calibration_primary_rejections_high_temporal_drift: int = 0
    field_calibration_primary_rejections_invalid_candidate: int = 0
    average_visible_pitch_keypoints: float = 0.0
    last_good_calibration_frame: int = -1
    detector_debug_sample_frames: int = 0
    raw_detector_boxes_sampled: int = 0
    raw_detector_class_histogram_sample: dict[str, int] = Field(default_factory=dict)
    goal_events_count: int = 0
    goal_label_source: str | None = None
    team_cluster_distance: float = 0.0
    jersey_crops_used: int = 0
    diagnostics_source: str = "heuristic"
    diagnostics_orchestrator: str | None = None
    diagnostics_provider: str | None = None
    diagnostics_model: str | None = None
    diagnostics_status: str = "unknown"
    diagnostics_summary_line: str = ""
    diagnostics_error: str = ""
    diagnostics_json: str | None = None
    diagnostics_prompt_context: dict[str, Any] | None = None
    diagnostics_prompt_version: str | None = None
    diagnostics_current_prompt_version: str | None = None
    diagnostics_stale: bool = False
    diagnostics_stale_reason: str = ""
    legacy_summary: bool = False
    created_at: str = ""
    learn_cards: list[LearnCard] = Field(default_factory=list)
    experiments: list[ExperimentCard] = Field(default_factory=list)
    top_tracks: list[TopTrackRow] = Field(default_factory=list)
    diagnostics: list[DiagnosticItem] = Field(default_factory=list)
    heuristic_diagnostics: list[DiagnosticItem] = Field(default_factory=list)


class JobStateResponse(BoundaryModel):
    job_state_version: str = JOB_STATE_SCHEMA_VERSION
    job_id: str
    run_id: str = ""
    status: str
    created_at: str
    progress: float
    logs: list[str] = Field(default_factory=list)
    run_dir: str = ""
    summary: RunSummary | dict[str, Any] | None = None
    error: str | None = None
    persisted: bool | None = None
    experiment_batch: str | None = None


class ActiveExperimentSummary(BoundaryModel):
    batch_name: str = ""
    games_requested: int = 0
    halves_total: int = 0
    halves_processed: int = 0
    current_game: str = ""
    current_game_index: int = 0
    current_half_tag: str = ""
    current_half_file: str = ""
    current_source_video_path: str = ""
    current_label_path: str = ""
    files: list[str] = Field(default_factory=list)


class ActiveExperimentResponse(BoundaryModel):
    job_state_version: str = JOB_STATE_SCHEMA_VERSION
    job_id: str
    status: str
    created_at: str
    progress: float
    logs: list[str] = Field(default_factory=list)
    run_dir: str = ""
    summary: ActiveExperimentSummary
    error: str | None = None


class AnalyzeAcceptedResponse(BoundaryModel):
    job_id: str
    run_id: str
    run_dir: str


class VideoMetadata(BoundaryModel):
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    size_mb: float


class SourceResponse(VideoMetadata):
    source_state_version: str = SOURCE_SCHEMA_VERSION
    source_id: str
    path: str
    display_name: str
    created_at: str
    uploaded: bool
    video_url: str


class ScanFolderEntry(BoundaryModel):
    name: str
    path: str
    size_mb: float | None = None


class FolderScanResponse(BoundaryModel):
    folder: str
    videos: list[ScanFolderEntry] = Field(default_factory=list)
    annotations: list[ScanFolderEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RuntimeProfileResponse(BoundaryModel):
    backend: str = ""
    backend_label: str = ""
    backend_version: str | None = None
    host_platform: str = ""
    host_arch: str = ""
    preferred_device: str = ""
    available_devices: list[str] = Field(default_factory=list)
    field_calibration_device_policy: str = ""
    detector_export_formats: list[str] = Field(default_factory=list)
    planned_backends: list[dict[str, str]] = Field(default_factory=list)
    runtime_notes: list[str] = Field(default_factory=list)
    license_notes: list[str] = Field(default_factory=list)


class ConfigResponse(BoundaryModel):
    contract_version: str = CONTRACT_SCHEMA_VERSION
    detector_models: list[str] = Field(default_factory=list)
    player_models: list[str] = Field(default_factory=list)
    tracker: str
    player_tracker_modes: list[str] = Field(default_factory=list)
    default_player_tracker_mode: str
    learn_cards: list[LearnCard] = Field(default_factory=list)
    help_catalog: Any = None
    field_calibration_refresh_frames: int
    field_calibration_mode: str
    soccernet_dataset_dir: str
    soccernet_video_files: list[str] = Field(default_factory=list)
    soccernet_label_files: list[str] = Field(default_factory=list)
    diagnostics_provider: str | None = None
    diagnostics_model: str | None = None
    training_available: bool
    training_error: str | None = None
    active_detector: str
    active_detector_label: str
    active_detector_is_custom: bool
    runtime_profile: RuntimeProfileResponse | dict[str, Any] = Field(default_factory=dict)


class SoccerNetConfigResponse(BoundaryModel):
    dataset_dir: str
    splits: list[str] = Field(default_factory=list)
    split_counts: dict[str, int] = Field(default_factory=dict)
    video_files: list[str] = Field(default_factory=list)
    label_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SoccerNetGamesResponse(BoundaryModel):
    split: str
    count: int
    games: list[str] = Field(default_factory=list)
