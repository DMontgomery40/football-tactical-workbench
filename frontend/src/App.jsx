import { useEffect, useMemo, useRef, useState } from 'react';

import { buildHelpIndex, FieldLabel, HelpPopover, SectionTitleWithHelp } from './helpUi';
import TrainingStudio from './TrainingStudio';

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1', '[::1]']);

function isLoopbackHost(hostname) {
  return LOOPBACK_HOSTS.has(String(hostname || '').toLowerCase());
}

function resolveApiBase() {
  if (typeof window === 'undefined') {
    return import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8431';
  }

  const browserProtocol = window.location?.protocol || 'http:';
  const browserHost = window.location?.hostname || '127.0.0.1';
  const fallbackBase = `${browserProtocol}//${browserHost}:8431`;
  const configuredBase = String(import.meta.env.VITE_API_BASE_URL || '').trim();

  if (!configuredBase) {
    return fallbackBase;
  }

  try {
    const url = new URL(configuredBase);
    if (isLoopbackHost(url.hostname) && !isLoopbackHost(browserHost)) {
      url.protocol = browserProtocol;
      url.hostname = browserHost;
    }
    return url.toString().replace(/\/$/, '');
  } catch {
    return fallbackBase;
  }
}

const API_BASE = resolveApiBase();
const SOCCERNET_AUTO_EXPAND_LIMIT = 8;

const defaultForm = {
  localVideoPath: '',
  labelPath: '',
  folderPath: '',
  detectorModel: 'soccana',
  trackerMode: 'hybrid_reid',
  includeBall: true,
  playerConf: '0.25',
  ballConf: '0.20',
  iou: '0.50',
};

const DEFAULT_SOCCERNET_FILES = ['1_720p.mkv', '2_720p.mkv', 'Labels-v2.json'];

const STORAGE_KEYS = {
  appSpace: 'fpw.appSpace',
  themeMode: 'fpw.themeMode',
  form: 'fpw.form',
  soccerNetSplit: 'fpw.soccerNetSplit',
  soccerNetQuery: 'fpw.soccerNetQuery',
  soccerNetFiles: 'fpw.soccerNetFiles',
  folderPath: 'fpw.folderPath',
  localVideoPath: 'fpw.localVideoPath',
  analysisSidebarWidth: 'fpw.analysisSidebarWidth',
};

const WORKSPACE_MODES = [
  { id: 'input', label: 'Input' },
  { id: 'live', label: 'Live Preview' },
  { id: 'job', label: 'Active Job' },
  { id: 'review', label: 'Run Review' },
];

const REVIEW_PANELS = [
  { id: 'overview', label: 'Overview' },
  { id: 'tracks', label: 'Tracks' },
  { id: 'files', label: 'Files' },
];

const REVIEW_METRIC_HELP_IDS = {
  'Tracker mode': 'review.metric.tracker_mode',
  'Raw player IDs': 'review.metric.raw_player_ids',
  'Player track IDs': 'review.metric.player_track_ids',
  'Tracklet merges': 'review.metric.tracklet_merges',
  'Calibration success': 'review.metric.calibration_success',
  'Stale recovery': 'review.metric.stale_recovery',
  'Calib gate rejects': 'review.metric.calib_gate_rejects',
  'Projection stages': 'review.metric.projection_stages',
  'Projected fresh/stale': 'review.metric.projected_fresh_stale',
};

const BACKEND_ACTIVITY_WINDOW_MS = 20000;
const BACKEND_FAILURE_WINDOW_MS = 20000;
const ACTIVE_JOB_STATUSES = new Set(['queued', 'running', 'paused', 'stopping']);
const DEFAULT_ANALYSIS_SIDEBAR_WIDTH = 356;
const MIN_ANALYSIS_SIDEBAR_WIDTH = 280;
const MAX_ANALYSIS_SIDEBAR_WIDTH = 520;
const ANALYSIS_MAIN_MIN_WIDTH = 620;
const SIDEBAR_RESIZER_WIDTH = 18;
const PITCH_LENGTH_CM = 10500;
const PITCH_WIDTH_CM = 6800;
const DEFAULT_TRAJECTORY_TRACK_COUNT = 4;
const TRAJECTORY_TRACK_DISPLAY_LIMIT = 12;
const TRAJECTORY_TRACK_COLORS = ['#1f5f92', '#c55a11', '#2f6f4f', '#8b1e3f', '#0f766e', '#8b5e34'];
const TRAJECTORY_WINDOW_SECONDS = 8;

function isActiveJobStatus(status) {
  return ACTIVE_JOB_STATUSES.has(String(status || ''));
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function readStoredJson(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function readStoredString(key, fallback = '') {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function basenameFromPath(value) {
  if (!value) return '';
  return String(value).split(/[\\/]+/).filter(Boolean).pop() || String(value);
}

function parseSoccerNetGame(raw) {
  if (!raw) return { match: raw, league: '', season: '', date: '', raw };
  const parts = raw.split('/').filter(Boolean);
  if (parts.length < 3) return { match: raw, league: '', season: '', date: '', raw };
  const league = parts[0].replace(/_/g, ' ');
  const season = parts[1];
  const matchSegment = parts.slice(2).join('/');
  const m = matchSegment.match(/^(\d{4}-\d{2}-\d{2})\s*-\s*\d{2}-\d{2}\s+(.+)$/);
  if (m) {
    return { match: m[2], league, season, date: m[1], raw };
  }
  return { match: matchSegment, league, season, date: '', raw };
}

function friendlyProviderName(value) {
  const provider = String(value || '').toLowerCase();
  if (provider === 'openai') return 'OpenAI';
  if (provider === 'openrouter') return 'OpenRouter';
  if (provider === 'anthropic') return 'Anthropic';
  if (provider === 'local') return 'Local';
  return value || 'provider';
}

function formatPercent(value, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '0%';
  return `${(numeric * 100).toFixed(digits)}%`;
}

function formatCalibrationRejectionSummary(summary) {
  if (!summary) return 'n/a';
  const parts = [
    `no cand ${summary.field_calibration_rejections_no_candidate || 0}`,
    `vis ${summary.field_calibration_rejections_low_visible_keypoints || 0}`,
    `inliers ${summary.field_calibration_rejections_low_inliers || 0}`,
    `reproj ${summary.field_calibration_rejections_high_reprojection_error || 0}`,
    `drift ${summary.field_calibration_rejections_high_temporal_drift || 0}`,
  ];
  if (summary.field_calibration_rejections_invalid_candidate) {
    parts.push(`invalid ${summary.field_calibration_rejections_invalid_candidate}`);
  }
  return parts.join(' · ');
}

function formatClassIds(classIds) {
  if (!Array.isArray(classIds) || classIds.length === 0) return 'none';
  return classIds.join(', ');
}

function parseProjectionCsv(text) {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return [];
  }

  const headers = lines[0].split(',').map((header) => header.trim());
  return lines.slice(1)
    .map((line) => {
      const values = line.split(',');
      const row = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
      const frameIndex = Number(row.frame_index);
      const trackId = Number(row.track_id);
      const fieldX = Number(row.field_x_cm);
      const fieldY = Number(row.field_y_cm);
      if (!Number.isFinite(frameIndex) || !Number.isFinite(trackId) || !Number.isFinite(fieldX) || !Number.isFinite(fieldY)) {
        return null;
      }
      return {
        frameIndex,
        rowType: String(row.row_type || ''),
        trackId,
        teamLabel: String(row.team_label || ''),
        fieldX,
        fieldY,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.frameIndex - b.frameIndex);
}

function fieldPointToSvgPoint(fieldX, fieldY) {
  const x = (fieldX / PITCH_LENGTH_CM) * 100;
  const y = (fieldY / PITCH_WIDTH_CM) * 64;
  return {
    x: Number.isFinite(x) ? Math.max(0, Math.min(100, x)) : 0,
    y: Number.isFinite(y) ? Math.max(0, Math.min(64, y)) : 0,
  };
}

function buildTrajectoryPath(points) {
  if (!points.length) return '';
  return points
    .map((point, index) => {
      const svgPoint = fieldPointToSvgPoint(point.fieldX, point.fieldY);
      return `${index === 0 ? 'M' : 'L'} ${svgPoint.x.toFixed(2)} ${svgPoint.y.toFixed(2)}`;
    })
    .join(' ');
}

function formatTeamLabel(value) {
  const label = String(value || '').trim();
  return label ? label : 'unassigned';
}

function formatSecondsLabel(value) {
  const totalSeconds = Math.max(0, Math.floor(Number(value) || 0));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function HeadlineDiagnosticCard({ item }) {
  return (
    <article className={`headline-diagnostic-card ${item.level === 'warn' ? 'warn' : 'good'}`}>
      <div className="micro-label">{item.level === 'warn' ? 'Needs attention' : 'Holding up'}</div>
      <div className="headline-diagnostic-title">{item.title}</div>
      {item.message ? <p>{item.message}</p> : null}
    </article>
  );
}

function MetricGroup({ title, items, helpIndex }) {
  return (
    <section className="card metric-group-card">
      <div className="metric-group-header">
        <div className="section-title">{title}</div>
      </div>
      <div className="metric-group-rows">
        {items.map((item) => {
          const helpEntry = item.helpId ? helpIndex.get(item.helpId) : null;
          return (
            <div key={item.label} className={`metric-row ${item.wide ? 'wide' : ''}`}>
              <div className="metric-label-row">
                <div className="metric-label">{item.label}</div>
                <HelpPopover entry={helpEntry} />
              </div>
              <div className="metric-value">{item.value}</div>
              {!helpEntry && item.hint ? <div className="metric-hint">{item.hint}</div> : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function DiagnosticCard({ item }) {
  const hasCodeDrilldown = item.implementation_diagnosis || item.suggested_fix || (item.code_refs || []).length;

  return (
    <div className={`card diagnostic ${item.level === 'warn' ? 'warn' : 'good'}`}>
      <div className="diagnostic-title">{item.title}</div>
      <p>{item.message}</p>
      <div className="diagnostic-next">Next action: {item.next_step}</div>
      {hasCodeDrilldown ? (
        <details className="diagnostic-drilldown">
          <summary>Code diagnosis</summary>
          {item.implementation_diagnosis ? (
            <div className="diagnostic-drilldown-block">
              <div className="micro-label">Why the code is failing</div>
              <div className="diagnostic-drilldown-text">{item.implementation_diagnosis}</div>
            </div>
          ) : null}
          {item.suggested_fix ? (
            <div className="diagnostic-drilldown-block">
              <div className="micro-label">Suggested change</div>
              <div className="diagnostic-drilldown-text">{item.suggested_fix}</div>
            </div>
          ) : null}
          {(item.code_refs || []).length ? (
            <div className="diagnostic-drilldown-block">
              <div className="micro-label">Code refs</div>
              <div className="diagnostic-code-refs">{item.code_refs.join('\n')}</div>
            </div>
          ) : null}
        </details>
      ) : null}
    </div>
  );
}

function PromptContextPanel({ summary }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [artifactState, setArtifactState] = useState({ loading: false, error: '', promptContext: null });

  const promptContext = summary?.diagnostics_prompt_context || artifactState.promptContext;

  useEffect(() => {
    if (!isExpanded || promptContext || artifactState.loading || !summary?.diagnostics_json) {
      return;
    }

    let cancelled = false;
    setArtifactState((current) => ({ ...current, loading: true, error: '' }));
    fetch(`${API_BASE}${summary.diagnostics_json}`)
      .then((response) => response.json().then((data) => ({ ok: response.ok, data })))
      .then(({ ok, data }) => {
        if (cancelled) return;
        if (!ok) {
          throw new Error(data?.detail || 'Could not load diagnostics artifact');
        }
        setArtifactState({ loading: false, error: '', promptContext: data?.prompt_context || null });
      })
      .catch((error) => {
        if (cancelled) return;
        setArtifactState({ loading: false, error: error.message || 'Could not load diagnostics artifact', promptContext: null });
      });

    return () => {
      cancelled = true;
    };
  }, [isExpanded, promptContext, artifactState.loading, summary?.diagnostics_json]);

  if (!summary?.diagnostics_json) {
    return null;
  }

  return (
    <details className="card prompt-context-card" onToggle={(event) => setIsExpanded(event.currentTarget.open)}>
      <summary className="section-title prompt-context-summary">Prompt Debug Context</summary>
      <div className="prompt-context-meta">
        This is the selected runtime context sent to the diagnostics model. It is collapsed by default so the main diagnostics view stays unchanged.
      </div>
      {artifactState.loading ? <div className="muted">Loading prompt context…</div> : null}
      {artifactState.error ? <div className="error-box">{artifactState.error}</div> : null}
      {promptContext ? (
        <>
          <div className="prompt-context-grid">
            <div><span className="micro-label">Context chars</span><div>{promptContext.budget?.context_json_chars ?? 'n/a'}</div></div>
            <div><span className="micro-label">Code slices</span><div>{promptContext.budget?.code_slice_count ?? 0}</div></div>
            <div><span className="micro-label">Recent logs</span><div>{promptContext.budget?.recent_log_count ?? 0}</div></div>
            <div><span className="micro-label">Max output tokens</span><div>{promptContext.budget?.max_output_tokens ?? 'n/a'}</div></div>
          </div>
          {(promptContext.recent_logs || []).length ? (
            <details className="prompt-context-subdetail">
              <summary>Recent Logs</summary>
              <pre className="prompt-context-code">{promptContext.recent_logs.join('\n')}</pre>
            </details>
          ) : null}
          {(promptContext.code_context || []).map((slice) => (
            <details key={slice.label} className="prompt-context-subdetail">
              <summary>{slice.label} · {slice.path}</summary>
              <div className="prompt-context-meta">{slice.reason}</div>
              <pre className="prompt-context-code">{slice.excerpt}</pre>
            </details>
          ))}
        </>
      ) : null}
    </details>
  );
}

function ExperimentCard({ item }) {
  return (
    <div className="card experiment-card">
      <div className="row-between">
        <div className="diagnostic-title">{item.title}</div>
        <div className="experiment-badge">{item.status || 'experimental'}</div>
      </div>
      <p>{item.summary}</p>
      <div className="experiment-metrics">
        {(item.metrics || []).map((metric) => (
          <div key={metric.label} className="experiment-metric">
            <div className="micro-label">{metric.label}</div>
            <div className="experiment-value">{metric.value}</div>
            {metric.hint ? <div className="stat-hint">{metric.hint}</div> : null}
          </div>
        ))}
      </div>
      {item.interpretation ? <div className="diagnostic-next">Interpretation: {item.interpretation}</div> : null}
    </div>
  );
}

function TrackTable({ tracks }) {
  if (!tracks || tracks.length === 0) {
    return <div className="card empty-card">Select a saved run to inspect track summaries.</div>;
  }

  return (
    <div className="card table-card">
      <div className="section-title">Reviewed run tracks</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Track</th>
              <th>Team</th>
              <th>Vote</th>
              <th>Frames</th>
              <th>First</th>
              <th>Last</th>
              <th>Avg conf</th>
              <th>Projected</th>
            </tr>
          </thead>
          <tbody>
            {tracks.map((track) => (
              <tr key={track.track_id ?? track.trackId}>
                <td>{track.track_id ?? track.trackId}</td>
                <td>{track.team_label || track.teamLabel || 'unassigned'}</td>
                <td>{track.team_vote_ratio ?? '0.0'}</td>
                <td>{track.frames ?? track.projectedPoints ?? 0}</td>
                <td>{track.first_frame ?? track.firstFrame ?? 'n/a'}</td>
                <td>{track.last_frame ?? track.lastFrame ?? 'n/a'}</td>
                <td>{track.average_confidence ?? track.averageConfidence ?? 'n/a'}</td>
                <td>{track.projected_points ?? track.projectedPoints ?? 'n/a'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FileLinks({ summary }) {
  const entries = [
    ['Overlay video', summary?.overlay_video],
    ['Detections CSV', summary?.detections_csv],
    ['Track summary CSV', summary?.track_summary_csv],
    ['Projection CSV', summary?.projection_csv],
    ['Calibration debug CSV', summary?.calibration_debug_csv],
    ['Entropy timeseries CSV', summary?.entropy_timeseries_csv],
    ['Goal events CSV', summary?.goal_events_csv],
    ['AI diagnostics JSON', summary?.diagnostics_json],
    ['Summary JSON', summary?.summary_json],
    ['All outputs zip', summary?.all_outputs_zip],
  ].filter((entry) => entry[1]);

  if (entries.length === 0) {
    return <div className="card empty-card">Select a saved run to inspect exported files.</div>;
  }

  return (
    <div className="card link-card">
      <div className="section-title">Reviewed run files</div>
      <div className="link-list">
        {entries.map(([label, path]) => (
          <a key={label} href={`${API_BASE}${path}`} target="_blank" rel="noreferrer">
            {label}
          </a>
        ))}
      </div>
    </div>
  );
}

function TrajectoryPanel({
  projectionState,
  selectedTrackIds,
  rankedTracks,
  currentFrame,
  fps,
  onToggleTrack,
  onResetSelection,
}) {
  const trackById = useMemo(
    () => new Map(rankedTracks.map((track) => [track.trackId, track])),
    [rankedTracks],
  );
  const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 25;
  const windowFrames = Math.max(Math.round(safeFps * TRAJECTORY_WINDOW_SECONDS), 90);
  const firstProjectedFrame = projectionState.rows.length ? projectionState.rows[0].frameIndex : 0;
  const focusFrame = Number.isFinite(currentFrame) && currentFrame > 0 ? currentFrame : firstProjectedFrame;
  const windowStartFrame = Math.max(0, focusFrame - windowFrames);
  const windowEndFrame = Math.max(focusFrame, firstProjectedFrame);

  const ballPoints = useMemo(
    () => projectionState.rows.filter((row) => row.rowType === 'ball' && row.frameIndex >= windowStartFrame && row.frameIndex <= windowEndFrame),
    [projectionState.rows, windowEndFrame, windowStartFrame],
  );
  const selectedTracks = selectedTrackIds
    .map((trackId) => trackById.get(trackId))
    .filter(Boolean)
    .map((track) => ({
      ...track,
      windowPoints: track.points.filter((point) => point.frameIndex >= windowStartFrame && point.frameIndex <= windowEndFrame),
    }))
    .filter((track) => track.windowPoints.length > 0);
  const candidateTracks = rankedTracks.slice(0, TRAJECTORY_TRACK_DISPLAY_LIMIT);
  const ballPath = buildTrajectoryPath(ballPoints);

  return (
    <section className="card trajectory-card">
      <div className="trajectory-header">
        <div>
          <div className="eyebrow">Projected movement</div>
          <div className="section-title trajectory-title">Ball Trajectory and Player Paths</div>
          <div className="trajectory-subtitle">
            Synced to the saved overlay. The pitch map shows the last {TRAJECTORY_WINDOW_SECONDS} seconds up to the current playback frame instead of the whole run at once.
          </div>
        </div>
        <button type="button" className="secondary-button compact-button" onClick={onResetSelection} disabled={!rankedTracks.length}>
          Reset to best 4
        </button>
      </div>

      {projectionState.loading ? <div className="empty-card">Loading projected trajectories…</div> : null}
      {!projectionState.loading && projectionState.error ? <div className="error-box">{projectionState.error}</div> : null}
      {!projectionState.loading && !projectionState.error && projectionState.rows.length === 0 ? (
        <div className="empty-card">No projected points were exported for this run, so there is no ball trajectory or player path to draw.</div>
      ) : null}

      {!projectionState.loading && !projectionState.error && projectionState.rows.length > 0 ? (
        <div className="trajectory-layout">
          <div className="trajectory-stage">
            <div className="trajectory-stage-meta">
              <div><span className="micro-label">Overlay focus</span><div>{formatSecondsLabel(focusFrame / safeFps)} · frame {focusFrame}</div></div>
              <div><span className="micro-label">Visible window</span><div>{formatSecondsLabel(windowStartFrame / safeFps)} to {formatSecondsLabel(windowEndFrame / safeFps)}</div></div>
            </div>
            <svg className="trajectory-pitch" viewBox="0 0 100 64" role="img" aria-label="Projected pitch trajectories">
              <rect x="0" y="0" width="100" height="64" rx="3" className="trajectory-pitch-bg" />
              <rect x="2.5" y="2.5" width="95" height="59" className="trajectory-pitch-line" />
              <line x1="50" y1="2.5" x2="50" y2="61.5" className="trajectory-pitch-line" />
              <circle cx="50" cy="32" r="9.15" className="trajectory-pitch-line" />
              <circle cx="50" cy="32" r="0.9" className="trajectory-pitch-fill" />
              <rect x="2.5" y="18.5" width="16.5" height="27" className="trajectory-pitch-line" />
              <rect x="2.5" y="25" width="5.5" height="14" className="trajectory-pitch-line" />
              <rect x="81" y="18.5" width="16.5" height="27" className="trajectory-pitch-line" />
              <rect x="92" y="25" width="5.5" height="14" className="trajectory-pitch-line" />
              <circle cx="13.5" cy="32" r="0.7" className="trajectory-pitch-fill" />
              <circle cx="86.5" cy="32" r="0.7" className="trajectory-pitch-fill" />

              {selectedTracks.map((track, index) => {
                const path = buildTrajectoryPath(track.windowPoints);
                const lastPoint = track.windowPoints[track.windowPoints.length - 1];
                const endPoint = lastPoint ? fieldPointToSvgPoint(lastPoint.fieldX, lastPoint.fieldY) : null;
                const color = TRAJECTORY_TRACK_COLORS[index % TRAJECTORY_TRACK_COLORS.length];
                if (!path || !endPoint) {
                  return null;
                }
                return (
                  <g key={track.trackId}>
                    <path d={path} className="trajectory-path" style={{ '--trajectory-color': color }} />
                    <circle cx={endPoint.x} cy={endPoint.y} r="1.2" className="trajectory-point" style={{ '--trajectory-color': color }} />
                    <text x={Math.min(96, endPoint.x + 1.4)} y={Math.max(4, endPoint.y - 1.4)} className="trajectory-label" style={{ '--trajectory-color': color }}>
                      #{track.trackId}
                    </text>
                  </g>
                );
              })}

              {ballPath ? (
                <g>
                  <path d={ballPath} className="trajectory-path ball" />
                  {ballPoints.length ? (() => {
                    const lastBallPoint = fieldPointToSvgPoint(
                      ballPoints[ballPoints.length - 1].fieldX,
                      ballPoints[ballPoints.length - 1].fieldY,
                    );
                    return (
                      <>
                        <circle cx={lastBallPoint.x} cy={lastBallPoint.y} r="1.15" className="trajectory-point ball" />
                        <text x={Math.min(95, lastBallPoint.x + 1.2)} y={Math.max(5, lastBallPoint.y - 1.2)} className="trajectory-label ball">
                          ball
                        </text>
                      </>
                    );
                  })() : null}
                </g>
              ) : null}
            </svg>
          </div>

          <div className="trajectory-sidebar">
            <div className="trajectory-summary-grid">
              <div className="trajectory-summary-card">
                <div className="micro-label">Ball samples in window</div>
                <div className="trajectory-summary-value">{ballPoints.length}</div>
              </div>
              <div className="trajectory-summary-card">
                <div className="micro-label">Selected players visible</div>
                <div className="trajectory-summary-value">{selectedTracks.length}</div>
              </div>
              <div className="trajectory-summary-card">
                <div className="micro-label">Projected tracks</div>
                <div className="trajectory-summary-value">{rankedTracks.length}</div>
              </div>
            </div>

            <div className="trajectory-chip-panel">
              <div className="micro-label">Toggle player paths</div>
              <div className="trajectory-chip-grid">
                {candidateTracks.map((track) => {
                  const isActive = selectedTrackIds.includes(track.trackId);
                  const firstFrame = track.firstFrame ?? track.first_frame ?? 0;
                  const lastFrame = track.lastFrame ?? track.last_frame ?? firstFrame;
                  return (
                    <button
                      key={track.trackId}
                      type="button"
                      className={`trajectory-chip ${isActive ? 'active-trajectory-chip' : ''}`}
                      onClick={() => onToggleTrack(track.trackId)}
                    >
                      <span className="trajectory-chip-title">#{track.trackId} · {formatTeamLabel(track.teamLabel)}</span>
                      <span className="trajectory-chip-meta">
                        {track.projectedPoints} projected · {track.frames} frames · visible {formatSecondsLabel(firstFrame / safeFps)} to {formatSecondsLabel(lastFrame / safeFps)}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export default function App() {
  const [appSpace, setAppSpace] = useState(() => {
    const stored = readStoredString(STORAGE_KEYS.appSpace, 'analysis');
    return stored === 'training' ? 'training' : 'analysis';
  });
  const [themeMode, setThemeMode] = useState(() => readStoredString(STORAGE_KEYS.themeMode, 'light'));
  const [config, setConfig] = useState({ player_models: [], ball_models: [], learn_cards: [], help_catalog: [] });
  const [soccerNetConfig, setSoccerNetConfig] = useState({ dataset_dir: '', splits: [], split_counts: {}, video_files: [], label_files: [], notes: [] });
  const [soccerNetSplit, setSoccerNetSplit] = useState(() => readStoredString(STORAGE_KEYS.soccerNetSplit, 'train'));
  const [soccerNetQuery, setSoccerNetQuery] = useState(() => readStoredString(STORAGE_KEYS.soccerNetQuery, ''));
  const [soccerNetGames, setSoccerNetGames] = useState([]);
  const [soccerNetGamesCount, setSoccerNetGamesCount] = useState(0);
  const [soccerNetResultLimit, setSoccerNetResultLimit] = useState(24);
  const [soccerNetResultsExpanded, setSoccerNetResultsExpanded] = useState(false);
  const [soccerNetSelectedGame, setSoccerNetSelectedGame] = useState('');
  const [soccerNetPassword, setSoccerNetPassword] = useState('');
  const [soccerNetFiles, setSoccerNetFiles] = useState(() => {
    const stored = readStoredJson(STORAGE_KEYS.soccerNetFiles, null);
    return Array.isArray(stored) && stored.length > 0 ? stored : DEFAULT_SOCCERNET_FILES;
  });
  const [soccerNetError, setSoccerNetError] = useState('');
  const [soccerNetLoadingGames, setSoccerNetLoadingGames] = useState(false);
  const [soccerNetDownloadJob, setSoccerNetDownloadJob] = useState(null);
  const [form, setForm] = useState(() => {
    const stored = readStoredJson(STORAGE_KEYS.form, {});
    return { ...defaultForm, ...(stored && typeof stored === 'object' ? stored : {}) };
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [source, setSource] = useState(null);
  const [sourceError, setSourceError] = useState('');
  const [isLoadingSource, setIsLoadingSource] = useState(false);
  const [livePreviewUrl, setLivePreviewUrl] = useState('');
  const [folderScan, setFolderScan] = useState(null);
  const [scanError, setScanError] = useState('');
  const [recentRuns, setRecentRuns] = useState([]);
  const [recentRunsError, setRecentRunsError] = useState('');
  const [selectedRun, setSelectedRun] = useState(null);
  const [reviewError, setReviewError] = useState('');
  const [isRefreshingDiagnostics, setIsRefreshingDiagnostics] = useState(false);
  const [projectionState, setProjectionState] = useState({ loading: false, error: '', rows: [] });
  const [selectedTrajectoryTrackIds, setSelectedTrajectoryTrackIds] = useState([]);
  const [reviewPlaybackTime, setReviewPlaybackTime] = useState(0);
  const [job, setJob] = useState(null);
  const [activeExperiment, setActiveExperiment] = useState(null);
  const [backendActivity, setBackendActivity] = useState({
    lastAttemptAt: 0,
    lastResponseAt: 0,
    lastFailureAt: 0,
    lastPath: '',
  });
  const [statusClock, setStatusClock] = useState(() => Date.now());
  const [jobError, setJobError] = useState('');
  const [jobActionPending, setJobActionPending] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [workspaceMode, setWorkspaceMode] = useState('input');
  const [reviewPanel, setReviewPanel] = useState('overview');
  const [analysisSidebarWidth, setAnalysisSidebarWidth] = useState(() => {
    const stored = Number(readStoredString(STORAGE_KEYS.analysisSidebarWidth, ''));
    return Number.isFinite(stored) && stored > 0 ? stored : DEFAULT_ANALYSIS_SIDEBAR_WIDTH;
  });
  const [layoutWidth, setLayoutWidth] = useState(0);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const pollRef = useRef(null);
  const soccerNetPollRef = useRef(null);
  const trajectorySelectionRunRef = useRef('');
  const sourceVideoRef = useRef(null);
  const reviewVideoRef = useRef(null);
  const layoutRef = useRef(null);

  async function apiFetch(url, options) {
    const startedAt = Date.now();
    const path = typeof url === 'string' ? url.replace(API_BASE, '') : 'request';
    setBackendActivity((current) => ({
      ...current,
      lastAttemptAt: startedAt,
      lastPath: path,
    }));

    try {
      const response = await fetch(url, options);
      setBackendActivity((current) => ({
        ...current,
        lastAttemptAt: startedAt,
        lastResponseAt: Date.now(),
        lastPath: path,
      }));
      return response;
    } catch (error) {
      setBackendActivity((current) => ({
        ...current,
        lastAttemptAt: startedAt,
        lastFailureAt: Date.now(),
        lastPath: path,
      }));
      throw error;
    }
  }

  useEffect(() => {
    apiFetch(`${API_BASE}/api/config`)
      .then((response) => response.json())
      .then((data) => {
        setConfig(data);
        setForm((current) => ({
          ...current,
          trackerMode: current.trackerMode || data.default_player_tracker_mode || defaultForm.trackerMode,
        }));
      })
      .catch((error) => {
        console.error(error);
        setJobError('Could not reach the backend. Start backend first.');
      });
  }, []);

  useEffect(() => {
    apiFetch(`${API_BASE}/api/soccernet/config`)
      .then((response) => response.json())
      .then((data) => {
        setSoccerNetConfig(data);
        setForm((current) => (
          current.folderPath
            ? current
            : { ...current, folderPath: data.dataset_dir || current.folderPath }
        ));
      })
      .catch((error) => {
        console.error(error);
        setSoccerNetError('Could not load SoccerNet config.');
      });
  }, []);

  useEffect(() => {
    loadRecentRuns();
  }, []);

  useEffect(() => {
    loadBackendJobs({ hydrateActive: true });
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.themeMode, themeMode);
    } catch {}
    const resolvedTheme = themeMode === 'auto'
      ? (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : themeMode;
    document.documentElement.setAttribute('data-theme', resolvedTheme);
    document.documentElement.setAttribute('data-theme-mode', themeMode);
  }, [themeMode]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.appSpace, appSpace);
    } catch {}
  }, [appSpace]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.form, JSON.stringify(form));
      window.localStorage.setItem(STORAGE_KEYS.localVideoPath, form.localVideoPath || '');
      window.localStorage.setItem(STORAGE_KEYS.folderPath, form.folderPath || '');
    } catch {}
  }, [form]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.soccerNetSplit, soccerNetSplit);
      window.localStorage.setItem(STORAGE_KEYS.soccerNetQuery, soccerNetQuery);
      window.localStorage.setItem(STORAGE_KEYS.soccerNetFiles, JSON.stringify(soccerNetFiles));
    } catch {}
  }, [soccerNetSplit, soccerNetQuery, soccerNetFiles]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.analysisSidebarWidth, String(Math.round(analysisSidebarWidth)));
    } catch {}
  }, [analysisSidebarWidth]);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      loadSoccerNetGames();
    }, 250);
    return () => window.clearTimeout(handle);
  }, [soccerNetSplit, soccerNetQuery]);

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
      }
      if (soccerNetPollRef.current) {
        window.clearInterval(soccerNetPollRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const handle = window.setInterval(() => {
      loadBackendJobs({ hydrateActive: true });
    }, 10000);
    return () => window.clearInterval(handle);
  }, [job?.job_id, job?.status]);

  useEffect(() => {
    const handle = window.setInterval(() => {
      setStatusClock(Date.now());
    }, 5000);
    return () => window.clearInterval(handle);
  }, []);

  useEffect(() => {
    if (!layoutRef.current || typeof ResizeObserver === 'undefined') {
      return undefined;
    }
    const node = layoutRef.current;
    const updateWidth = () => setLayoutWidth(node.getBoundingClientRect().width);
    updateWidth();
    const observer = new ResizeObserver(() => updateWidth());
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (workspaceMode !== 'job' || isActiveJobStatus(job?.status)) {
      return;
    }
    setWorkspaceMode(selectedRun ? 'review' : 'input');
  }, [job?.status, selectedRun?.run_id, workspaceMode]);

  useEffect(() => {
    const projectionCsvPath = selectedRun?.summary?.projection_csv;
    if (!projectionCsvPath) {
      setProjectionState({ loading: false, error: '', rows: [] });
      setSelectedTrajectoryTrackIds([]);
      trajectorySelectionRunRef.current = '';
      return;
    }

    let cancelled = false;
    setProjectionState((current) => ({ ...current, loading: true, error: '' }));

    apiFetch(`${API_BASE}${projectionCsvPath}`)
      .then(async (response) => {
        const text = await response.text();
        if (!response.ok) {
          throw new Error(text || 'Could not load projection CSV');
        }
        return text;
      })
      .then((text) => {
        if (cancelled) return;
        setProjectionState({ loading: false, error: '', rows: parseProjectionCsv(text) });
      })
      .catch((error) => {
        if (cancelled) return;
        setProjectionState({ loading: false, error: error.message || 'Could not load projection CSV', rows: [] });
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRun?.summary?.projection_csv]);

  const maxAnalysisSidebarWidth = useMemo(() => {
    if (!layoutWidth) {
      return MAX_ANALYSIS_SIDEBAR_WIDTH;
    }
    return Math.max(
      MIN_ANALYSIS_SIDEBAR_WIDTH,
      Math.min(
        MAX_ANALYSIS_SIDEBAR_WIDTH,
        layoutWidth - ANALYSIS_MAIN_MIN_WIDTH - SIDEBAR_RESIZER_WIDTH,
      ),
    );
  }, [layoutWidth]);

  const effectiveAnalysisSidebarWidth = workspaceMode === 'review'
    ? DEFAULT_ANALYSIS_SIDEBAR_WIDTH
    : clampNumber(analysisSidebarWidth, MIN_ANALYSIS_SIDEBAR_WIDTH, maxAnalysisSidebarWidth);

  useEffect(() => {
    if (!isResizingSidebar) {
      return undefined;
    }

    function stopResizing() {
      setIsResizingSidebar(false);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    }

    function handlePointerMove(event) {
      if (!layoutRef.current) {
        return;
      }
      const rect = layoutRef.current.getBoundingClientRect();
      const nextWidth = clampNumber(
        event.clientX - rect.left,
        MIN_ANALYSIS_SIDEBAR_WIDTH,
        maxAnalysisSidebarWidth,
      );
      setAnalysisSidebarWidth(nextWidth);
    }

    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'ew-resize';
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResizing);
    window.addEventListener('pointercancel', stopResizing);

    return () => {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResizing);
      window.removeEventListener('pointercancel', stopResizing);
    };
  }, [isResizingSidebar, maxAnalysisSidebarWidth]);

  const currentJob = job;
  const hasActiveJob = isActiveJobStatus(currentJob?.status);
  const canPauseJob = currentJob?.status === 'running' || currentJob?.status === 'queued';
  const canResumeJob = currentJob?.status === 'paused';
  const canStopJob = isActiveJobStatus(currentJob?.status);
  const latestBackendFailure = backendActivity.lastFailureAt > backendActivity.lastResponseAt ? backendActivity.lastFailureAt : 0;
  const backendStatus = useMemo(() => {
    if (latestBackendFailure && statusClock - latestBackendFailure < BACKEND_FAILURE_WINDOW_MS) {
      return {
        label: 'backend offline',
        tone: 'offline',
        title: backendActivity.lastPath ? `Latest failed request: ${backendActivity.lastPath}` : 'Latest API request failed',
      };
    }
    if (backendActivity.lastResponseAt && statusClock - backendActivity.lastResponseAt < BACKEND_ACTIVITY_WINDOW_MS) {
      return {
        label: 'backend active',
        tone: 'online',
        title: backendActivity.lastPath ? `Latest API response: ${backendActivity.lastPath}` : 'Backend responded recently',
      };
    }
    if (backendActivity.lastAttemptAt) {
      return {
        label: 'backend idle',
        tone: 'idle',
        title: backendActivity.lastPath ? `Last API activity: ${backendActivity.lastPath}` : 'No recent API traffic',
      };
    }
    return {
      label: 'backend waiting',
      tone: 'idle',
      title: 'No API traffic yet',
    };
  }, [backendActivity.lastAttemptAt, backendActivity.lastPath, backendActivity.lastResponseAt, latestBackendFailure, statusClock]);
  const reviewedRun = selectedRun;
  const summary = reviewedRun?.summary || null;
  const sourceLabel = source
    ? source.display_name || basenameFromPath(source.video_url)
    : selectedFile?.name || basenameFromPath(form.localVideoPath) || '';
  const reviewedRunId = reviewedRun?.run_id || '';
  const reviewedClipName = basenameFromPath(summary?.input_video);
  const activeExperimentSourcePath = activeExperiment?.summary?.current_source_video_path || '';
  const activeExperimentLabelPath = activeExperiment?.summary?.current_label_path || '';
  const activeExperimentClipName = basenameFromPath(activeExperimentSourcePath);
  const previewMatchesActiveExperiment = Boolean(
    source?.path && activeExperimentSourcePath && source.path === activeExperimentSourcePath,
  );
  const detectorModelOptions = config.detector_models?.length ? config.detector_models : config.player_models || [];
  const playerTrackerModes = config.player_tracker_modes?.length ? config.player_tracker_modes : ['hybrid_reid', 'bytetrack'];
  const activeDetectorLabel = config.active_detector_label || config.active_detector || 'soccana';
  const activeDetectorIsCustom = Boolean(config.active_detector_is_custom && config.active_detector !== 'soccana');
  const helpIndex = useMemo(
    () => buildHelpIndex(config.help_catalog),
    [config.help_catalog],
  );
  const visibleSoccerNetGames = useMemo(
    () => soccerNetGames.slice(0, soccerNetResultLimit),
    [soccerNetGames, soccerNetResultLimit],
  );
  const selectedSoccerNetGameMeta = useMemo(
    () => parseSoccerNetGame(soccerNetSelectedGame),
    [soccerNetSelectedGame],
  );
  const hasLongSoccerNetResultList = soccerNetGamesCount > SOCCERNET_AUTO_EXPAND_LIMIT;
  const canExpandSoccerNetResults = soccerNetGamesCount > 0;
  const soccerNetResultsButtonLabel = soccerNetResultsExpanded
    ? 'Hide matches'
    : `Browse matches${soccerNetGamesCount ? ` (${soccerNetGamesCount})` : ''}`;
  const shouldShowCollapsedSoccerNetPreview = canExpandSoccerNetResults && hasLongSoccerNetResultList && !soccerNetResultsExpanded;
  const headlineDiagnostics = useMemo(
    () => (summary?.diagnostics || []).slice(0, 4),
    [summary],
  );
  const hasExplicitIdentityMetrics = summary && summary.tracklet_merges_applied != null && summary.raw_unique_player_track_ids != null;
  const reviewQuickFacts = useMemo(() => {
    if (!summary) return [];
    const calibrationAttempts = summary.field_calibration_refresh_attempts || 0;
    const calibrationSuccesses = summary.field_calibration_refresh_successes || 0;
    const calibrationRate = summary.field_calibration_success_rate ?? (calibrationAttempts > 0 ? calibrationSuccesses / calibrationAttempts : 0);
    return [
      ['Clip', reviewedClipName || 'Unknown'],
      ['Frames', summary.frames_processed || 0],
      ['Tracker', summary.player_tracker_mode || 'n/a'],
      ['Calibration', `${calibrationSuccesses}/${calibrationAttempts} (${formatPercent(calibrationRate)})`],
      ['Player IDs', hasExplicitIdentityMetrics ? `${summary.raw_unique_player_track_ids} -> ${summary.unique_player_track_ids || 0}` : (summary.unique_player_track_ids || 0)],
      ['Teams', `${summary.home_tracks || 0} home / ${summary.away_tracks || 0} away`],
      ['Ball / frame', summary.average_ball_detections_per_frame ?? 'n/a'],
    ];
  }, [summary, reviewedClipName, hasExplicitIdentityMetrics]);

  const runStats = useMemo(() => {
    if (!summary) return [];
    const refreshFrames = summary.field_calibration_refresh_frames || 10;
    const calibrationAttempts = summary.field_calibration_refresh_attempts || 0;
    const calibrationSuccesses = summary.field_calibration_refresh_successes || 0;
    const calibrationRate = summary.field_calibration_success_rate ?? (calibrationAttempts > 0 ? calibrationSuccesses / calibrationAttempts : 0);
    const rows = [
      ['Frames', summary.frames_processed || 0, 'Decoded frames pushed through detection and tracking'],
      ['Tracker mode', summary.player_tracker_mode || 'n/a', 'Hybrid ReID adds appearance features and a stitch pass; ByteTrack is the legacy fallback'],
      ['Detector classes', `P ${formatClassIds(summary.player_detector_class_ids)} · B ${formatClassIds(summary.ball_detector_class_ids)}`, `Resolved from ${summary.detector_class_names_source || 'unknown'}`],
      ['Player track IDs', summary.unique_player_track_ids || 0, hasExplicitIdentityMetrics ? 'Canonical stitched IDs after any tracklet merges' : 'Track IDs from the saved run summary'],
      ['Home tracks', summary.home_tracks || 0, 'Unsupervised jersey-color split, not official metadata'],
      ['Away tracks', summary.away_tracks || 0, 'Should roughly match the second main kit cluster'],
      ['Raw detector sample', summary.raw_detector_boxes_sampled || 0, `Unfiltered detector boxes across the first ${summary.detector_debug_sample_frames || 0} frames`],
      ['Projected player anchors', summary.projected_player_points || 0, 'Per-frame player anchor samples that landed on the pitch map, not unique players'],
      ['Projection stages', `H ${summary.frames_with_field_homography || 0} · Fresh ${summary.frames_with_usable_homography || 0} · Blocked ${summary.frames_projection_blocked_by_stale || 0} · P ${summary.frames_with_projected_points || 0}`, 'Frames with any homography, fresh projection, stale-blocked projection, and projected output'],
      ['Projected fresh/stale', `P ${summary.projected_player_points_fresh || 0}/${summary.projected_player_points_stale || 0} · B ${summary.projected_ball_points_fresh || 0}/${summary.projected_ball_points_stale || 0}`, 'Projected player and ball points while calibration was fresh vs stale'],
      ['Rows fresh/stale', `${summary.player_rows_while_calibration_fresh || 0}/${summary.player_rows_while_calibration_stale || 0}`, 'Player detections observed while calibration was fresh vs stale'],
      ['Avg pitch keypoints', summary.average_visible_pitch_keypoints || 0, `Visible field keypoints on each ${refreshFrames}-frame calibration refresh`],
      ['Calibration success', formatPercent(calibrationRate), `Accepted refreshes ${calibrationSuccesses}/${calibrationAttempts}`],
      ['Stale recovery', `${summary.field_calibration_stale_recovery_successes || 0}/${summary.field_calibration_stale_recovery_attempts || 0}`, 'Refresh attempts accepted while calibration had already gone stale'],
      ['Calib gate rejects', formatCalibrationRejectionSummary(summary), 'Calibration gate-hit counts for rejected refreshes; categories can overlap'],
    ];
    if (hasExplicitIdentityMetrics) {
      rows.splice(2, 0,
        ['Raw player IDs', summary.raw_unique_player_track_ids, 'Unstitched online tracker IDs before the merge pass'],
        ['Tracklet merges', summary.tracklet_merges_applied ?? 0, 'Accepted raw-to-canonical identity merges'],
      );
    }
    return rows;
  }, [summary, hasExplicitIdentityMetrics]);

  const runMetricSections = useMemo(() => {
    if (!runStats.length) return [];
    const metricByLabel = new Map(runStats.map(([label, value, hint]) => [label, { label, value, hint }]));
    const pick = (label, wide = false) => {
      const item = metricByLabel.get(label);
      return item ? { ...item, wide, helpId: REVIEW_METRIC_HELP_IDS[label] } : null;
    };
    return [
      {
        title: 'Tracking',
        items: [
          pick('Tracker mode'),
          pick('Raw player IDs'),
          pick('Player track IDs'),
          pick('Tracklet merges'),
          pick('Home tracks'),
          pick('Away tracks'),
          pick('Rows fresh/stale', true),
        ].filter(Boolean),
      },
      {
        title: 'Calibration',
        items: [
          pick('Avg pitch keypoints'),
          pick('Calibration success'),
          pick('Stale recovery'),
          pick('Calib gate rejects', true),
        ].filter(Boolean),
      },
      {
        title: 'Projection',
        items: [
          pick('Projected player anchors'),
          pick('Projection stages', true),
          pick('Projected fresh/stale', true),
        ].filter(Boolean),
      },
      {
        title: 'Detection',
        items: [
          pick('Frames'),
          pick('Detector classes', true),
          pick('Raw detector sample'),
        ].filter(Boolean),
      },
    ].filter((section) => section.items.length > 0);
  }, [runStats]);

  const rankedTrajectoryTracks = useMemo(() => {
    const rows = projectionState.rows.filter((row) => row.rowType === 'player');
    if (!rows.length) {
      return [];
    }

    const trackSummaryById = new Map(
      (summary?.top_tracks || []).map((track) => [Number(track.track_id), track]),
    );
    const groupedTracks = new Map();

    rows.forEach((row) => {
      if (!groupedTracks.has(row.trackId)) {
        groupedTracks.set(row.trackId, {
          trackId: row.trackId,
          teamLabel: row.teamLabel,
          points: [],
        });
      }
      groupedTracks.get(row.trackId).points.push(row);
    });

    return Array.from(groupedTracks.values())
      .map((track) => {
        const summaryTrack = trackSummaryById.get(track.trackId);
        const firstFrame = Number(summaryTrack?.first_frame ?? track.points[0]?.frameIndex ?? 0);
        const lastFrame = Number(summaryTrack?.last_frame ?? track.points[track.points.length - 1]?.frameIndex ?? firstFrame);
        return {
          ...track,
          teamLabel: track.teamLabel || summaryTrack?.team_label || '',
          team_vote_ratio: summaryTrack?.team_vote_ratio ?? 0,
          projectedPoints: track.points.length,
          projected_points: track.points.length,
          frames: Number(summaryTrack?.frames || track.points.length),
          firstFrame,
          first_frame: firstFrame,
          lastFrame,
          last_frame: lastFrame,
          averageConfidence: Number(summaryTrack?.average_confidence || 0),
          average_confidence: Number(summaryTrack?.average_confidence || 0),
        };
      })
      .sort((a, b) => (
        b.projectedPoints - a.projectedPoints
        || b.frames - a.frames
        || b.averageConfidence - a.averageConfidence
        || a.trackId - b.trackId
      ));
  }, [projectionState.rows, summary?.top_tracks]);

  useEffect(() => {
    if (!reviewedRunId) {
      setSelectedTrajectoryTrackIds([]);
      trajectorySelectionRunRef.current = '';
      return;
    }
    if (!rankedTrajectoryTracks.length || trajectorySelectionRunRef.current === reviewedRunId) {
      return;
    }
    trajectorySelectionRunRef.current = reviewedRunId;
    setSelectedTrajectoryTrackIds(
      rankedTrajectoryTracks
        .slice(0, DEFAULT_TRAJECTORY_TRACK_COUNT)
        .map((track) => track.trackId),
    );
  }, [reviewedRunId, rankedTrajectoryTracks]);

  useEffect(() => {
    setReviewPlaybackTime(0);
    if (reviewVideoRef.current) {
      reviewVideoRef.current.currentTime = 0;
    }
  }, [reviewedRunId]);

  const reviewTrackRows = useMemo(
    () => (rankedTrajectoryTracks.length ? rankedTrajectoryTracks : summary?.top_tracks || []),
    [rankedTrajectoryTracks, summary?.top_tracks],
  );
  const reviewPlaybackFrame = useMemo(() => {
    const fps = Number(summary?.fps);
    const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 25;
    return Math.max(0, Math.round(reviewPlaybackTime * safeFps));
  }, [reviewPlaybackTime, summary?.fps]);

  function startSidebarResize(event) {
    if (workspaceMode === 'review' || event.button !== 0) {
      return;
    }
    event.preventDefault();
    setIsResizingSidebar(true);
  }

  function resetSidebarWidth() {
    setAnalysisSidebarWidth(DEFAULT_ANALYSIS_SIDEBAR_WIDTH);
  }

  function resetTrajectorySelection() {
    setSelectedTrajectoryTrackIds(
      rankedTrajectoryTracks
        .slice(0, DEFAULT_TRAJECTORY_TRACK_COUNT)
        .map((track) => track.trackId),
    );
  }

  function toggleTrajectoryTrack(trackId) {
    setSelectedTrajectoryTrackIds((current) => (
      current.includes(trackId)
        ? current.filter((value) => value !== trackId)
        : [...current, trackId]
    ));
  }

  function handleSidebarResizerKeyDown(event) {
    const step = event.shiftKey ? 32 : 16;
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      setAnalysisSidebarWidth((current) => clampNumber(current - step, MIN_ANALYSIS_SIDEBAR_WIDTH, maxAnalysisSidebarWidth));
      return;
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      setAnalysisSidebarWidth((current) => clampNumber(current + step, MIN_ANALYSIS_SIDEBAR_WIDTH, maxAnalysisSidebarWidth));
      return;
    }
    if (event.key === 'Home') {
      event.preventDefault();
      setAnalysisSidebarWidth(MIN_ANALYSIS_SIDEBAR_WIDTH);
      return;
    }
    if (event.key === 'End') {
      event.preventDefault();
      setAnalysisSidebarWidth(maxAnalysisSidebarWidth);
    }
  }

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function handleTrainingActiveDetectorChange(nextActiveDetector) {
    const nextId = typeof nextActiveDetector === 'string'
      ? nextActiveDetector
      : nextActiveDetector?.id || nextActiveDetector?.active_detector || 'soccana';
    const nextLabel = typeof nextActiveDetector === 'object'
      ? nextActiveDetector?.label || nextActiveDetector?.active_detector_label || nextId
      : nextId;
    setConfig((current) => ({
      ...current,
      active_detector: nextId,
      active_detector_label: nextLabel,
      active_detector_is_custom: nextId !== 'soccana',
    }));
  }

  async function scanFolderPath(folderPath) {
    setScanError('');
    setFolderScan(null);
    const response = await apiFetch(`${API_BASE}/api/scan-folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_path: folderPath }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Scan failed');
    }
    setFolderScan(data);
  }

  async function loadSoccerNetGames() {
    setSoccerNetError('');
    setSoccerNetLoadingGames(true);
    try {
      const params = new URLSearchParams({
        split: soccerNetSplit,
        query: soccerNetQuery,
        limit: '200',
      });
      const response = await apiFetch(`${API_BASE}/api/soccernet/games?${params.toString()}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load SoccerNet games');
      }
      setSoccerNetGames(data.games || []);
      setSoccerNetGamesCount(data.count || 0);
      setSoccerNetResultLimit(24);
      setSoccerNetResultsExpanded((data.games || []).length > 0 && (data.games || []).length <= SOCCERNET_AUTO_EXPAND_LIMIT);
      if ((data.games || []).length > 0) {
        setSoccerNetSelectedGame((current) => (current && data.games.includes(current) ? current : data.games[0]));
      } else {
        setSoccerNetSelectedGame('');
      }
    } catch (error) {
      setSoccerNetError(error.message || 'Could not load SoccerNet games');
    } finally {
      setSoccerNetLoadingGames(false);
    }
  }

  function toggleSoccerNetFile(fileName) {
    setSoccerNetFiles((current) => (
      current.includes(fileName)
        ? current.filter((item) => item !== fileName)
        : [...current, fileName]
    ));
  }

  function stopSoccerNetPolling() {
    if (soccerNetPollRef.current) {
      window.clearInterval(soccerNetPollRef.current);
      soccerNetPollRef.current = null;
    }
  }

  function startSoccerNetPolling(jobId) {
    stopSoccerNetPolling();

    async function tick() {
      try {
        const response = await apiFetch(`${API_BASE}/api/soccernet/downloads/${jobId}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Could not poll SoccerNet download');
        }
        setSoccerNetDownloadJob(data);
        if (data.status === 'completed' || data.status === 'failed') {
          stopSoccerNetPolling();
        }
      } catch (error) {
        console.error(error);
        setSoccerNetError(error.message || 'Could not poll SoccerNet download');
        stopSoccerNetPolling();
      }
    }

    tick();
    soccerNetPollRef.current = window.setInterval(tick, 1500);
  }

  async function handleSoccerNetDownload() {
    setSoccerNetError('');
    if (!soccerNetSelectedGame) {
      setSoccerNetError('Select a SoccerNet game first.');
      return;
    }
    if (!soccerNetPassword.trim()) {
      setSoccerNetError('SoccerNet password is required.');
      return;
    }
    if (soccerNetFiles.length === 0) {
      setSoccerNetError('Select at least one SoccerNet file to download.');
      return;
    }

    try {
      const response = await apiFetch(`${API_BASE}/api/soccernet/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          split: soccerNetSplit,
          game: soccerNetSelectedGame,
          password: soccerNetPassword,
          files: soccerNetFiles,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not start SoccerNet download');
      }
      setSoccerNetDownloadJob(data);
      startSoccerNetPolling(data.job_id);
    } catch (error) {
      setSoccerNetError(error.message || 'Could not start SoccerNet download');
    }
  }

  async function handleUseSoccerNetFolder() {
    const datasetDir = soccerNetConfig.dataset_dir;
    if (!datasetDir) {
      setSoccerNetError('SoccerNet dataset directory is unavailable.');
      return;
    }
    updateForm('folderPath', datasetDir);
    try {
      await scanFolderPath(datasetDir);
    } catch (error) {
      setScanError(error.message || 'Scan failed');
    }
  }

  async function handleLoadSource(overridePathOrEvent = null) {
    const overridePath = typeof overridePathOrEvent === 'string' ? overridePathOrEvent : null;
    const localVideoPath = String(overridePath ?? form.localVideoPath ?? '').trim();
    const useSelectedFile = !overridePath && selectedFile && !localVideoPath;

    if (!useSelectedFile && !localVideoPath) {
      setSourceError('Choose a video file or enter a local path first.');
      return;
    }

    setSourceError('');
    setIsLoadingSource(true);
    setLivePreviewUrl('');
    try {
      const payload = new FormData();
      if (useSelectedFile) {
        payload.append('video_file', selectedFile);
      } else {
        payload.append('local_video_path', localVideoPath);
      }

      const response = await apiFetch(`${API_BASE}/api/source`, {
        method: 'POST',
        body: payload,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load input clip');
      }
      setSource(data);
      setWorkspaceMode('input');
    } catch (error) {
      setSourceError(error.message || 'Could not load input clip');
    } finally {
      setIsLoadingSource(false);
    }
  }

  function handleStartLivePreview() {
    if (!source?.source_id) {
      setSourceError('Load an input clip before starting live preview.');
      return;
    }
    const params = new URLSearchParams({
      source_id: source.source_id,
      detector_model: form.detectorModel,
      tracker_mode: form.trackerMode,
      include_ball: String(form.includeBall),
      player_conf: form.playerConf,
      ball_conf: form.ballConf,
      iou: form.iou,
    });
    setLivePreviewUrl(`${API_BASE}/api/live-preview?${params.toString()}&ts=${Date.now()}`);
    setWorkspaceMode('live');
  }

  function handleStopLivePreview() {
    setLivePreviewUrl('');
  }

  function clearPolling() {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  async function loadBackendJobs({ hydrateActive = false } = {}) {
    try {
      const response = await apiFetch(`${API_BASE}/api/jobs`);
      const data = await response.json();
      if (!response.ok || !Array.isArray(data)) {
        throw new Error('Could not load backend jobs');
      }

      if (!hydrateActive) {
        return data;
      }

      const activeJob = data.find((item) => isActiveJobStatus(item?.status)) || null;
      if (activeJob) {
        setActiveExperiment(null);
        setJob((current) => {
          if (current?.job_id === activeJob.job_id && isActiveJobStatus(current?.status)) {
            return current;
          }
          return activeJob;
        });
        if (isActiveJobStatus(activeJob.status)) {
          startPolling(activeJob.job_id);
        }
        return data;
      }

      clearPolling();
      setJob(null);

      try {
        const experimentResponse = await apiFetch(`${API_BASE}/api/experiments/active`);
        if (experimentResponse.ok) {
          const experiment = await experimentResponse.json();
          setActiveExperiment(experiment || null);
        } else {
          setActiveExperiment(null);
        }
      } catch (error) {
        console.error(error);
        setActiveExperiment(null);
      }
      return data;
    } catch (error) {
      console.error(error);
      clearPolling();
      setJob(null);
      setActiveExperiment(null);
      return [];
    }
  }

  function resetLoadedSource() {
    setSource(null);
    setLivePreviewUrl('');
  }

  async function handleUseActiveExperimentClip() {
    if (!activeExperimentSourcePath) {
      return;
    }
    if (activeExperimentLabelPath) {
      updateForm('labelPath', activeExperimentLabelPath);
    }
    updateForm('localVideoPath', activeExperimentSourcePath);
    resetLoadedSource();
    await handleLoadSource(activeExperimentSourcePath);
  }

  function handleResetSavedWorkspace() {
    try {
      Object.values(STORAGE_KEYS).forEach((key) => window.localStorage.removeItem(key));
    } catch {}
    window.location.reload();
  }

  async function loadRecentRuns() {
    setRecentRunsError('');
    try {
      const response = await apiFetch(`${API_BASE}/api/runs/recent?limit=1000`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error('Could not load recent runs');
      }
      setRecentRuns(data);
    } catch (error) {
      console.error(error);
      setRecentRunsError(error.message || 'Could not load recent runs');
    }
  }

  async function handleLoadRun(runId) {
    setReviewError('');
    try {
      const response = await apiFetch(`${API_BASE}/api/runs/${runId}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load run');
      }
      setSelectedRun(data);
      setWorkspaceMode('review');
      setReviewPanel('overview');
    } catch (error) {
      setReviewError(error.message || 'Could not load run');
    }
  }

  async function handleRefreshDiagnostics() {
    if (!reviewedRunId) return;
    setReviewError('');
    setIsRefreshingDiagnostics(true);
    try {
      const response = await apiFetch(`${API_BASE}/api/runs/${reviewedRunId}/refresh-diagnostics`, {
        method: 'POST',
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not refresh diagnostics');
      }
      setSelectedRun(data);
      setWorkspaceMode('review');
      setReviewPanel('overview');
      loadRecentRuns();
    } catch (error) {
      setReviewError(error.message || 'Could not refresh diagnostics');
    } finally {
      setIsRefreshingDiagnostics(false);
    }
  }

  async function handleScanFolder() {
    try {
      await scanFolderPath(form.folderPath);
    } catch (error) {
      setScanError(error.message || 'Scan failed');
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setJobError('');
    setIsSubmitting(true);
    setLivePreviewUrl('');

    const localVideoPath = String(form.localVideoPath || '').trim();
    const useSelectedFile = !source?.source_id && selectedFile && !localVideoPath;
    if (!source?.source_id && !useSelectedFile && !localVideoPath) {
      setJobError('Load an input clip or pick a file/path first.');
      setIsSubmitting(false);
      return;
    }

    const payload = new FormData();
    if (source?.source_id) {
      payload.append('source_id', source.source_id);
    } else if (useSelectedFile) {
      payload.append('video_file', selectedFile);
    }
    payload.append('local_video_path', source?.source_id ? '' : localVideoPath);
    payload.append('label_path', form.labelPath);
    payload.append('detector_model', form.detectorModel);
    payload.append('tracker_mode', form.trackerMode);
    payload.append('include_ball', String(form.includeBall));
    payload.append('player_conf', form.playerConf);
    payload.append('ball_conf', form.ballConf);
    payload.append('iou', form.iou);

    try {
      const response = await apiFetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: payload,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Run failed to start');
      }
      const nextJob = {
        job_id: data.job_id,
        status: 'queued',
        logs: ['Run created. Waiting for first update...'],
      };
      setJob(nextJob);
      setWorkspaceMode('job');
      startPolling(data.job_id);
    } catch (error) {
      setJobError(error.message || 'Run failed to start');
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleJobAction(action) {
    if (!currentJob?.job_id) {
      return;
    }
    setJobError('');
    setJobActionPending(action);
    try {
      const response = await apiFetch(`${API_BASE}/api/jobs/${currentJob.job_id}/${action}`, {
        method: 'POST',
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Could not ${action} job`);
      }
      setJob(data);
    } catch (error) {
      setJobError(error.message || `Could not ${action} job`);
    } finally {
      setJobActionPending('');
    }
  }

  function startPolling(jobId) {
    clearPolling();

    async function tick() {
      try {
        const response = await apiFetch(`${API_BASE}/api/jobs/${jobId}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Could not poll analysis job');
        }
        setJob(data);
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
          clearPolling();
          if (data.status === 'completed' && data.run_id) {
            await handleLoadRun(data.run_id);
          } else if (data.status === 'completed' && data.summary) {
            setSelectedRun(data);
            setWorkspaceMode('review');
            setReviewPanel('overview');
          }
          loadRecentRuns();
          loadBackendJobs({ hydrateActive: true });
        }
      } catch (error) {
        console.error(error);
      }
    }

    tick();
    pollRef.current = window.setInterval(tick, 1500);
  }

  return (
    <div className="page-shell">
      <header className="hero card">
        <div>
          <div className="eyebrow">football tactical demo</div>
          <h1>Detect, track, calibrate the field, and project the play.</h1>
          <p>
            Local React + FastAPI demo for wide-angle football analysis with football-specific detection, hybrid appearance-aware player tracking, per-frame field-keypoint calibration with rolling homography smoothing, live model preview, and projected tactical overlays.
          </p>
        </div>
        <div className="hero-pills">
          <span className="connection-chip online" title="This frontend is mounted and interactive">
            <span className="chip-dot" aria-hidden="true" />
            frontend live
          </span>
          <span className={`connection-chip ${backendStatus.tone}`} title={backendStatus.title}>
            <span className="chip-dot" aria-hidden="true" />
            {backendStatus.label}
          </span>
          <span>wide-angle first</span>
          <div className="theme-switcher" role="group" aria-label="Theme mode">
            {['light', 'dark', 'auto'].map((mode) => (
              <button
                key={mode}
                type="button"
                className={`theme-chip ${themeMode === mode ? 'active-theme-chip' : ''}`}
                onClick={() => setThemeMode(mode)}
              >
                {mode}
              </button>
            ))}
            <button type="button" className="theme-chip" onClick={handleResetSavedWorkspace}>
              Reset
            </button>
          </div>
        </div>
      </header>

      <section className="card app-switcher" aria-label="App workspace switcher">
        <button
          type="button"
          className={`switcher-tab ${appSpace === 'analysis' ? 'active-switcher-tab' : ''}`}
          onClick={() => setAppSpace('analysis')}
        >
          Analysis Workspace
        </button>
        <button
          type="button"
          className={`switcher-tab ${appSpace === 'training' ? 'active-switcher-tab' : ''}`}
          onClick={() => setAppSpace('training')}
        >
          Training Studio
          {activeDetectorIsCustom ? <span className="switcher-status-badge">custom detector active</span> : null}
        </button>
      </section>

      {appSpace === 'training' ? (
        <TrainingStudio
          apiBase={API_BASE}
          activeDetector={config.active_detector || 'soccana'}
          helpCatalog={config.help_catalog || []}
          onActiveDetectorChange={handleTrainingActiveDetectorChange}
        />
      ) : (
      <main
        ref={layoutRef}
        className={`layout-grid${workspaceMode === 'review' ? ' sidebar-hidden' : ''}${isResizingSidebar ? ' sidebar-resizing' : ''}`}
        style={workspaceMode === 'review' ? undefined : { '--analysis-sidebar-width': `${Math.round(effectiveAnalysisSidebarWidth)}px` }}
      >
        <aside className="left-sidebar">
          <section className="left-column">
            <form className="card form-card" onSubmit={handleSubmit}>
            <SectionTitleWithHelp title="Prepare an input clip" entry={helpIndex.get('analysis.prepare_input')} />
            <label>
              <span>Upload a video file</span>
              <input
                type="file"
                accept="video/*"
                onChange={(event) => {
                  resetLoadedSource();
                  updateForm('localVideoPath', '');
                  setSelectedFile(event.target.files?.[0] || null);
                }}
              />
            </label>
            <div className="muted">or use a local path on the same Mac</div>
            <label>
              <span>Local video path</span>
              <input
                type="text"
                value={form.localVideoPath}
                onChange={(event) => {
                  resetLoadedSource();
                  setSelectedFile(null);
                  updateForm('localVideoPath', event.target.value);
                }}
                placeholder="/Users/you/path/to/video.mp4"
              />
            </label>

            <label>
              <span>Optional label path</span>
              <input
                type="text"
                value={form.labelPath}
                onChange={(event) => updateForm('labelPath', event.target.value)}
                placeholder="/Users/you/path/to/Labels-v2.json"
              />
            </label>

            <label>
              <FieldLabel label="Detector weights" entry={helpIndex.get('analysis.detector_weights')} />
              <input
                list="detector-models"
                type="text"
                value={form.detectorModel}
                onChange={(event) => updateForm('detectorModel', event.target.value)}
              />
            </label>
            {activeDetectorIsCustom ? (
              <div className="active-detector-banner">
                <div className="micro-label">Active detector override</div>
                <div className="active-detector-name">{activeDetectorLabel}</div>
                <div className="field-note">
                  Live preview and analysis runs will use this activated checkpoint while the selector remains on `soccana`.
                </div>
              </div>
            ) : null}

            <label>
              <FieldLabel label="Player tracker" entry={helpIndex.get('analysis.player_tracker')} />
              <select value={form.trackerMode} onChange={(event) => updateForm('trackerMode', event.target.value)}>
                {playerTrackerModes.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>

            <datalist id="detector-models">
              {detectorModelOptions.map((item) => <option key={item} value={item} />)}
            </datalist>

            <div className="checkbox-row">
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  checked={form.includeBall}
                  onChange={(event) => updateForm('includeBall', event.target.checked)}
                />
                <span className="checkbox-label-row">
                  <span>Include ball tracking</span>
                  <HelpPopover entry={helpIndex.get('analysis.include_ball')} />
                </span>
              </label>
            </div>

            <div className="three-col">
              <label>
                <FieldLabel label="Player confidence" entry={helpIndex.get('analysis.player_conf')} />
                <input type="number" step="0.01" value={form.playerConf} onChange={(event) => updateForm('playerConf', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="Ball confidence" entry={helpIndex.get('analysis.ball_conf')} />
                <input type="number" step="0.01" value={form.ballConf} onChange={(event) => updateForm('ballConf', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="IOU" entry={helpIndex.get('analysis.iou')} />
                <input type="number" step="0.01" value={form.iou} onChange={(event) => updateForm('iou', event.target.value)} />
              </label>
            </div>

            <div className="inline-help-row">
              <span className="micro-label">Automatic field calibration</span>
              <HelpPopover entry={helpIndex.get('analysis.field_registration')} />
            </div>

            <button className="secondary-button" onClick={() => handleLoadSource()} type="button">
              {isLoadingSource ? 'Loading input clip...' : source ? 'Reload input clip' : 'Load input clip'}
            </button>
            {sourceError ? <div className="error-box">{sourceError}</div> : null}

            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Starting analysis...' : 'Analyze loaded clip'}
            </button>
            {jobError ? <div className="error-box">{jobError}</div> : null}
            </form>

            <section className="card form-card">
              <SectionTitleWithHelp title="SoccerNet" entry={helpIndex.get('analysis.soccernet')} />

              <div className="two-col">
                <label>
                  <span>Split</span>
                  <select value={soccerNetSplit} onChange={(event) => setSoccerNetSplit(event.target.value)}>
                    {(soccerNetConfig.splits || ['train', 'valid', 'test', 'challenge']).map((split) => (
                      <option key={split} value={split}>
                        {split} {soccerNetConfig.split_counts?.[split] ? `(${soccerNetConfig.split_counts[split]})` : ''}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Search games</span>
                  <input
                    type="text"
                    value={soccerNetQuery}
                    onChange={(event) => setSoccerNetQuery(event.target.value)}
                    placeholder="league / season / club"
                  />
                </label>
              </div>

              <div className="field-note">
                {soccerNetLoadingGames
                  ? 'Updating matches...'
                  : 'Matches update automatically as you change split or search. Long result sets stay collapsed until you open them.'}
              </div>

              <div className="soccernet-results-section">
                <div className="row-between soccernet-results-header">
                  <div className="soccernet-results-title">Matching games</div>
                  {canExpandSoccerNetResults && hasLongSoccerNetResultList ? (
                    <button
                      className="secondary-button compact-button"
                      type="button"
                      onClick={() => setSoccerNetResultsExpanded((current) => !current)}
                    >
                      {soccerNetResultsButtonLabel}
                    </button>
                  ) : null}
                </div>
                {shouldShowCollapsedSoccerNetPreview ? (
                  <button
                    type="button"
                    className="soccernet-results-preview"
                    onClick={() => setSoccerNetResultsExpanded(true)}
                  >
                    <div className="soccernet-results-preview-title">
                      {selectedSoccerNetGameMeta.match || 'Browse matching games'}
                    </div>
                    <div className="soccernet-results-preview-meta">
                      {[
                        selectedSoccerNetGameMeta.league,
                        selectedSoccerNetGameMeta.season,
                        selectedSoccerNetGameMeta.date,
                      ].filter(Boolean).join(' \u00b7 ') || 'Current selection'}
                    </div>
                    <div className="soccernet-results-preview-hint">
                      {soccerNetGamesCount} matches found. Open the list only when you want to change the selection.
                    </div>
                  </button>
                ) : (
                  <div className="game-browser">
                    <div className="game-list" role="listbox" aria-label="SoccerNet games">
                      {visibleSoccerNetGames.length === 0 ? (
                        <div className="muted">Load a split to browse games.</div>
                      ) : (
                        visibleSoccerNetGames.map((game) => {
                          const parsed = parseSoccerNetGame(game);
                          return (
                            <button
                              key={game}
                              type="button"
                              title={game}
                              className={`game-row ${soccerNetSelectedGame === game ? 'active-game-row' : ''}`}
                              aria-pressed={soccerNetSelectedGame === game}
                              onClick={() => {
                                setSoccerNetSelectedGame(game);
                                if (hasLongSoccerNetResultList) {
                                  setSoccerNetResultsExpanded(false);
                                }
                              }}
                            >
                              <div className="game-row-match">{parsed.match}</div>
                              {(parsed.league || parsed.season || parsed.date) ? (
                                <div className="game-row-meta">
                                  {[parsed.league, parsed.season, parsed.date].filter(Boolean).join(' \u00b7 ')}
                                </div>
                              ) : null}
                            </button>
                          );
                        })
                      )}
                    </div>
                  </div>
                )}
              </div>
              <div className="field-note">
                {soccerNetGamesCount ? `${soccerNetGamesCount} games matched this split/query.` : 'Load a split to browse games.'}
              </div>
              {soccerNetResultsExpanded && soccerNetGamesCount > visibleSoccerNetGames.length ? (
                <div className="source-toolbar">
                  <button className="secondary-button compact-button" type="button" onClick={() => setSoccerNetResultLimit((current) => current + 24)}>
                    Show more matches
                  </button>
                </div>
              ) : null}
              {soccerNetSelectedGame ? (
                <div className="selected-game-path">
                  <div className="selected-game-match">{selectedSoccerNetGameMeta.match}</div>
                  <div className="selected-game-detail">{soccerNetSelectedGame}</div>
                </div>
              ) : null}

              <label>
                <span>SoccerNet password</span>
                <input
                  type="password"
                  value={soccerNetPassword}
                  onChange={(event) => setSoccerNetPassword(event.target.value)}
                  placeholder="s0cc3rn3t"
                />
              </label>

              <div className="micro-label">Files to download</div>
              <div className="checkbox-row">
                {[...(soccerNetConfig.video_files || []), ...(soccerNetConfig.label_files || [])].map((fileName) => (
                  <label key={fileName} className="checkbox-item">
                    <input
                      type="checkbox"
                      checked={soccerNetFiles.includes(fileName)}
                      onChange={() => toggleSoccerNetFile(fileName)}
                    />
                    <span>{fileName}</span>
                  </label>
                ))}
              </div>
              <div className="field-note">
                Keep `Labels-v2.json` selected if you want the experimental signal attached to actual goal timestamps.
              </div>

              <div className="source-toolbar">
                <button className="secondary-button compact-button" onClick={handleSoccerNetDownload} type="button">
                  Download selected game
                </button>
                <button className="secondary-button compact-button" onClick={handleUseSoccerNetFolder} type="button">
                  Scan SoccerNet folder
                </button>
              </div>

              {soccerNetError ? <div className="error-box">{soccerNetError}</div> : null}
              {soccerNetDownloadJob ? (
                <div className="download-job">
                  <div className="row-between">
                    <div className="section-title">Download status</div>
                    <div className={`status-pill ${soccerNetDownloadJob.status || 'idle'}`}>{soccerNetDownloadJob.status || 'idle'}</div>
                  </div>
                  <div className="progress-shell">
                    <div className="progress-bar" style={{ width: `${soccerNetDownloadJob.progress || 0}%` }} />
                  </div>
                  <div className="muted">{soccerNetDownloadJob.game || 'No game selected'} · {soccerNetDownloadJob.progress || 0}%</div>
                  <div className="log-panel">
                    {(soccerNetDownloadJob.logs || ['No SoccerNet download yet.']).slice().reverse().map((line, index) => (
                      <div key={`${line}-${index}`}>{line}</div>
                    ))}
                  </div>
                </div>
              ) : null}
            </section>

            <section className="card form-card">
              <SectionTitleWithHelp title="Scan a local dataset folder" entry={helpIndex.get('analysis.dataset_scan')} />
              <label>
                <span>Folder path</span>
                <input
                  type="text"
                  value={form.folderPath}
                  onChange={(event) => updateForm('folderPath', event.target.value)}
                  placeholder="/Users/you/datasets/football"
                />
              </label>
              <button className="secondary-button" onClick={handleScanFolder} type="button">
                Scan folder
              </button>
              {scanError ? <div className="error-box">{scanError}</div> : null}
              {folderScan ? (
                <div className="scan-results">
                  <div className="micro-label">Videos</div>
                  <div className="path-list">
                    {folderScan.videos.length === 0 ? <div className="muted">No videos found.</div> : null}
                    {folderScan.videos.map((video) => (
                    <button
                      key={video.path}
                      type="button"
                      className="path-chip"
                      onClick={async () => {
                        resetLoadedSource();
                        setSelectedFile(null);
                        updateForm('localVideoPath', video.path);
                        await handleLoadSource(video.path);
                      }}
                    >
                      {video.name} · {video.size_mb} MB
                    </button>
                    ))}
                  </div>
                  <div className="micro-label">Possible labels or notes</div>
                  <div className="path-list">
                    {folderScan.annotations.slice(0, 20).map((item) => (
                      <button
                        key={item.path}
                        type="button"
                        className={`path-chip ${form.labelPath === item.path ? 'active-chip' : ''}`}
                        onClick={() => updateForm('labelPath', item.path)}
                      >
                        {basenameFromPath(item.path)}
                      </button>
                    ))}
                  </div>
                  {form.labelPath ? <div className="field-note">Selected label file: {form.labelPath}</div> : null}
                </div>
              ) : null}
            </section>
          </section>
        </aside>

        <div
          className="sidebar-resizer"
          role="separator"
          aria-label="Resize analysis sidebar"
          aria-orientation="vertical"
          aria-valuemin={MIN_ANALYSIS_SIDEBAR_WIDTH}
          aria-valuemax={Math.round(maxAnalysisSidebarWidth)}
          aria-valuenow={Math.round(effectiveAnalysisSidebarWidth)}
          tabIndex={workspaceMode === 'review' ? -1 : 0}
          onPointerDown={startSidebarResize}
          onDoubleClick={resetSidebarWidth}
          onKeyDown={handleSidebarResizerKeyDown}
        >
          <span className="sidebar-resizer-rail" aria-hidden="true" />
          <span className="sidebar-resizer-grip" aria-hidden="true" />
        </div>

        <section className="right-column">
          <section className="card workspace-shell">
            <div className="workspace-header">
              <div className="eyebrow">workspace</div>
              <div className="section-title">Analysis Workspace</div>
            </div>

            <div className="workspace-tabs">
              {WORKSPACE_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={`workspace-tab ${workspaceMode === mode.id ? 'active-workspace-tab' : ''}`}
                  onClick={() => setWorkspaceMode(mode.id)}
                  disabled={mode.id === 'job' && !hasActiveJob}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </section>

          {workspaceMode === 'input' ? (
            <section className="card video-card workspace-panel">
              <div className="row-between">
                <SectionTitleWithHelp title="Input Clip" entry={helpIndex.get('analysis.input_clip')} />
                <div className="muted">{source ? `${source.display_name} · ${source.width || 0}x${source.height || 0}` : 'No input clip loaded'}</div>
              </div>
              {source ? (
                <>
                  <div className="video-stage">
                    <video ref={sourceVideoRef} controls src={`${API_BASE}${source.video_url}`} className="video-player" />
                  </div>
                  <div className="workspace-actions">
                    <button className="secondary-button compact-button" type="button" onClick={() => handleLoadSource()}>
                      {isLoadingSource ? 'Loading input clip...' : 'Reload input clip'}
                    </button>
                    <button className="secondary-button compact-button" type="button" onClick={handleStartLivePreview}>
                      Open live preview
                    </button>
                    <button
                      className="primary-button compact-button workspace-primary-action"
                      type="button"
                      onClick={() => setWorkspaceMode('job')}
                      disabled={!hasActiveJob}
                    >
                      {hasActiveJob ? 'Go to active job' : 'No active job'}
                    </button>
                  </div>
                </>
              ) : (
                <div className="empty-card">Upload or paste a local path in the sidebar to load a clip.</div>
              )}
            </section>
          ) : null}

          {workspaceMode === 'live' ? (
            <section className="card video-card workspace-panel">
              <div className="row-between">
                <SectionTitleWithHelp title="Live Inference Preview" entry={helpIndex.get('analysis.live_preview')} />
                <div className="muted">{livePreviewUrl ? 'Streaming from backend' : 'Idle'}</div>
              </div>
              <div className="field-note">
                Preview source: {source ? `${source.display_name}` : 'No clip loaded'}
              </div>
              {activeExperiment ? (
                <div className="field-note">
                  Active experiment clip: {activeExperimentClipName || 'Unknown clip'}
                  {activeExperiment?.summary?.current_game ? ` · ${activeExperiment.summary.current_game}` : ''}
                  {activeExperiment?.summary?.current_half_file ? ` · ${activeExperiment.summary.current_half_file}` : ''}
                </div>
              ) : null}
              {activeExperimentSourcePath && !previewMatchesActiveExperiment ? (
                <div className="error-box">
                  Live preview is showing a different clip than the active experiment.
                </div>
              ) : null}
              <div className="workspace-actions">
                <button className="secondary-button compact-button" type="button" onClick={handleStartLivePreview} disabled={!source?.source_id}>
                  Start live preview
                </button>
                <button className="secondary-button compact-button" type="button" onClick={handleStopLivePreview} disabled={!livePreviewUrl}>
                  Stop live preview
                </button>
                <button
                  className="secondary-button compact-button"
                  type="button"
                  onClick={handleUseActiveExperimentClip}
                  disabled={!activeExperimentSourcePath}
                >
                  Use active experiment clip
                </button>
                <button
                  className="primary-button compact-button workspace-primary-action"
                  type="button"
                  onClick={() => setWorkspaceMode('job')}
                  disabled={!hasActiveJob}
                >
                  {hasActiveJob ? 'Go to active job' : 'No active job'}
                </button>
              </div>
              {livePreviewUrl ? (
                <img src={livePreviewUrl} alt="Live model preview" className="video-player live-preview-frame" />
              ) : (
                <div className="empty-card">Start live preview after loading an input clip. Completed analysis runs create the saved review output.</div>
              )}
            </section>
          ) : null}

          {workspaceMode === 'job' ? (
            <section className="card run-card workspace-panel">
              <div className="row-between">
                <div className="section-title">Current Analysis Job</div>
                <div className={`status-pill ${currentJob?.status || 'idle'}`}>{currentJob?.status || 'idle'}</div>
              </div>
              {currentJob ? (
                <>
                  <div className="progress-shell">
                    <div className="progress-bar" style={{ width: `${currentJob?.progress || 0}%` }} />
                  </div>
                  <div className="muted">{`${currentJob.progress || 0}% complete`}</div>
                  <div className="log-panel">
                    {(currentJob.logs || ['Logs will appear here.']).slice().reverse().map((line, index) => (
                      <div key={`${line}-${index}`}>{line}</div>
                    ))}
                  </div>
                  <div className="workspace-actions">
                    <button className="secondary-button compact-button" type="button" onClick={() => setWorkspaceMode('input')}>
                      Back to input
                    </button>
                    {canPauseJob ? (
                      <button
                        className="secondary-button compact-button"
                        type="button"
                        onClick={() => handleJobAction('pause')}
                        disabled={jobActionPending !== ''}
                      >
                        {jobActionPending === 'pause' ? 'Pausing...' : 'Pause'}
                      </button>
                    ) : null}
                    {canResumeJob ? (
                      <button
                        className="secondary-button compact-button"
                        type="button"
                        onClick={() => handleJobAction('resume')}
                        disabled={jobActionPending !== ''}
                      >
                        {jobActionPending === 'resume' ? 'Resuming...' : 'Resume'}
                      </button>
                    ) : null}
                    {canStopJob ? (
                      <button
                        className="secondary-button compact-button"
                        type="button"
                        onClick={() => handleJobAction('stop')}
                        disabled={jobActionPending !== '' || currentJob.status === 'stopping'}
                      >
                        {currentJob.status === 'stopping' || jobActionPending === 'stop' ? 'Stopping...' : 'Stop'}
                      </button>
                    ) : null}
                    {currentJob.status === 'completed' ? (
                      <button className="primary-button compact-button workspace-primary-action" type="button" onClick={() => { setWorkspaceMode('review'); setReviewPanel('overview'); }}>
                        Review completed run
                      </button>
                    ) : null}
                  </div>
                  {jobError ? <div className="error-box">{jobError}</div> : null}
                </>
              ) : (
                <div className="empty-card">No analysis running. Load a clip, then click Analyze.</div>
              )}
            </section>
          ) : null}

          {workspaceMode === 'review' ? (
            <>
              <section className="card workspace-panel">
                <div className="row-between">
                  <SectionTitleWithHelp title="Saved Run Review" entry={helpIndex.get('analysis.saved_run_review')} />
                  <div className="muted">{recentRuns.length ? `${recentRuns.length} found on disk` : 'None yet'}</div>
                </div>
                <div className="field-note">
                  Select one saved run. All review content below belongs to that run only.
                </div>
                {recentRunsError ? <div className="error-box">{recentRunsError}</div> : null}
                {reviewError ? <div className="error-box">{reviewError}</div> : null}
                {recentRuns.length > 0 ? (
                  <label>
                    <span>Saved runs</span>
                    <select
                      value={reviewedRun?.run_id || ''}
                      onChange={(event) => {
                        if (event.target.value) {
                          handleLoadRun(event.target.value);
                        }
                      }}
                    >
                      <option value="" disabled>Select a completed run</option>
                      {recentRuns.map((run) => (
                        <option key={run.run_id} value={run.run_id}>
                          {`${run.run_id} · ${basenameFromPath(run.summary?.input_video)} · ${run.summary?.frames_processed || 0} frames`}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : (
                  <div className="muted">Completed runs with a `summary.json` will appear here automatically.</div>
                )}
                {reviewedRun ? (
                  <>
                    <div className="run-review-meta">
                      <div><span className="micro-label">Run</span><div>{reviewedRunId}</div></div>
                      <div><span className="micro-label">Clip</span><div>{reviewedClipName || 'Unknown source'}</div></div>
                      <div><span className="micro-label">Status</span><div>{reviewedRun.status || 'completed'}</div></div>
                      <div><span className="micro-label">Frames</span><div>{summary?.frames_processed || 0}</div></div>
                    </div>
                    <div className="review-tabs">
                      {REVIEW_PANELS.map((panel) => (
                        <button
                          key={panel.id}
                          type="button"
                          className={`review-tab ${reviewPanel === panel.id ? 'active-review-tab' : ''}`}
                          onClick={() => setReviewPanel(panel.id)}
                        >
                          {panel.label}
                        </button>
                      ))}
                    </div>
                  </>
                ) : null}
              </section>

              {!reviewedRun ? (
                <div className="card empty-card workspace-panel">Select a saved run to open the review workspace.</div>
              ) : null}

              {reviewedRun && reviewPanel === 'overview' ? (
                <>
                  <section className="review-overview-stack workspace-panel">
                    <section className="card review-hero-card">
                      <div className="review-hero-header">
                        <div className="review-hero-copy">
                          <div className="eyebrow">Saved overlay</div>
                          <div className="section-title review-hero-title">Overlay Playback</div>
                          <div className="review-hero-subtitle">{reviewedClipName || 'Unknown clip'}</div>
                          <div className="review-hero-meta">
                            <div className="review-hero-detail">
                              <div className="micro-label">Run</div>
                              <div className="review-hero-detail-value">{reviewedRunId || 'Unknown run'}</div>
                            </div>
                            <div className="review-hero-detail">
                              <div className="micro-label">Frames</div>
                              <div className="review-hero-detail-value">{summary?.frames_processed || 0}</div>
                            </div>
                            <div className="review-hero-detail">
                              <div className="micro-label">Tracker</div>
                              <div className="review-hero-detail-value">{summary?.player_tracker_mode || 'n/a'}</div>
                            </div>
                          </div>
                        </div>
                        <div className="review-hero-actions">
                          <div className={`review-hero-status-panel ${summary?.homography_enabled ? 'good' : 'warn'}`}>
                            <div className="micro-label">Calibration state</div>
                            <div className="review-hero-status-value">
                              {summary?.homography_enabled ? 'Auto-calibrated' : 'Image-space only'}
                            </div>
                          </div>
                          <button
                            className="secondary-button compact-button"
                            type="button"
                            onClick={handleRefreshDiagnostics}
                            disabled={!reviewedRunId || isRefreshingDiagnostics}
                          >
                            {isRefreshingDiagnostics ? (summary?.diagnostics_stale ? 'Regenerating...' : 'Refreshing...') : (summary?.diagnostics_stale ? 'Regenerate' : 'Refresh')}
                          </button>
                        </div>
                      </div>
                      {summary?.overlay_video ? (
                        <div className="review-video-stage">
                          <video
                            ref={reviewVideoRef}
                            controls
                            src={`${API_BASE}${summary.overlay_video}`}
                            className="video-player review-video-player"
                            onLoadedMetadata={(event) => setReviewPlaybackTime(event.currentTarget.currentTime || 0)}
                            onTimeUpdate={(event) => setReviewPlaybackTime(event.currentTarget.currentTime || 0)}
                            onSeeked={(event) => setReviewPlaybackTime(event.currentTarget.currentTime || 0)}
                          />
                        </div>
                      ) : (
                        <div className="empty-card">No overlay output is available for the selected run.</div>
                      )}
                      <div className="review-summary-strip">
                        {reviewQuickFacts.map(([label, value]) => (
                          <div key={label} className={`review-summary-fact ${label === 'Clip' ? 'wide' : ''}`}>
                            <div className="micro-label">{label}</div>
                            <div className="review-summary-value">{value}</div>
                          </div>
                        ))}
                      </div>
                    </section>

                  <section className="card review-brief-card">
                      <div className="section-title">Run Brief</div>
                      {summary?.diagnostics_summary_line ? (
                        <div className="workspace-summary-line">{summary.diagnostics_summary_line}</div>
                      ) : null}
                      <div className="diagnostics-meta">
                        {summary.diagnostics_source === 'ai'
                          ? `AI-curated for this run via ${friendlyProviderName(summary.diagnostics_provider)}${summary.diagnostics_model ? ` · ${summary.diagnostics_model}` : ''}.`
                          : 'Heuristic run diagnostics are showing for this run.'}
                        {summary.diagnostics_error ? ` Last generation error: ${summary.diagnostics_error}` : ''}
                      </div>
                      {summary?.diagnostics_stale ? (
                        <div className="error-box">
                          {summary.diagnostics_stale_reason || 'Stored AI diagnostics are outdated.'} Use `Regenerate` to rebuild them.
                        </div>
                      ) : null}
                      {headlineDiagnostics.length > 0 ? (
                        <>
                          <div className="review-brief-note">Top findings stay on the same page as the overlay. The full drilldown remains below.</div>
                          <div className="headline-diagnostic-grid">
                            {headlineDiagnostics.map((item) => (
                              <HeadlineDiagnosticCard key={item.title} item={item} />
                            ))}
                          </div>
                        </>
                      ) : (
                        <div className="empty-card">No diagnostics available for the selected run.</div>
                      )}
                    </section>
                  </section>

                  <section className="workspace-panel">
                    <TrajectoryPanel
                      projectionState={projectionState}
                      selectedTrackIds={selectedTrajectoryTrackIds}
                      rankedTracks={rankedTrajectoryTracks}
                      currentFrame={reviewPlaybackFrame}
                      fps={summary?.fps}
                      onToggleTrack={toggleTrajectoryTrack}
                      onResetSelection={resetTrajectorySelection}
                    />
                  </section>

                  <section className="workspace-panel">
                    <div className="section-title inset-title">Run Metrics</div>
                    {runMetricSections.length === 0 ? (
                      <div className="card empty-card">Select a saved run to populate the tactical summary cards.</div>
                    ) : (
                      <div className="metric-group-grid">
                        {runMetricSections.map((section) => (
                          <MetricGroup key={section.title} title={section.title} items={section.items} helpIndex={helpIndex} />
                        ))}
                      </div>
                    )}
                  </section>

                  <section className="workspace-panel">
                    <div className="section-title inset-title">Experimental Signals</div>
                    {summary?.experiments?.length ? (
                      <div className="experiment-grid">
                        {summary.experiments.map((item) => <ExperimentCard key={item.id || item.title} item={item} />)}
                      </div>
                    ) : (
                      <div className="card empty-card">No experiments attached to the selected run yet.</div>
                    )}
                  </section>

                  <section className="workspace-panel diagnostics-stack">
                    <div className="section-title inset-title">Detailed Diagnostics</div>
                    {summary?.diagnostics?.length ? summary.diagnostics.map((item) => <DiagnosticCard key={item.title} item={item} />) : <div className="card empty-card">No diagnostics available for the selected run.</div>}
                  </section>

                  <section className="workspace-panel">
                    <PromptContextPanel summary={summary} />
                  </section>
                </>
              ) : null}

              {reviewedRun && reviewPanel === 'tracks' ? (
                <section className="workspace-panel">
                  <TrackTable tracks={reviewTrackRows} />
                </section>
              ) : null}

              {reviewedRun && reviewPanel === 'files' ? (
                <section className="workspace-panel">
                  <FileLinks summary={summary} />
                </section>
              ) : null}
            </>
          ) : null}
        </section>
      </main>
      )}
    </div>
  );
}
