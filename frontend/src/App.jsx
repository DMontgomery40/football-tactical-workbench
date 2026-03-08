import { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8431';

const defaultForm = {
  localVideoPath: '',
  folderPath: '',
  detectorModel: 'soccana',
  includeBall: true,
  playerConf: '0.25',
  ballConf: '0.20',
  iou: '0.50',
};

const STORAGE_KEYS = {
  themeMode: 'fpw.themeMode',
  form: 'fpw.form',
  soccerNetSplit: 'fpw.soccerNetSplit',
  soccerNetQuery: 'fpw.soccerNetQuery',
  soccerNetFiles: 'fpw.soccerNetFiles',
  folderPath: 'fpw.folderPath',
  localVideoPath: 'fpw.localVideoPath',
};

const WORKSPACE_MODES = [
  { id: 'input', label: 'Input', hint: 'Raw clip before inference' },
  { id: 'live', label: 'Live Preview', hint: 'Ephemeral backend stream' },
  { id: 'job', label: 'Active Job', hint: 'Current analysis progress' },
  { id: 'review', label: 'Run Review', hint: 'Saved overlay and outputs' },
];

const REVIEW_PANELS = [
  { id: 'overview', label: 'Overview' },
  { id: 'tracks', label: 'Tracks' },
  { id: 'files', label: 'Files' },
];

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
  return String(value).split('/').filter(Boolean).pop() || String(value);
}

function friendlyProviderName(value) {
  const provider = String(value || '').toLowerCase();
  if (provider === 'openai') return 'OpenAI';
  if (provider === 'openrouter') return 'OpenRouter';
  if (provider === 'anthropic') return 'Anthropic';
  if (provider === 'local') return 'Local';
  return value || 'provider';
}

function StatCard({ label, value, hint }) {
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint ? <div className="stat-hint">{hint}</div> : null}
    </div>
  );
}

function DiagnosticCard({ item }) {
  return (
    <div className={`card diagnostic ${item.level === 'warn' ? 'warn' : 'good'}`}>
      <div className="diagnostic-title">{item.title}</div>
      <p>{item.message}</p>
      <div className="diagnostic-next">Next action: {item.next_step}</div>
    </div>
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
              <tr key={track.track_id}>
                <td>{track.track_id}</td>
                <td>{track.team_label || 'unassigned'}</td>
                <td>{track.team_vote_ratio ?? '0.0'}</td>
                <td>{track.frames}</td>
                <td>{track.first_frame}</td>
                <td>{track.last_frame}</td>
                <td>{track.average_confidence ?? 'n/a'}</td>
                <td>{track.projected_points ?? 'n/a'}</td>
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

export default function App() {
  const [themeMode, setThemeMode] = useState(() => readStoredString(STORAGE_KEYS.themeMode, 'light'));
  const [config, setConfig] = useState({ player_models: [], ball_models: [], learn_cards: [] });
  const [soccerNetConfig, setSoccerNetConfig] = useState({ dataset_dir: '', splits: [], split_counts: {}, video_files: [], label_files: [], notes: [] });
  const [soccerNetSplit, setSoccerNetSplit] = useState(() => readStoredString(STORAGE_KEYS.soccerNetSplit, 'train'));
  const [soccerNetQuery, setSoccerNetQuery] = useState(() => readStoredString(STORAGE_KEYS.soccerNetQuery, ''));
  const [soccerNetGames, setSoccerNetGames] = useState([]);
  const [soccerNetGamesCount, setSoccerNetGamesCount] = useState(0);
  const [soccerNetResultLimit, setSoccerNetResultLimit] = useState(24);
  const [soccerNetSelectedGame, setSoccerNetSelectedGame] = useState('');
  const [soccerNetPassword, setSoccerNetPassword] = useState('');
  const [soccerNetFiles, setSoccerNetFiles] = useState(() => {
    const stored = readStoredJson(STORAGE_KEYS.soccerNetFiles, null);
    return Array.isArray(stored) && stored.length > 0 ? stored : ['1_720p.mkv', '2_720p.mkv', 'Labels-v2.json'];
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
  const [job, setJob] = useState(null);
  const [jobError, setJobError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [workspaceMode, setWorkspaceMode] = useState('input');
  const [reviewPanel, setReviewPanel] = useState('overview');
  const pollRef = useRef(null);
  const soccerNetPollRef = useRef(null);
  const sourceVideoRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then((response) => response.json())
      .then((data) => setConfig(data))
      .catch((error) => {
        console.error(error);
        setJobError('Could not reach the backend. Start backend first.');
      });
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/soccernet/config`)
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

  const currentJob = job;
  const reviewedRun = selectedRun;
  const summary = reviewedRun?.summary || null;
  const sourceLabel = source
    ? source.display_name || basenameFromPath(source.video_url)
    : selectedFile?.name || basenameFromPath(form.localVideoPath) || '';
  const reviewedRunId = reviewedRun?.run_id || '';
  const reviewedClipName = basenameFromPath(summary?.input_video);
  const detectorModelOptions = config.detector_models?.length ? config.detector_models : config.player_models || [];
  const visibleSoccerNetGames = useMemo(
    () => soccerNetGames.slice(0, soccerNetResultLimit),
    [soccerNetGames, soccerNetResultLimit],
  );
  const headlineDiagnostics = useMemo(
    () => (summary?.diagnostics || []).slice(0, 4),
    [summary],
  );
  const reviewQuickFacts = useMemo(() => {
    if (!summary) return [];
    return [
      ['Clip', reviewedClipName || 'Unknown'],
      ['Frames', summary.frames_processed || 0],
      ['Calibration', `${summary.field_calibration_refresh_successes || 0}/${summary.field_calibration_refresh_attempts || 0}`],
      ['Player IDs', summary.unique_player_track_ids || 0],
      ['Teams', `${summary.home_tracks || 0} home / ${summary.away_tracks || 0} away`],
      ['Ball / frame', summary.average_ball_detections_per_frame ?? 'n/a'],
    ];
  }, [summary, reviewedClipName]);

  const runStats = useMemo(() => {
    if (!summary) return [];
    return [
      ['Frames', summary.frames_processed || 0, 'Decoded frames pushed through detection and tracking'],
      ['Player track IDs', summary.unique_player_track_ids || 0, 'Tracker stability is more important than raw box count'],
      ['Home tracks', summary.home_tracks || 0, 'Unsupervised jersey-color split, not official metadata'],
      ['Away tracks', summary.away_tracks || 0, 'Should roughly match the second main kit cluster'],
      ['Projected player anchors', summary.projected_player_points || 0, 'Per-frame player anchor samples that landed on the pitch map, not unique players'],
      ['Avg pitch keypoints', summary.average_visible_pitch_keypoints || 0, 'Visible field keypoints on each 10-frame calibration refresh'],
    ];
  }, [summary]);

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function scanFolderPath(folderPath) {
    setScanError('');
    setFolderScan(null);
    const response = await fetch(`${API_BASE}/api/scan-folder`, {
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
      const response = await fetch(`${API_BASE}/api/soccernet/games?${params.toString()}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load SoccerNet games');
      }
      setSoccerNetGames(data.games || []);
      setSoccerNetGamesCount(data.count || 0);
      setSoccerNetResultLimit(24);
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
        const response = await fetch(`${API_BASE}/api/soccernet/downloads/${jobId}`);
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
      const response = await fetch(`${API_BASE}/api/soccernet/download`, {
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

  async function handleLoadSource(overridePath = null) {
    setSourceError('');
    setIsLoadingSource(true);
    setLivePreviewUrl('');
    try {
      const payload = new FormData();
      if (!overridePath && selectedFile) {
        payload.append('video_file', selectedFile);
      }
      payload.append('local_video_path', overridePath ?? form.localVideoPath);

      const response = await fetch(`${API_BASE}/api/source`, {
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

  function resetLoadedSource() {
    setSource(null);
    setLivePreviewUrl('');
  }

  async function loadRecentRuns() {
    setRecentRunsError('');
    try {
      const response = await fetch(`${API_BASE}/api/runs/recent?limit=6`);
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
      const response = await fetch(`${API_BASE}/api/runs/${runId}`);
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
      const response = await fetch(`${API_BASE}/api/runs/${reviewedRunId}/refresh-diagnostics`, {
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

    const payload = new FormData();
    if (source?.source_id) {
      payload.append('source_id', source.source_id);
    } else if (selectedFile) {
      payload.append('video_file', selectedFile);
    }
    payload.append('local_video_path', source?.source_id ? '' : form.localVideoPath);
    payload.append('detector_model', form.detectorModel);
    payload.append('include_ball', String(form.includeBall));
    payload.append('player_conf', form.playerConf);
    payload.append('ball_conf', form.ballConf);
    payload.append('iou', form.iou);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
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

  function startPolling(jobId) {
    clearPolling();

    async function tick() {
      try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        const data = await response.json();
        setJob(data);
        if (data.status === 'completed' || data.status === 'failed') {
          clearPolling();
          if (data.status === 'completed' && data.summary) {
            setSelectedRun(data);
            setWorkspaceMode('review');
            setReviewPanel('overview');
          }
          loadRecentRuns();
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
      <div className="notice-strip">
        <div className="notice-label">Pre-launch workbench</div>
        <div className="notice-text">
          Wide-angle tracking and experiment workflows are being refined against real clips and SoccerNet data before a public release.
        </div>
      </div>

      <header className="hero card">
        <div>
          <div className="eyebrow">football tactical demo</div>
          <h1>Detect, track, calibrate the field, and project the play.</h1>
          <p>
            Local React + FastAPI demo for wide-angle football analysis with football-specific detection, automatic field-keypoint calibration every 10 frames, live model preview, and projected tactical overlays.
          </p>
        </div>
        <div className="hero-pills">
          <span>frontend 4317</span>
          <span>backend 8431</span>
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
          </div>
        </div>
      </header>

      <main className="layout-grid">
        <aside className="left-sidebar">
          <section className="left-column">
            <form className="card form-card" onSubmit={handleSubmit}>
            <div className="section-title">Prepare an input clip</div>
            <label>
              <span>Upload a video file</span>
              <input
                type="file"
                accept="video/*"
                onChange={(event) => {
                  resetLoadedSource();
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
                  updateForm('localVideoPath', event.target.value);
                }}
                placeholder="/Users/you/path/to/video.mp4"
              />
            </label>

            <label>
              <span>Detector weights</span>
              <input
                list="detector-models"
                type="text"
                value={form.detectorModel}
                onChange={(event) => updateForm('detectorModel', event.target.value)}
              />
            </label>
            <div className="field-note">
              `soccana` is the default detector for players, ball, and referees.
            </div>

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
                <span>Include ball tracking</span>
              </label>
            </div>

            <div className="three-col">
              <label>
                <span>Player confidence</span>
                <input type="number" step="0.01" value={form.playerConf} onChange={(event) => updateForm('playerConf', event.target.value)} />
              </label>
              <label>
                <span>Ball confidence</span>
                <input type="number" step="0.01" value={form.ballConf} onChange={(event) => updateForm('ballConf', event.target.value)} />
              </label>
              <label>
                <span>IOU</span>
                <input type="number" step="0.01" value={form.iou} onChange={(event) => updateForm('iou', event.target.value)} />
              </label>
            </div>

            <div className="field-note">
              Field calibration is automatic now. The backend refreshes the pitch transform every 10 frames from pitch keypoints, so there is no manual homography step in the UI.
            </div>

            <button className="secondary-button" onClick={handleLoadSource} type="button">
              {isLoadingSource ? 'Loading input clip...' : source ? 'Reload input clip' : 'Load input clip'}
            </button>
            {sourceError ? <div className="error-box">{sourceError}</div> : null}

            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Starting analysis...' : 'Analyze loaded clip'}
            </button>
            {jobError ? <div className="error-box">{jobError}</div> : null}
            </form>

            <section className="card form-card">
              <div className="section-title">SoccerNet</div>
              <div className="field-note">
                Browse official SoccerNet games, download the files you want into the local dataset folder, then scan that folder in the same UI.
              </div>

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
                  : 'Matches update automatically as you change split or search.'}
              </div>

              <label>
                <span>Matching games</span>
                <div className="game-browser">
                  <div className="game-list" role="listbox" aria-label="SoccerNet games">
                    {visibleSoccerNetGames.length === 0 ? (
                      <div className="muted">Load a split to browse games.</div>
                    ) : (
                      visibleSoccerNetGames.map((game) => (
                        <button
                          key={game}
                          type="button"
                          title={game}
                          className={`game-row ${soccerNetSelectedGame === game ? 'active-game-row' : ''}`}
                          aria-pressed={soccerNetSelectedGame === game}
                          onClick={() => setSoccerNetSelectedGame(game)}
                        >
                          {game}
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </label>
              <div className="field-note">
                {soccerNetGamesCount ? `${soccerNetGamesCount} games matched this split/query.` : 'Load a split to browse games.'}
              </div>
              {soccerNetGamesCount > visibleSoccerNetGames.length ? (
                <div className="source-toolbar">
                  <button className="secondary-button compact-button" type="button" onClick={() => setSoccerNetResultLimit((current) => current + 24)}>
                    Show more matches
                  </button>
                </div>
              ) : null}
              {soccerNetSelectedGame ? (
                <div className="selected-game-path">{soccerNetSelectedGame}</div>
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
              <div className="section-title">Scan a local dataset folder</div>
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
                        updateForm('localVideoPath', video.path);
                        await handleLoadSource(video.path);
                      }}
                    >
                      {video.name} · {video.size_mb} MB
                    </button>
                    ))}
                  </div>
                  <div className="micro-label">Possible labels or notes</div>
                  <div className="path-list plain-text-list">
                    {folderScan.annotations.slice(0, 20).map((item) => (
                      <div key={item.path}>{item.path}</div>
                    ))}
                  </div>
                </div>
              ) : null}
            </section>
          </section>
        </aside>

        <section className="right-column">
          <section className="card workspace-shell">
            <div className="workspace-header">
              <div>
                <div className="eyebrow">workspace</div>
                <div className="section-title">Analysis Workspace</div>
                <div className="field-note workspace-subtitle">
                  The right side is mode-driven. Input clip, live preview, active job, and saved run review are separate states.
                </div>
              </div>
              <div className="workspace-snapshot-grid">
                <div className={`snapshot-card ${workspaceMode === 'input' ? 'active-snapshot' : ''}`}>
                  <span className="micro-label">Input</span>
                  <span className="snapshot-value">{sourceLabel || 'None loaded'}</span>
                </div>
                <div className={`snapshot-card ${workspaceMode === 'job' ? 'active-snapshot' : ''}`}>
                  <span className="micro-label">Job</span>
                  <span className="snapshot-value">{currentJob?.status || 'idle'}</span>
                </div>
                <div className={`snapshot-card ${workspaceMode === 'review' ? 'active-snapshot' : ''}`}>
                  <span className="micro-label">Reviewed run</span>
                  <span className="snapshot-value">{reviewedRunId || 'None selected'}</span>
                </div>
              </div>
            </div>

            <div className="workspace-tabs">
              {WORKSPACE_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={`workspace-tab ${workspaceMode === mode.id ? 'active-workspace-tab' : ''}`}
                  onClick={() => setWorkspaceMode(mode.id)}
                >
                  <span>{mode.label}</span>
                  <span className="workspace-tab-hint">{mode.hint}</span>
                </button>
              ))}
            </div>
          </section>

          {workspaceMode === 'input' ? (
            <section className="card video-card workspace-panel">
              <div className="row-between">
                <div className="section-title">Input Clip</div>
                <div className="muted">{source ? `${source.display_name} · ${source.width || 0}x${source.height || 0}` : 'No input clip loaded'}</div>
              </div>
              {source ? (
                <>
                  <div className="video-stage">
                    <video ref={sourceVideoRef} controls src={`${API_BASE}${source.video_url}`} className="video-player" />
                  </div>
                  <div className="workspace-actions">
                    <button className="secondary-button compact-button" type="button" onClick={handleLoadSource}>
                      {isLoadingSource ? 'Loading input clip...' : 'Reload input clip'}
                    </button>
                    <button className="secondary-button compact-button" type="button" onClick={handleStartLivePreview}>
                      Open live preview
                    </button>
                    <button className="primary-button compact-button workspace-primary-action" type="button" onClick={() => setWorkspaceMode('job')}>
                      Go to active job
                    </button>
                  </div>
                  <div className="field-note">
                    This panel is only the raw input video. It is separate from the saved overlay produced by analysis.
                  </div>
                </>
              ) : (
                <div className="empty-card">Load an input clip from the left sidebar. Nothing on the run-review side changes until you explicitly load or complete a run.</div>
              )}
            </section>
          ) : null}

          {workspaceMode === 'live' ? (
            <section className="card video-card workspace-panel">
              <div className="row-between">
                <div className="section-title">Live Inference Preview</div>
                <div className="muted">{livePreviewUrl ? 'Streaming from backend' : 'Idle'}</div>
              </div>
              <div className="workspace-actions">
                <button className="secondary-button compact-button" type="button" onClick={handleStartLivePreview} disabled={!source?.source_id}>
                  Start live preview
                </button>
                <button className="secondary-button compact-button" type="button" onClick={handleStopLivePreview} disabled={!livePreviewUrl}>
                  Stop live preview
                </button>
                <button className="primary-button compact-button workspace-primary-action" type="button" onClick={() => setWorkspaceMode('job')}>
                  Go to active job
                </button>
              </div>
              {livePreviewUrl ? (
                <img src={livePreviewUrl} alt="Live model preview" className="video-player live-preview-frame" />
              ) : (
                <div className="empty-card">Start live preview after loading an input clip. This stream is ephemeral and is not the saved review output.</div>
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
                    {currentJob.status === 'completed' ? (
                      <button className="primary-button compact-button workspace-primary-action" type="button" onClick={() => { setWorkspaceMode('review'); setReviewPanel('overview'); }}>
                        Review completed run
                      </button>
                    ) : null}
                  </div>
                </>
              ) : (
                <div className="empty-card">No analysis job has been started from this browser session yet. Loading an input clip does not create a run.</div>
              )}
            </section>
          ) : null}

          {workspaceMode === 'review' ? (
            <>
              <section className="card workspace-panel">
                <div className="row-between">
                  <div className="section-title">Saved Run Review</div>
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
                  <section className="review-overview-grid workspace-panel">
                    <div className="review-main-column">
                      <section className="card video-card">
                        <div className="row-between">
                          <div className="section-title">Overlay Playback</div>
                          <div className="muted">
                            {summary?.homography_enabled ? 'Auto-calibrated' : 'Image-space only'}
                          </div>
                        </div>
                        {summary?.overlay_video ? (
                          <video controls src={`${API_BASE}${summary.overlay_video}`} className="video-player" />
                        ) : (
                          <div className="empty-card">No overlay output is available for the selected run.</div>
                        )}
                      </section>
                    </div>

                    <div className="review-side-column">
                      <section className="card overview-brief-card">
                        <div className="row-between">
                          <div className="section-title inset-title">Run Brief</div>
                          <button
                            className="secondary-button compact-button"
                            type="button"
                            onClick={handleRefreshDiagnostics}
                            disabled={!reviewedRunId || isRefreshingDiagnostics}
                          >
                            {isRefreshingDiagnostics ? 'Refreshing...' : 'Refresh'}
                          </button>
                        </div>
                        {summary?.diagnostics_summary_line ? (
                          <div className="workspace-summary-line">{summary.diagnostics_summary_line}</div>
                        ) : null}
                        <div className="diagnostics-meta">
                          {summary.diagnostics_source === 'ai'
                            ? `Model-generated via ${friendlyProviderName(summary.diagnostics_provider)}${summary.diagnostics_model ? ` · ${summary.diagnostics_model}` : ''}.`
                            : 'Fallback diagnostics are showing because model generation is unavailable for this run.'}
                          {summary.diagnostics_error ? ` Last generation error: ${summary.diagnostics_error}` : ''}
                        </div>
                        <div className="overview-facts-grid">
                          {reviewQuickFacts.map(([label, value]) => (
                            <div key={label} className="overview-fact">
                              <div className="micro-label">{label}</div>
                              <div className="overview-fact-value">{value}</div>
                            </div>
                          ))}
                        </div>
                        <div className="overview-callouts">
                          {headlineDiagnostics.length > 0 ? headlineDiagnostics.map((item) => (
                            <div key={item.title} className={`overview-callout ${item.level === 'warn' ? 'warn' : 'good'}`}>
                              <div className="overview-callout-title">{item.title}</div>
                              <div className="overview-callout-text">{item.message}</div>
                            </div>
                          )) : (
                            <div className="empty-card">No diagnostics available for the selected run.</div>
                          )}
                        </div>
                      </section>
                    </div>
                  </section>

                  <section className="summary-grid workspace-panel">
                    <div className="section-title inset-title summary-title-span">Run Metrics</div>
                    {runStats.length === 0 ? (
                      <div className="card empty-card">Select a saved run to populate the tactical summary cards.</div>
                    ) : runStats.map(([label, value, hint]) => (
                      <StatCard key={label} label={label} value={value} hint={hint} />
                    ))}
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
                </>
              ) : null}

              {reviewedRun && reviewPanel === 'tracks' ? (
                <section className="workspace-panel">
                  <TrackTable tracks={summary?.top_tracks || []} />
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
    </div>
  );
}
