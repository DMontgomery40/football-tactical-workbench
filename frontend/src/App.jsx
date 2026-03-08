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
      <div className="diagnostic-next">Next: {item.next_step}</div>
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
    return <div className="card empty-card">No track preview yet.</div>;
  }

  return (
    <div className="card table-card">
      <div className="section-title">Top tracks</div>
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
    ['Summary JSON', summary?.summary_json],
    ['All outputs zip', summary?.all_outputs_zip],
  ].filter((entry) => entry[1]);

  if (entries.length === 0) {
    return <div className="card empty-card">Output links will appear here after a run finishes.</div>;
  }

  return (
    <div className="card link-card">
      <div className="section-title">Output files</div>
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
  const [config, setConfig] = useState({ player_models: [], ball_models: [], learn_cards: [] });
  const [soccerNetConfig, setSoccerNetConfig] = useState({ dataset_dir: '', splits: [], split_counts: {}, video_files: [], label_files: [], notes: [] });
  const [soccerNetSplit, setSoccerNetSplit] = useState('train');
  const [soccerNetQuery, setSoccerNetQuery] = useState('');
  const [soccerNetGames, setSoccerNetGames] = useState([]);
  const [soccerNetGamesCount, setSoccerNetGamesCount] = useState(0);
  const [soccerNetSelectedGame, setSoccerNetSelectedGame] = useState('');
  const [soccerNetPassword, setSoccerNetPassword] = useState('');
  const [soccerNetFiles, setSoccerNetFiles] = useState(['1_720p.mkv', '2_720p.mkv', 'Labels-v2.json']);
  const [soccerNetError, setSoccerNetError] = useState('');
  const [soccerNetLoadingGames, setSoccerNetLoadingGames] = useState(false);
  const [soccerNetDownloadJob, setSoccerNetDownloadJob] = useState(null);
  const [form, setForm] = useState(defaultForm);
  const [selectedFile, setSelectedFile] = useState(null);
  const [source, setSource] = useState(null);
  const [sourceError, setSourceError] = useState('');
  const [isLoadingSource, setIsLoadingSource] = useState(false);
  const [livePreviewUrl, setLivePreviewUrl] = useState('');
  const [folderScan, setFolderScan] = useState(null);
  const [scanError, setScanError] = useState('');
  const [recentRuns, setRecentRuns] = useState([]);
  const [recentRunsError, setRecentRunsError] = useState('');
  const [job, setJob] = useState(null);
  const [jobError, setJobError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
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
      .then((data) => setSoccerNetConfig(data))
      .catch((error) => {
        console.error(error);
        setSoccerNetError('Could not load SoccerNet config.');
      });
  }, []);

  useEffect(() => {
    loadRecentRuns({ hydrateLatest: true });
  }, []);

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

  const summary = job?.summary || null;
  const detectorModelOptions = config.detector_models?.length ? config.detector_models : config.player_models || [];

  const runStats = useMemo(() => {
    if (!summary) return [];
    return [
      ['Frames', summary.frames_processed || 0, 'Decoded frames pushed through detection and tracking'],
      ['Player track IDs', summary.unique_player_track_ids || 0, 'Tracker stability is more important than raw box count'],
      ['Home tracks', summary.home_tracks || 0, 'Unsupervised jersey-color split, not official metadata'],
      ['Away tracks', summary.away_tracks || 0, 'Should roughly match the second main kit cluster'],
      ['Calibrated players', summary.projected_player_points || 0, 'Player anchors that actually landed on the pitch map'],
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

  async function handleLoadSource() {
    setSourceError('');
    setIsLoadingSource(true);
    setLivePreviewUrl('');
    try {
      const payload = new FormData();
      if (selectedFile) {
        payload.append('video_file', selectedFile);
      }
      payload.append('local_video_path', form.localVideoPath);

      const response = await fetch(`${API_BASE}/api/source`, {
        method: 'POST',
        body: payload,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load source clip');
      }
      setSource(data);
    } catch (error) {
      setSourceError(error.message || 'Could not load source clip');
    } finally {
      setIsLoadingSource(false);
    }
  }

  function handleStartLivePreview() {
    if (!source?.source_id) {
      setSourceError('Load a source clip before starting live preview.');
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

  async function loadRecentRuns({ hydrateLatest = false } = {}) {
    setRecentRunsError('');
    try {
      const response = await fetch(`${API_BASE}/api/runs/recent?limit=6`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error('Could not load recent runs');
      }
      setRecentRuns(data);
      if (hydrateLatest && data.length > 0) {
        setJob((current) => current || data[0]);
      }
    } catch (error) {
      console.error(error);
      setRecentRunsError(error.message || 'Could not load recent runs');
    }
  }

  async function handleLoadRun(runId) {
    setJobError('');
    try {
      const response = await fetch(`${API_BASE}/api/runs/${runId}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load run');
      }
      clearPolling();
      setJob(data);
    } catch (error) {
      setJobError(error.message || 'Could not load run');
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
        </div>
      </header>

      <main className="layout-grid">
        <section className="left-column">
          <form className="card form-card" onSubmit={handleSubmit}>
            <div className="section-title">Run a clip</div>
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

            <div className="two-col">
              <label>
                <span>Detector weights</span>
                <input
                  list="detector-models"
                  type="text"
                  value={form.detectorModel}
                  onChange={(event) => updateForm('detectorModel', event.target.value)}
                />
              </label>
              <div className="field-note form-inline-note">
                `soccana` is the default detector for players, ball, and referees.
              </div>
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
              {isLoadingSource ? 'Loading clip...' : source ? 'Reload clip for marking' : 'Load clip for marking'}
            </button>
            {sourceError ? <div className="error-box">{sourceError}</div> : null}

            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Starting tactical run...' : 'Run tactical demo'}
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

            <button className="secondary-button" onClick={loadSoccerNetGames} type="button">
              {soccerNetLoadingGames ? 'Loading games...' : 'Load SoccerNet games'}
            </button>

            <label>
              <span>Games</span>
              <select size={8} value={soccerNetSelectedGame} onChange={(event) => setSoccerNetSelectedGame(event.target.value)}>
                {soccerNetGames.map((game) => (
                  <option key={game} value={game}>{game}</option>
                ))}
              </select>
            </label>
            <div className="field-note">
              {soccerNetGamesCount ? `${soccerNetGamesCount} games matched this split/query.` : 'Load a split to browse games.'}
            </div>

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
                      onClick={() => {
                        resetLoadedSource();
                        updateForm('localVideoPath', video.path);
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

        <section className="right-column">
          <section className="card run-card">
            <div className="row-between">
              <div className="section-title">Run status</div>
              <div className={`status-pill ${job?.status || 'idle'}`}>{job?.status || 'idle'}</div>
            </div>
            <div className="progress-shell">
              <div className="progress-bar" style={{ width: `${job?.progress || 0}%` }} />
            </div>
            <div className="muted">{job ? `${job.progress || 0}% complete` : 'No run yet.'}</div>
            <div className="log-panel">
              {(job?.logs || ['Logs will appear here.']).slice().reverse().map((line, index) => (
                <div key={`${line}-${index}`}>{line}</div>
              ))}
            </div>
          </section>

          <section className="card">
            <div className="row-between">
              <div className="section-title">Recent runs</div>
              <div className="muted">{recentRuns.length ? `${recentRuns.length} found on disk` : 'None yet'}</div>
            </div>
            {recentRunsError ? <div className="error-box">{recentRunsError}</div> : null}
            {recentRuns.length > 0 ? (
              <div className="path-list">
                {recentRuns.map((run) => (
                  <button
                    key={run.run_id}
                    type="button"
                    className={`path-chip ${job?.run_dir === run.run_dir ? 'active-chip' : ''}`}
                    onClick={() => handleLoadRun(run.run_id)}
                  >
                    {run.run_id} · {run.summary?.frames_processed || 0} frames
                  </button>
                ))}
              </div>
            ) : (
              <div className="muted">Completed runs with a `summary.json` will appear here automatically.</div>
            )}
          </section>

          <section className="card video-card">
            <div className="row-between">
              <div className="section-title">Loaded clip</div>
              <div className="muted">
                {source ? `${source.display_name} · ${source.width || 0}x${source.height || 0}` : 'Load a clip first'}
              </div>
            </div>
            {source ? (
              <>
                <div className="video-stage">
                  <video ref={sourceVideoRef} controls src={`${API_BASE}${source.video_url}`} className="video-player" />
                </div>
                <div className="source-toolbar">
                  <button className="secondary-button compact-button" type="button" onClick={handleLoadSource}>
                    {isLoadingSource ? 'Loading clip...' : 'Reload clip'}
                  </button>
                </div>
                <div className="field-note">
                  Automatic field calibration runs in the backend from pitch keypoints and refreshes every 10 frames. Use this panel to inspect the source clip only.
                </div>
              </>
            ) : (
              <div className="empty-card">Load a clip to preview the source video before starting the live model stream.</div>
            )}
          </section>

          <section className="card video-card">
            <div className="row-between">
              <div className="section-title">Live model preview</div>
              <div className="muted">{livePreviewUrl ? 'Server inference is streaming' : 'Idle'}</div>
            </div>
            <div className="source-toolbar">
              <button className="secondary-button compact-button" type="button" onClick={handleStartLivePreview}>
                Start live preview
              </button>
              <button className="secondary-button compact-button" type="button" onClick={handleStopLivePreview} disabled={!livePreviewUrl}>
                Stop live preview
              </button>
            </div>
            {livePreviewUrl ? (
              <img src={livePreviewUrl} alt="Live model preview" className="video-player live-preview-frame" />
            ) : (
              <div className="empty-card">Start live preview after loading a clip. The backend will run Soccana detection and field-keypoint calibration, refreshing the pitch transform every 10 frames.</div>
            )}
          </section>

          <section className="summary-grid">
            <div className="section-title inset-title summary-title-span">Core metrics</div>
            {runStats.length === 0 ? (
              <div className="card empty-card">Run a clip to populate the tactical summary cards.</div>
            ) : runStats.map(([label, value, hint]) => (
              <StatCard key={label} label={label} value={value} hint={hint} />
            ))}
          </section>

          <section>
            <div className="section-title inset-title">Experimental signals</div>
            {summary?.experiments?.length ? (
              <div className="experiment-grid">
                {summary.experiments.map((item) => <ExperimentCard key={item.id || item.title} item={item} />)}
              </div>
            ) : (
              <div className="card empty-card">No experiments attached to this run yet.</div>
            )}
          </section>

          <section className="card video-card">
            <div className="row-between">
              <div className="section-title">Overlay preview</div>
              <div className="muted">
                {summary?.homography_enabled ? 'Auto-calibrated' : 'Image-space only'}
              </div>
            </div>
            {summary?.overlay_video ? (
              <video controls src={`${API_BASE}${summary.overlay_video}`} className="video-player" />
            ) : (
              <div className="empty-card">Your tactical overlay preview will appear here after the run finishes.</div>
            )}
          </section>

          <section>
            <div className="section-title inset-title">Diagnostics</div>
            {summary?.diagnostics?.length ? summary.diagnostics.map((item) => <DiagnosticCard key={item.title} item={item} />) : <div className="card empty-card">No diagnostics yet.</div>}
          </section>

          <TrackTable tracks={summary?.top_tracks || []} />
          <FileLinks summary={summary} />
        </section>
      </main>
    </div>
  );
}
