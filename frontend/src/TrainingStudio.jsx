import { useEffect, useRef, useState } from 'react';

const STUDIO_TABS = [
  { id: 'datasets', label: 'Datasets' },
  { id: 'train', label: 'Train' },
  { id: 'jobs', label: 'Jobs' },
  { id: 'registry', label: 'Registry' },
];

const ACTIVE_JOB_STATUSES = new Set(['queued', 'running', 'stopping']);
const STORAGE_KEYS = {
  datasetPath: 'fpw.trainingDatasetPath',
};

function readStoredString(key, fallback = '') {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function formatTimestamp(value) {
  if (!value) return 'Unknown';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatMetric(value) {
  if (value === null || value === undefined || value === '') return 'n/a';
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return numeric.toFixed(3);
}

function ActiveDetectorSummary({ registry, activeDetector }) {
  const activeId = registry?.active_detector || activeDetector || 'soccana';
  const activeEntry = (registry?.detectors || []).find((item) => item.id === activeId);

  return (
    <div className="studio-active-detector">
      <div>
        <div className="micro-label">Active detector</div>
        <div className="studio-active-detector-name">{activeEntry?.label || activeId}</div>
      </div>
      <div className={`active-badge ${activeId === 'soccana' ? 'default-active-badge' : ''}`}>
        {activeId === 'soccana' ? 'pretrained active' : 'custom checkpoint active'}
      </div>
    </div>
  );
}

export default function TrainingStudio({ apiBase, activeDetector, onActiveDetectorChange }) {
  const [studioTab, setStudioTab] = useState('datasets');
  const [datasetPath, setDatasetPath] = useState(() => readStoredString(STORAGE_KEYS.datasetPath, ''));
  const [trainingConfig, setTrainingConfig] = useState(null);
  const [registry, setRegistry] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [datasetScan, setDatasetScan] = useState(null);
  const [scanError, setScanError] = useState('');
  const [jobsError, setJobsError] = useState('');
  const [registryError, setRegistryError] = useState('');
  const [trainError, setTrainError] = useState('');
  const [globalError, setGlobalError] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [pendingActionRunId, setPendingActionRunId] = useState('');
  const [form, setForm] = useState({
    baseWeights: 'soccana',
    runName: '',
    epochs: '50',
    imgsz: '640',
    batch: '16',
    device: 'auto',
    workers: '4',
    patience: '20',
    freeze: '',
    cache: false,
  });
  const pollRef = useRef(null);

  async function requestJson(path, options = {}) {
    const response = await fetch(`${apiBase}${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || payload.error || 'Request failed');
    }
    return payload;
  }

  async function loadTrainingConfig() {
    const data = await requestJson('/api/train/config');
    setTrainingConfig(data);
    setForm((current) => ({
      ...current,
      baseWeights: current.baseWeights || data.default_base_weights || 'soccana',
    }));
    return data;
  }

  async function loadRegistry() {
    const data = await requestJson('/api/train/registry');
    setRegistry(data);
    const activeEntry = (data.detectors || []).find((item) => item.id === data.active_detector);
    onActiveDetectorChange?.({
      id: data.active_detector,
      label: activeEntry?.label || data.active_detector,
    });
    return data;
  }

  async function loadJobs() {
    const data = await requestJson('/api/train/jobs');
    setJobs(Array.isArray(data) ? data : []);
    return data;
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        setGlobalError('');
        await Promise.all([loadTrainingConfig(), loadRegistry(), loadJobs()]);
      } catch (error) {
        if (!cancelled) {
          setGlobalError(error.message || 'Could not load the training studio.');
        }
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.datasetPath, datasetPath);
    } catch {}
  }, [datasetPath]);

  useEffect(() => {
    const hasActiveJob = jobs.some((job) => ACTIVE_JOB_STATUSES.has(String(job?.status || '')));
    if (!hasActiveJob) {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return undefined;
    }

    async function pollJobs() {
      try {
        setJobsError('');
        await loadJobs();
      } catch (error) {
        setJobsError(error.message || 'Could not refresh training jobs.');
      }
    }

    pollJobs();
    pollRef.current = window.setInterval(pollJobs, 2000);

    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [jobs]);

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleScanDataset() {
    if (!datasetPath.trim()) {
      setScanError('Enter a dataset path first.');
      return;
    }

    setIsScanning(true);
    setScanError('');
    setDatasetScan(null);
    try {
      const data = await requestJson('/api/train/datasets/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: datasetPath.trim() }),
      });
      setDatasetScan(data);
    } catch (error) {
      setScanError(error.message || 'Could not scan dataset.');
    } finally {
      setIsScanning(false);
    }
  }

  async function handleStartTraining(event) {
    event.preventDefault();
    setTrainError('');
    if (!datasetPath.trim()) {
      setTrainError('Choose a dataset path first.');
      setStudioTab('datasets');
      return;
    }

    setIsStarting(true);
    try {
      const payload = {
        base_weights: form.baseWeights,
        dataset_path: datasetPath.trim(),
        run_name: form.runName.trim(),
        epochs: Number(form.epochs) || 50,
        imgsz: Number(form.imgsz) || 640,
        batch: Number(form.batch) || 16,
        device: form.device.trim() || 'auto',
        workers: Number(form.workers) || 4,
        patience: Number(form.patience) || 20,
        freeze: form.freeze === '' ? null : Number(form.freeze),
        cache: Boolean(form.cache),
      };
      await requestJson('/api/train/jobs/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await loadJobs();
      setStudioTab('jobs');
    } catch (error) {
      setTrainError(error.message || 'Could not start detector fine-tuning.');
    } finally {
      setIsStarting(false);
    }
  }

  async function handleStopJob(jobId) {
    setJobsError('');
    setPendingActionRunId(jobId);
    try {
      await requestJson(`/api/train/jobs/${jobId}/stop`, { method: 'POST' });
      await loadJobs();
    } catch (error) {
      setJobsError(error.message || 'Could not stop training job.');
    } finally {
      setPendingActionRunId('');
    }
  }

  async function handleActivateRun(runId) {
    setRegistryError('');
    setPendingActionRunId(runId);
    try {
      await requestJson(`/api/train/runs/${runId}/activate`, { method: 'POST' });
      const nextRegistry = await loadRegistry();
      const activeEntry = (nextRegistry.detectors || []).find((item) => item.id === nextRegistry.active_detector);
      onActiveDetectorChange?.({
        id: nextRegistry.active_detector,
        label: activeEntry?.label || nextRegistry.active_detector,
      });
    } catch (error) {
      setRegistryError(error.message || 'Could not activate detector.');
    } finally {
      setPendingActionRunId('');
    }
  }

  const enabledBaseWeights = trainingConfig?.available_base_weights || [{ id: 'soccana', label: 'soccana (football-pretrained)' }];
  const activeRegistryId = registry?.active_detector || activeDetector || 'soccana';

  return (
    <section className="studio-shell">
      <section className="card studio-header">
        <div className="studio-header-copy">
          <div className="eyebrow">training studio</div>
          <div className="section-title">Fine-tune the football detector without leaving the workbench.</div>
          <p className="studio-intro">
            Jobs run in isolated Python subprocesses, keep their own logs on disk, and can promote a finished checkpoint into the analysis workspace with one activation click.
          </p>
        </div>
        <ActiveDetectorSummary registry={registry} activeDetector={activeDetector} />
      </section>

      <section className="card studio-nav">
        {STUDIO_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`studio-tab ${studioTab === tab.id ? 'active-studio-tab' : ''}`}
            onClick={() => setStudioTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </section>

      {globalError ? <div className="error-box">{globalError}</div> : null}

      {studioTab === 'datasets' ? (
        <section className="studio-panel-grid">
          <section className="card studio-panel">
            <div className="section-title">YOLO dataset scan</div>
            <div className="field-note">
              Point this at a YOLO dataset root. The scanner looks for `dataset.yaml`, split folders, image counts, and label coverage before you spend time on a training run.
            </div>
            <label>
              <span>Dataset path</span>
              <input
                type="text"
                value={datasetPath}
                onChange={(event) => {
                  setDatasetPath(event.target.value);
                  setDatasetScan(null);
                }}
                placeholder="/Users/you/datasets/football-yolo"
              />
            </label>
            <div className="source-toolbar">
              <button className="secondary-button compact-button" type="button" onClick={handleScanDataset}>
                {isScanning ? 'Scanning...' : 'Scan dataset'}
              </button>
              <button className="secondary-button compact-button" type="button" onClick={() => setStudioTab('train')}>
                Open training form
              </button>
            </div>
            {scanError ? <div className="error-box">{scanError}</div> : null}
          </section>

          <section className={`card studio-panel scan-result-card ${datasetScan?.tier || 'neutral'}`}>
            <div className="row-between">
              <div className="section-title">Scan result</div>
              <div className={`status-pill ${datasetScan?.tier === 'valid' ? 'completed' : datasetScan?.tier === 'invalid' ? 'failed' : 'stopping'}`}>
                {datasetScan?.tier || 'waiting'}
              </div>
            </div>
            {!datasetScan ? (
              <div className="empty-card">Run a dataset scan to inspect splits, classes, and warnings before training.</div>
            ) : (
              <>
                <div className="studio-meta-grid">
                  <div>
                    <div className="micro-label">Dataset root</div>
                    <div className="studio-meta-value">{datasetScan.path}</div>
                  </div>
                  <div>
                    <div className="micro-label">YAML</div>
                    <div className="studio-meta-value">{datasetScan.has_yaml ? datasetScan.yaml_path : 'Generated at runtime'}</div>
                  </div>
                </div>

                <div className="micro-label">Classes</div>
                <div className="class-chip-row">
                  {(datasetScan.classes || []).length ? (
                    datasetScan.classes.map((item) => (
                      <span key={item} className="class-chip">{item}</span>
                    ))
                  ) : (
                    <span className="muted">No class names were parsed from the dataset metadata.</span>
                  )}
                </div>

                <div className="studio-split-grid">
                  {Object.entries(datasetScan.splits || {}).map(([splitName, counts]) => (
                    <div key={splitName} className="studio-split-card">
                      <div className="micro-label">{splitName}</div>
                      <div className="studio-split-count">{counts.images || 0} images</div>
                      <div className="muted">{counts.labels || 0} labels</div>
                    </div>
                  ))}
                </div>

                {(datasetScan.warnings || []).length ? (
                  <div className="studio-list-block warn-list">
                    <div className="micro-label">Warnings</div>
                    {(datasetScan.warnings || []).map((item) => <div key={item}>{item}</div>)}
                  </div>
                ) : null}

                {(datasetScan.errors || []).length ? (
                  <div className="studio-list-block error-list">
                    <div className="micro-label">Errors</div>
                    {(datasetScan.errors || []).map((item) => <div key={item}>{item}</div>)}
                  </div>
                ) : null}
              </>
            )}
          </section>
        </section>
      ) : null}

      {studioTab === 'train' ? (
        <section className="studio-panel-grid">
          <section className="card studio-panel">
            <div className="section-title">Training family</div>
            <div className="studio-family-grid">
              <div className="studio-family-card active-family-card">
                <div className="micro-label">Enabled now</div>
                <div className="studio-family-title">Detector fine-tuning</div>
                <div className="muted">Starts from `soccana` and writes a new custom checkpoint into the registry flow.</div>
              </div>
              <div className="studio-family-card">
                <div className="micro-label">Coming soon</div>
                <div className="studio-family-title">Field calibration</div>
                <div className="muted">Reserved for future keypoint or calibration training workflows.</div>
              </div>
            </div>
          </section>

          <form className="card studio-panel" onSubmit={handleStartTraining}>
            <div className="row-between">
              <div className="section-title">Detector fine-tuning</div>
              <div className="muted">{datasetPath ? datasetPath : 'Choose a dataset first'}</div>
            </div>
            <div className="training-form-grid">
              <label>
                <span>Base weights</span>
                <select value={form.baseWeights} onChange={(event) => updateForm('baseWeights', event.target.value)}>
                  {enabledBaseWeights.map((item) => (
                    <option key={item.id} value={item.id}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>Run name</span>
                <input
                  type="text"
                  value={form.runName}
                  onChange={(event) => updateForm('runName', event.target.value)}
                  placeholder="my-domain-tune"
                />
              </label>
              <label>
                <span>Epochs</span>
                <input type="number" min="1" value={form.epochs} onChange={(event) => updateForm('epochs', event.target.value)} />
              </label>
              <label>
                <span>Image size</span>
                <input type="number" min="32" step="32" value={form.imgsz} onChange={(event) => updateForm('imgsz', event.target.value)} />
              </label>
              <label>
                <span>Batch</span>
                <input type="number" min="1" value={form.batch} onChange={(event) => updateForm('batch', event.target.value)} />
              </label>
              <label>
                <span>Device</span>
                <input type="text" value={form.device} onChange={(event) => updateForm('device', event.target.value)} placeholder="auto / cpu / cuda / mps" />
              </label>
              <label>
                <span>Workers</span>
                <input type="number" min="0" value={form.workers} onChange={(event) => updateForm('workers', event.target.value)} />
              </label>
              <label>
                <span>Patience</span>
                <input type="number" min="0" value={form.patience} onChange={(event) => updateForm('patience', event.target.value)} />
              </label>
              <label>
                <span>Freeze</span>
                <input type="number" min="0" value={form.freeze} onChange={(event) => updateForm('freeze', event.target.value)} placeholder="leave blank for none" />
              </label>
              <label className="studio-checkbox-field">
                <span>Cache images</span>
                <input type="checkbox" checked={form.cache} onChange={(event) => updateForm('cache', event.target.checked)} />
              </label>
            </div>
            <div className="field-note">
              Training uses a separate worker process so the API stays responsive. If your dataset scan still shows warnings, those will be carried into the job log before the worker starts.
            </div>
            <div className="source-toolbar">
              <button className="secondary-button compact-button" type="button" onClick={() => setStudioTab('datasets')}>
                Back to dataset scan
              </button>
              <button className="primary-button compact-button" type="submit" disabled={isStarting}>
                {isStarting ? 'Starting fine-tuning...' : 'Start fine-tuning'}
              </button>
            </div>
            {trainError ? <div className="error-box">{trainError}</div> : null}
          </form>
        </section>
      ) : null}

      {studioTab === 'jobs' ? (
        <section className="studio-stack">
          {jobsError ? <div className="error-box">{jobsError}</div> : null}
          {jobs.length === 0 ? (
            <div className="card empty-card studio-panel">No training jobs yet. Start a fine-tuning run from the Train tab.</div>
          ) : (
            jobs.map((job) => (
              <section key={job.job_id} className="card train-job-card">
                <div className="row-between">
                  <div>
                    <div className="micro-label">Run</div>
                    <div className="section-title">{job.config?.run_name || job.run_id}</div>
                  </div>
                  <div className={`status-pill ${job.status || 'idle'}`}>{job.status || 'idle'}</div>
                </div>
                <div className="studio-meta-grid">
                  <div>
                    <div className="micro-label">Base weights</div>
                    <div className="studio-meta-value">{job.config?.base_weights || 'soccana'}</div>
                  </div>
                  <div>
                    <div className="micro-label">Created</div>
                    <div className="studio-meta-value">{formatTimestamp(job.created_at)}</div>
                  </div>
                  <div>
                    <div className="micro-label">Epoch</div>
                    <div className="studio-meta-value">{job.current_epoch || 0} / {job.total_epochs || 0}</div>
                  </div>
                  <div>
                    <div className="micro-label">Best checkpoint</div>
                    <div className="studio-meta-value">{job.best_checkpoint || 'Not available yet'}</div>
                  </div>
                </div>
                <div className="progress-shell">
                  <div className="progress-bar" style={{ width: `${job.progress || 0}%` }} />
                </div>
                <div className="row-between">
                  <div className="muted">{job.progress || 0}% complete</div>
                  <div className="studio-metric-inline">
                    <span>mAP50 {formatMetric(job.metrics?.mAP50)}</span>
                    <span>mAP50-95 {formatMetric(job.metrics?.mAP50_95)}</span>
                  </div>
                </div>
                {job.error ? <div className="error-box">{job.error}</div> : null}
                <div className="source-toolbar">
                  <button className="secondary-button compact-button" type="button" onClick={() => setStudioTab('registry')}>
                    Open registry
                  </button>
                  {ACTIVE_JOB_STATUSES.has(String(job.status || '')) ? (
                    <button
                      className="secondary-button compact-button"
                      type="button"
                      onClick={() => handleStopJob(job.job_id)}
                      disabled={pendingActionRunId === job.job_id}
                    >
                      {pendingActionRunId === job.job_id ? 'Stopping...' : 'Stop'}
                    </button>
                  ) : null}
                  {job.status === 'completed' && job.best_checkpoint ? (
                    <button
                      className="primary-button compact-button"
                      type="button"
                      onClick={() => handleActivateRun(job.run_id)}
                      disabled={pendingActionRunId === job.run_id}
                    >
                      {pendingActionRunId === job.run_id ? 'Activating...' : 'Activate detector'}
                    </button>
                  ) : null}
                </div>
                <details className="studio-log-details">
                  <summary>Logs</summary>
                  <div className="log-panel">
                    {(job.logs || ['No logs yet.']).slice().reverse().map((line, index) => (
                      <div key={`${job.job_id}-${index}-${line}`}>{line}</div>
                    ))}
                  </div>
                </details>
              </section>
            ))
          )}
        </section>
      ) : null}

      {studioTab === 'registry' ? (
        <section className="studio-stack">
          {registryError ? <div className="error-box">{registryError}</div> : null}
          <section className="card studio-panel">
            <div className="row-between">
              <div>
                <div className="section-title">Detector registry</div>
                <div className="field-note">Analysis and live preview use the active detector when the analysis selector stays on `soccana`.</div>
              </div>
              <div className="active-badge">{activeRegistryId === 'soccana' ? 'using pretrained detector' : 'using custom detector'}</div>
            </div>
          </section>

          {(registry?.detectors || []).length === 0 ? (
            <div className="card empty-card studio-panel">No detectors are registered yet.</div>
          ) : (
            (registry?.detectors || []).map((entry) => (
              <section key={entry.id} className={`card registry-row ${entry.id === activeRegistryId ? 'active-registry-row' : ''}`}>
                <div className="registry-main">
                  <div className="row-between">
                    <div>
                      <div className="section-title">{entry.label || entry.id}</div>
                      <div className="muted">{entry.id}</div>
                    </div>
                    <div className={`active-badge ${entry.id === activeRegistryId ? '' : 'inactive-badge'}`}>
                      {entry.id === activeRegistryId ? 'Active' : entry.is_pretrained ? 'Pretrained' : 'Available'}
                    </div>
                  </div>
                  <div className="studio-meta-grid">
                    <div>
                      <div className="micro-label">Created</div>
                      <div className="studio-meta-value">{formatTimestamp(entry.created_at)}</div>
                    </div>
                    <div>
                      <div className="micro-label">Base weights</div>
                      <div className="studio-meta-value">{entry.base_weights || 'soccana'}</div>
                    </div>
                    <div>
                      <div className="micro-label">mAP50</div>
                      <div className="studio-meta-value">{formatMetric(entry.metrics?.mAP50)}</div>
                    </div>
                    <div>
                      <div className="micro-label">Checkpoint</div>
                      <div className="studio-meta-value">{entry.path}</div>
                    </div>
                  </div>
                </div>
                {entry.id !== activeRegistryId && entry.training_run_id ? (
                  <button
                    className="primary-button compact-button"
                    type="button"
                    onClick={() => handleActivateRun(entry.training_run_id)}
                    disabled={pendingActionRunId === entry.training_run_id}
                  >
                    {pendingActionRunId === entry.training_run_id ? 'Activating...' : 'Activate'}
                  </button>
                ) : null}
              </section>
            ))
          )}
        </section>
      ) : null}
    </section>
  );
}
