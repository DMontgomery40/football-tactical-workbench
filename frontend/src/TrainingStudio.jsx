import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { buildHelpIndex, FieldLabel, HelpPopover, MicroLabelWithHelp, SectionTitleWithHelp } from './helpUi';

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

function formatCurveAxis(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '';
  return `E${numeric.toFixed(1)}`;
}

function formatCurveTooltipLabel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'Training sample';
  return `Epoch progress ${numeric.toFixed(2)}`;
}

function formatPathTail(value) {
  if (!value) return 'n/a';
  const parts = String(value).split(/[\\/]+/).filter(Boolean);
  if (parts.length <= 3) return String(value);
  return `.../${parts.slice(-3).join('/')}`;
}

function formatClassIds(value) {
  if (!Array.isArray(value) || value.length === 0) return 'none';
  return value.join(', ');
}

function ActiveDetectorSummary({ registry, activeDetector, helpIndex }) {
  const activeId = registry?.active_detector || activeDetector || 'soccana';
  const activeEntry = (registry?.detectors || []).find((item) => item.id === activeId);

  return (
    <div className="studio-active-detector">
      <div>
        <MicroLabelWithHelp label="Active detector" entry={helpIndex.get('training.active_detector')} />
        <div className="studio-active-detector-name">{activeEntry?.label || activeId}</div>
        <div className="muted">{formatPathTail(activeEntry?.path || '')}</div>
      </div>
      <div className={`active-badge ${activeId === 'soccana' ? 'default-active-badge' : ''}`}>
        {activeId === 'soccana' ? 'pretrained active' : 'custom checkpoint active'}
      </div>
    </div>
  );
}

function ArtifactList({ artifacts }) {
  const entries = Object.entries(artifacts || {}).filter(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    return Boolean(value);
  });

  if (entries.length === 0) {
    return <div className="muted">No artifacts written yet.</div>;
  }

  return (
    <div className="studio-artifact-list">
      {entries.map(([key, value]) => (
        <div key={key} className="studio-artifact-row">
          <div className="micro-label">{key.replace(/_/g, ' ')}</div>
          {Array.isArray(value) ? (
            <div className="studio-artifact-values">
              {value.map((item) => (
                <div key={item} className="studio-meta-value">{item}</div>
              ))}
            </div>
          ) : (
            <div className="studio-meta-value">{value}</div>
          )}
        </div>
      ))}
    </div>
  );
}

function TrainingCurvesPanel({ trainingCurves }) {
  const lossData = Array.isArray(trainingCurves?.loss) ? trainingCurves.loss : [];
  const optimizerData = Array.isArray(trainingCurves?.optimizer) ? trainingCurves.optimizer : [];

  if (lossData.length === 0 && optimizerData.length === 0) {
    return null;
  }

  return (
    <section className="studio-chart-grid">
      <div className="card studio-chart-card">
        <div className="row-between">
          <div>
            <div className="micro-label">Live loss curves</div>
            <div className="section-title">Detection losses</div>
          </div>
          <div className="muted">{lossData.length} samples</div>
        </div>
        <div className="studio-chart-shell">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={lossData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(125, 143, 160, 0.24)" />
              <XAxis dataKey="epoch_progress" tickFormatter={formatCurveAxis} minTickGap={24} stroke="currentColor" />
              <YAxis stroke="currentColor" width={44} />
              <Tooltip
                formatter={(value) => formatMetric(value)}
                labelFormatter={formatCurveTooltipLabel}
                contentStyle={{ borderRadius: 12, border: '1px solid rgba(125, 143, 160, 0.32)' }}
              />
              <Legend />
              <Line type="monotone" dataKey="box_loss" name="box" stroke="#1f5f92" strokeWidth={2.2} dot={false} connectNulls />
              <Line type="monotone" dataKey="cls_loss" name="cls" stroke="#b5541f" strokeWidth={2.2} dot={false} connectNulls />
              <Line type="monotone" dataKey="dfl_loss" name="dfl" stroke="#2f6f4f" strokeWidth={2.2} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card studio-chart-card">
        <div className="row-between">
          <div>
            <div className="micro-label">Live optimizer signals</div>
            <div className="section-title">Gradient norm and learning rate</div>
          </div>
          <div className="muted">{optimizerData.length} samples</div>
        </div>
        <div className="studio-chart-shell">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={optimizerData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(125, 143, 160, 0.24)" />
              <XAxis dataKey="epoch_progress" tickFormatter={formatCurveAxis} minTickGap={24} stroke="currentColor" />
              <YAxis yAxisId="left" stroke="currentColor" width={44} />
              <YAxis yAxisId="right" orientation="right" stroke="currentColor" width={52} tickFormatter={(value) => Number(value).toExponential(0)} />
              <Tooltip
                formatter={(value, name) => (
                  name === 'lr' ? Number(value).toExponential(3) : formatMetric(value)
                )}
                labelFormatter={formatCurveTooltipLabel}
                contentStyle={{ borderRadius: 12, border: '1px solid rgba(125, 143, 160, 0.32)' }}
              />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="grad_norm" name="grad norm" stroke="#7d2349" strokeWidth={2.2} dot={false} connectNulls />
              <Line yAxisId="right" type="monotone" dataKey="lr" name="lr" stroke="#0f766e" strokeWidth={2.2} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

export default function TrainingStudio({ apiBase, activeDetector, helpCatalog = [], onActiveDetectorChange }) {
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
  const helpIndex = useMemo(() => buildHelpIndex(helpCatalog), [helpCatalog]);

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
      epochs: current.epochs || String(data.default_hyperparameters?.epochs || 50),
      imgsz: current.imgsz || String(data.default_hyperparameters?.imgsz || 640),
      batch: current.batch || String(data.default_hyperparameters?.batch || 16),
      device: current.device || data.default_hyperparameters?.device || 'auto',
      workers: current.workers || String(data.default_hyperparameters?.workers || 4),
      patience: current.patience || String(data.default_hyperparameters?.patience || 20),
      freeze: current.freeze ?? '',
      cache: current.cache ?? Boolean(data.default_hyperparameters?.cache),
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
    const normalized = Array.isArray(data) ? data : [];
    setJobs(normalized);
    return normalized;
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
        const latestJobs = await loadJobs();
        if (!latestJobs.some((job) => ACTIVE_JOB_STATUSES.has(String(job?.status || '')))) {
          await loadRegistry();
        }
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
    try {
      const data = await requestJson('/api/train/datasets/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: datasetPath.trim() }),
      });
      setDatasetScan(data);
    } catch (error) {
      setDatasetScan(null);
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
    if (!datasetScan?.can_start) {
      setTrainError('Run a clean dataset scan first, then fix any blocking issues before starting detector fine-tuning.');
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
      await loadRegistry();
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
      setStudioTab('registry');
    } catch (error) {
      setRegistryError(error.message || 'Could not activate detector.');
    } finally {
      setPendingActionRunId('');
    }
  }

  async function handleActivateRegistryEntry(entry) {
    setRegistryError('');
    setPendingActionRunId(entry.id);
    try {
      await requestJson('/api/train/registry/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detector_id: entry.id }),
      });
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
  const deviceOptions = trainingConfig?.device_options || [
    { id: 'auto', label: 'Auto' },
    { id: 'mps', label: 'Apple Silicon MPS' },
    { id: 'cpu', label: 'CPU only' },
    { id: 'cuda', label: 'CUDA GPU' },
  ];
  const activeRegistryId = registry?.active_detector || activeDetector || 'soccana';
  const scanTierClass = datasetScan?.tier === 'valid' ? 'completed' : datasetScan?.tier === 'invalid' ? 'failed' : 'stopping';
  const trainDisabledReason = !datasetPath.trim()
    ? 'Choose a dataset path first.'
    : !datasetScan
      ? 'Run a dataset scan before starting training.'
      : !datasetScan.can_start
        ? 'Fix the blocking dataset issues before starting a run.'
        : '';
  const completedJobs = useMemo(
    () => jobs.filter((job) => job.status === 'completed'),
    [jobs],
  );

  return (
    <section className="studio-shell">
      <section className="card studio-header training-studio-hero">
        <div className="studio-header-copy">
          <div className="eyebrow">training studio</div>
          <SectionTitleWithHelp
            title="Fine-tune the football detector in its own workspace."
            entry={helpIndex.get('training.studio_overview')}
          />
          <p className="studio-intro">
            This V1 stays focused: start from the football-pretrained `soccana` checkpoint, adapt it to your camera domain locally,
            and then promote the best checkpoint straight back into analysis.
          </p>
          <div className="studio-status-ribbon">
            <span>{trainingConfig?.backend_label || 'Training backend'}</span>
            {trainingConfig?.backend_version ? <span>v{trainingConfig.backend_version}</span> : null}
            <span>mac-first via MPS</span>
            <span>CUDA-ready path preserved</span>
          </div>
        </div>
        <div className="studio-header-side">
          <ActiveDetectorSummary registry={registry} activeDetector={activeDetector} helpIndex={helpIndex} />
          {trainingConfig?.license_note ? (
            <div className="studio-license-note">
              <div className="micro-label">Licensing caveat</div>
              <div>{trainingConfig.license_note}</div>
            </div>
          ) : null}
        </div>
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
            <SectionTitleWithHelp title="Dataset intake" entry={helpIndex.get('training.dataset_intake')} />
            <div className="field-note">
              Point this at a YOLO detector dataset root. The scanner checks split structure, label integrity, class mapping, and whether the trained checkpoint can safely come back into analysis.
            </div>
            <label>
              <FieldLabel label="Dataset path" entry={helpIndex.get('training.dataset_path')} />
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
              <div>
                <SectionTitleWithHelp title="Scan result" entry={helpIndex.get('training.scan_result')} />
                <div className="muted">
                  {datasetScan ? (datasetScan.can_start ? 'Ready for detector fine-tuning.' : 'Needs fixes before a run can start.') : 'Run a scan to inspect readiness.'}
                </div>
              </div>
              <div className={`status-pill ${scanTierClass}`}>{datasetScan?.tier || 'waiting'}</div>
            </div>

            {!datasetScan ? (
              <div className="empty-card">Run a dataset scan to inspect splits, classes, and blocking issues before training.</div>
            ) : (
              <>
                <div className="studio-meta-grid">
                  <div>
                    <div className="micro-label">Dataset root</div>
                    <div className="studio-meta-value">{datasetScan.path}</div>
                  </div>
                  <div>
                    <div className="micro-label">Dataset YAML</div>
                    <div className="studio-meta-value">{datasetScan.has_yaml ? datasetScan.yaml_path : 'Missing'}</div>
                  </div>
                  <div>
                    <MicroLabelWithHelp label="Validation strategy" entry={helpIndex.get('training.validation_strategy')} />
                    <div className="studio-meta-value">{datasetScan.suggested_validation_strategy || 'existing_split'}</div>
                  </div>
                  <div>
                    <div className="micro-label">Class source</div>
                    <div className="studio-meta-value">{datasetScan.classes_source || 'missing'}</div>
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

                <div className="studio-class-map">
                  <div className="studio-class-map-card">
                    <MicroLabelWithHelp label="Player / keeper ids" entry={helpIndex.get('training.class_mapping')} />
                    <div className="studio-meta-value">{formatClassIds(datasetScan.class_mapping?.player_class_ids)}</div>
                  </div>
                  <div className="studio-class-map-card">
                    <MicroLabelWithHelp label="Ball ids" entry={helpIndex.get('training.class_mapping')} />
                    <div className="studio-meta-value">{formatClassIds(datasetScan.class_mapping?.ball_class_ids)}</div>
                  </div>
                  <div className="studio-class-map-card">
                    <MicroLabelWithHelp label="Referee ids" entry={helpIndex.get('training.class_mapping')} />
                    <div className="studio-meta-value">{formatClassIds(datasetScan.class_mapping?.referee_class_ids)}</div>
                  </div>
                </div>

                <div className="studio-split-grid">
                  {Object.entries(datasetScan.splits || {}).map(([splitName, split]) => (
                    <div key={splitName} className="studio-split-card detailed-split-card">
                      <div className="micro-label">{splitName}</div>
                      <div className="studio-split-count">{split.images || 0} images</div>
                      <div className="muted">{split.label_files || 0} label files</div>
                      <div className="muted">{split.labeled_images || 0} labeled images</div>
                      <div className="muted">{split.instances || 0} instances</div>
                      <div className="muted">path: {split.path || 'missing'}</div>
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
                    <div className="micro-label">Blocking issues</div>
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
            <SectionTitleWithHelp title="Training family" entry={helpIndex.get('training.training_family')} />
            <div className="studio-family-grid">
              <div className="studio-family-card active-family-card">
                <div className="micro-label">Enabled now</div>
                <div className="studio-family-title">Detector fine-tuning</div>
                <div className="muted">Starts from `soccana`, keeps training local, and writes a detector registry entry when the run finishes.</div>
              </div>
              <div className="studio-family-card">
                <div className="micro-label">Reserved next</div>
                <div className="studio-family-title">Field calibration</div>
                <div className="muted">Future keypoint or calibration-family training slots in here without polluting the detector flow.</div>
              </div>
            </div>

            <div className={`studio-readiness-card ${datasetScan?.can_start ? 'ready' : ''}`}>
              <div className="micro-label">Dataset readiness</div>
              <div className="studio-readiness-title">
                {datasetScan?.can_start ? 'Ready to fine-tune' : 'Dataset scan still required'}
              </div>
              <div className="muted">
                {datasetScan
                  ? (datasetScan.can_start ? 'The worker will generate a run-local manifest and leave your source dataset untouched.' : 'Fix the blocking dataset issues from the Datasets tab before starting a run.')
                  : 'Run a dataset scan first so the training form can verify class mapping and split integrity.'}
              </div>
            </div>
          </section>

          <form className="card studio-panel" onSubmit={handleStartTraining}>
            <div className="row-between">
              <SectionTitleWithHelp title="Detector fine-tuning" entry={helpIndex.get('training.detector_finetuning')} />
              <div className="muted">{datasetPath ? datasetPath : 'Choose a dataset first'}</div>
            </div>
            <div className="training-form-grid">
              <label>
                <FieldLabel label="Base weights" entry={helpIndex.get('training.base_weights')} />
                <select value={form.baseWeights} onChange={(event) => updateForm('baseWeights', event.target.value)}>
                  {enabledBaseWeights.map((item) => (
                    <option key={item.id} value={item.id}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label>
                <FieldLabel label="Run name" entry={helpIndex.get('training.run_name')} />
                <input
                  type="text"
                  value={form.runName}
                  onChange={(event) => updateForm('runName', event.target.value)}
                  placeholder="broadcast-side-cam-tune"
                />
              </label>
              <label>
                <FieldLabel label="Epochs" entry={helpIndex.get('training.hyperparameters')} />
                <input type="number" min="1" value={form.epochs} onChange={(event) => updateForm('epochs', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="Image size" entry={helpIndex.get('training.hyperparameters')} />
                <input type="number" min="32" step="32" value={form.imgsz} onChange={(event) => updateForm('imgsz', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="Batch" entry={helpIndex.get('training.hyperparameters')} />
                <input type="number" min="1" value={form.batch} onChange={(event) => updateForm('batch', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="Device" entry={helpIndex.get('training.device')} />
                <select value={form.device} onChange={(event) => updateForm('device', event.target.value)}>
                  {deviceOptions.map((item) => (
                    <option key={item.id || item} value={item.id || item}>{item.label || item}</option>
                  ))}
                </select>
              </label>
              <label>
                <FieldLabel label="Workers" entry={helpIndex.get('training.hyperparameters')} />
                <input type="number" min="0" value={form.workers} onChange={(event) => updateForm('workers', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="Patience" entry={helpIndex.get('training.hyperparameters')} />
                <input type="number" min="0" value={form.patience} onChange={(event) => updateForm('patience', event.target.value)} />
              </label>
              <label>
                <FieldLabel label="Freeze" entry={helpIndex.get('training.hyperparameters')} />
                <input type="number" min="0" value={form.freeze} onChange={(event) => updateForm('freeze', event.target.value)} placeholder="leave blank for none" />
              </label>
              <label className="studio-checkbox-field">
                <span className="checkbox-label-row">
                  <span>Cache images</span>
                  <HelpPopover entry={helpIndex.get('training.hyperparameters')} />
                </span>
                <input type="checkbox" checked={form.cache} onChange={(event) => updateForm('cache', event.target.checked)} />
              </label>
            </div>

            <div className="studio-runtime-note">
              <MicroLabelWithHelp label="Runtime behavior" entry={helpIndex.get('training.runtime_behavior')} />
              <div>
                The worker gets a run-local <code>dataset_runtime.yaml</code>, writes <code>summary.json</code>, <code>train.log</code>,
                plot artifacts, and the best checkpoint under <code>backend/training_runs/&lt;run_id&gt;/</code>.
              </div>
              {trainingConfig?.device_guidance ? <div className="muted">{trainingConfig.device_guidance}</div> : null}
            </div>

            <div className="source-toolbar">
              <button className="secondary-button compact-button" type="button" onClick={() => setStudioTab('datasets')}>
                Back to dataset scan
              </button>
              <button className="primary-button compact-button" type="submit" disabled={isStarting || Boolean(trainDisabledReason)}>
                {isStarting ? 'Starting fine-tuning...' : 'Start fine-tuning'}
              </button>
            </div>
            {trainDisabledReason ? <div className="field-note">{trainDisabledReason}</div> : null}
            {trainError ? <div className="error-box">{trainError}</div> : null}
          </form>
        </section>
      ) : null}

      {studioTab === 'jobs' ? (
        <section className="studio-stack">
          <section className="card studio-panel">
            <SectionTitleWithHelp title="Training jobs" entry={helpIndex.get('training.jobs')} />
            <div className="field-note">Use this view to judge run health, inspect artifacts, and promote completed checkpoints into the detector registry.</div>
          </section>
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
                    <div className="micro-label">Backend</div>
                    <div className="studio-meta-value">
                      {[job.backend, job.backend_version ? `v${job.backend_version}` : null].filter(Boolean).join(' ')}
                    </div>
                  </div>
                  <div>
                    <div className="micro-label">Resolved device</div>
                    <div className="studio-meta-value">{job.resolved_device || job.config?.device || 'pending'}</div>
                  </div>
                  <div>
                    <div className="micro-label">Epoch</div>
                    <div className="studio-meta-value">{job.current_epoch || 0} / {job.total_epochs || 0}</div>
                  </div>
                  <div>
                    <div className="micro-label">Started</div>
                    <div className="studio-meta-value">{formatTimestamp(job.started_at || job.created_at)}</div>
                  </div>
                  <div>
                    <div className="micro-label">Finished</div>
                    <div className="studio-meta-value">{job.finished_at ? formatTimestamp(job.finished_at) : 'still running'}</div>
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
                    <span>Precision {formatMetric(job.metrics?.precision)}</span>
                    <span>Recall {formatMetric(job.metrics?.recall)}</span>
                  </div>
                </div>

                <TrainingCurvesPanel trainingCurves={job.training_curves} />

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
                  <summary>Run artifacts and dataset contract</summary>
                  <div className="studio-detail-stack">
                    <div className="studio-runtime-note">
                      <MicroLabelWithHelp label="Generated dataset manifest" entry={helpIndex.get('training.job_artifacts')} />
                      <div>{job.generated_dataset_yaml || 'Not written yet'}</div>
                    </div>
                    <ArtifactList artifacts={job.artifacts} />
                    {job.dataset_scan ? (
                      <div className="studio-dataset-review">
                        <div className="micro-label">Dataset scan snapshot</div>
                        <div className="studio-meta-grid">
                          <div>
                            <div className="micro-label">Tier</div>
                            <div className="studio-meta-value">{job.dataset_scan.tier}</div>
                          </div>
                          <div>
                            <div className="micro-label">Validation strategy</div>
                            <div className="studio-meta-value">{job.validation_strategy || job.dataset_scan.suggested_validation_strategy}</div>
                          </div>
                          <div>
                            <div className="micro-label">Player ids</div>
                            <div className="studio-meta-value">{formatClassIds(job.dataset_scan.class_mapping?.player_class_ids)}</div>
                          </div>
                          <div>
                            <div className="micro-label">Ball ids</div>
                            <div className="studio-meta-value">{formatClassIds(job.dataset_scan.class_mapping?.ball_class_ids)}</div>
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </details>

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
          {completedJobs.length > 0 ? (
            <div className="field-note">
              Completed runs stay on disk under `backend/training_runs/` and can be promoted again later from the Registry tab.
            </div>
          ) : null}
        </section>
      ) : null}

      {studioTab === 'registry' ? (
        <section className="studio-stack">
          {registryError ? <div className="error-box">{registryError}</div> : null}
          <section className="card studio-panel">
            <div className="row-between">
              <div>
                <SectionTitleWithHelp title="Detector registry" entry={helpIndex.get('training.registry')} />
                <div className="field-note">Analysis and live preview use the active detector whenever the analysis selector stays on `soccana`.</div>
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
                      <div className="micro-label">Resolved device</div>
                      <div className="studio-meta-value">{entry.resolved_device || 'n/a'}</div>
                    </div>
                    <div>
                      <div className="micro-label">Backend</div>
                      <div className="studio-meta-value">
                        {[entry.backend, entry.backend_version ? `v${entry.backend_version}` : null].filter(Boolean).join(' ') || 'n/a'}
                      </div>
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

                  {!entry.is_pretrained ? (
                    <div className="studio-class-map compact-class-map">
                      <div className="studio-class-map-card">
                        <div className="micro-label">Player ids</div>
                        <div className="studio-meta-value">{formatClassIds(entry.class_ids?.player_class_ids)}</div>
                      </div>
                      <div className="studio-class-map-card">
                        <div className="micro-label">Ball ids</div>
                        <div className="studio-meta-value">{formatClassIds(entry.class_ids?.ball_class_ids)}</div>
                      </div>
                      <div className="studio-class-map-card">
                        <div className="micro-label">Ref ids</div>
                        <div className="studio-meta-value">{formatClassIds(entry.class_ids?.referee_class_ids)}</div>
                      </div>
                    </div>
                  ) : null}

                  {entry.summary_path ? (
                    <div className="studio-runtime-note compact-note">
                      <div className="micro-label">Run summary</div>
                      <div>{entry.summary_path}</div>
                    </div>
                  ) : null}
                </div>

                {entry.id !== activeRegistryId ? (
                  <button
                    className="primary-button compact-button"
                    type="button"
                    onClick={() => handleActivateRegistryEntry(entry)}
                    disabled={pendingActionRunId === entry.id}
                  >
                    {pendingActionRunId === entry.id ? 'Activating...' : entry.is_pretrained ? 'Use pretrained' : 'Activate'}
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
