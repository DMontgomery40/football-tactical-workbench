import { useEffect, useMemo, useReducer, useRef } from 'react';

import { buildHelpIndex } from '../helpUi';
import DatasetsTab from './DatasetsTab';
import JobsTab from './JobsTab';
import RegistryTab from './RegistryTab';
import StudioHeader from './StudioHeader';
import StudioTabsNav from './StudioTabsNav';
import TrainTab from './TrainTab';
import {
  ACTIVE_JOB_STATUSES,
  createInitialStudioState,
  DEFAULT_FORM,
  normalizeDatasetPath,
  STORAGE_KEYS,
  trainingStudioReducer,
} from './state';
import { writeStoredJson, writeStoredValue } from './storage';

function buildFormPatch(trainingConfig, currentForm) {
  return {
    baseWeights: currentForm.baseWeights || trainingConfig.default_base_weights || DEFAULT_FORM.baseWeights,
    epochs: currentForm.epochs || String(trainingConfig.default_hyperparameters?.epochs || DEFAULT_FORM.epochs),
    imgsz: currentForm.imgsz || String(trainingConfig.default_hyperparameters?.imgsz || DEFAULT_FORM.imgsz),
    batch: currentForm.batch || String(trainingConfig.default_hyperparameters?.batch || DEFAULT_FORM.batch),
    device: currentForm.device || trainingConfig.default_hyperparameters?.device || DEFAULT_FORM.device,
    workers: currentForm.workers || String(trainingConfig.default_hyperparameters?.workers || DEFAULT_FORM.workers),
    patience: currentForm.patience || String(trainingConfig.default_hyperparameters?.patience || DEFAULT_FORM.patience),
    freeze: currentForm.freeze ?? '',
    cache: currentForm.cache ?? Boolean(trainingConfig.default_hyperparameters?.cache),
  };
}

export default function TrainingStudioShell({ apiBase, activeDetector, helpCatalog = [], onActiveDetectorChange }) {
  const [state, dispatch] = useReducer(trainingStudioReducer, undefined, createInitialStudioState);
  const autoScanAttemptedRef = useRef(false);
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

  function syncActiveDetector(registrySnapshot) {
    const activeEntry = (registrySnapshot?.detectors || []).find((item) => item.id === registrySnapshot?.active_detector);
    onActiveDetectorChange?.({
      id: registrySnapshot?.active_detector || 'soccana',
      label: activeEntry?.label || registrySnapshot?.active_detector || 'soccana',
    });
  }

  async function loadRegistry() {
    const registry = await requestJson('/api/train/registry');
    dispatch({ type: 'registry/loaded', registry });
    syncActiveDetector(registry);
    return registry;
  }

  async function loadJobs() {
    const jobs = await requestJson('/api/train/jobs');
    dispatch({ type: 'jobs/loaded', jobs: Array.isArray(jobs) ? jobs : [] });
    return Array.isArray(jobs) ? jobs : [];
  }

  async function scanDataset(pathValue) {
    const normalizedPath = normalizeDatasetPath(pathValue);
    if (!normalizedPath) {
      dispatch({ type: 'scan/error', message: 'Enter a dataset path first.' });
      return null;
    }

    dispatch({ type: 'scan/start' });
    try {
      const scan = await requestJson('/api/train/datasets/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: normalizedPath }),
      });
      dispatch({ type: 'scan/success', scan, path: normalizedPath });
      return scan;
    } catch (error) {
      dispatch({ type: 'scan/error', message: error.message || 'Could not scan dataset.' });
      return null;
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      dispatch({ type: 'bootstrap/start' });
      try {
        const [trainingConfig, registry, jobs] = await Promise.all([
          requestJson('/api/train/config'),
          requestJson('/api/train/registry'),
          requestJson('/api/train/jobs'),
        ]);
        if (cancelled) return;
        dispatch({
          type: 'bootstrap/success',
          trainingConfig,
          registry,
          jobs: Array.isArray(jobs) ? jobs : [],
          formPatch: buildFormPatch(trainingConfig, state.form),
        });
        syncActiveDetector(registry);
      } catch (error) {
        if (!cancelled) {
          dispatch({ type: 'bootstrap/error', message: error.message || 'Could not load the training studio.' });
        }
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!state.bootstrapped || autoScanAttemptedRef.current) {
      return;
    }
    autoScanAttemptedRef.current = true;
    const persistedPath = normalizeDatasetPath(state.datasetPath);
    if (persistedPath) {
      void scanDataset(persistedPath);
    }
  }, [state.bootstrapped, state.datasetPath]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.datasetPath, state.datasetPath);
  }, [state.datasetPath]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.studioTab, state.studioTab);
  }, [state.studioTab]);

  useEffect(() => {
    writeStoredJson(STORAGE_KEYS.formDraft, state.form);
  }, [state.form]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.selectedJobId, state.selectedJobId);
  }, [state.selectedJobId]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.selectedRegistryEntryId, state.selectedRegistryEntryId);
  }, [state.selectedRegistryEntryId]);

  const hasActiveJob = useMemo(
    () => state.jobs.some((job) => ACTIVE_JOB_STATUSES.has(String(job?.status || ''))),
    [state.jobs],
  );

  useEffect(() => {
    if (!hasActiveJob) {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return undefined;
    }

    async function pollTrainingState() {
      try {
        await Promise.all([loadJobs(), loadRegistry()]);
      } catch (error) {
        dispatch({ type: 'jobs/error', message: error.message || 'Could not refresh training jobs.' });
      }
    }

    pollTrainingState();
    pollRef.current = window.setInterval(pollTrainingState, 2000);

    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [hasActiveJob]);

  async function handleStartTraining(event) {
    event.preventDefault();
    const normalizedPath = normalizeDatasetPath(state.datasetPath);
    const scanMatchesCurrentPath = Boolean(state.datasetScan && state.scannedDatasetPath === normalizedPath);
    if (!normalizedPath) {
      dispatch({ type: 'train/error', message: 'Choose a dataset path first.', tab: 'datasets' });
      return;
    }
    if (!state.datasetScan || !scanMatchesCurrentPath || !state.datasetScan.can_start) {
      dispatch({
        type: 'train/error',
        message: 'Run a clean dataset scan for the current path first, then fix any blocking issues before starting detector fine-tuning.',
        tab: 'datasets',
      });
      return;
    }

    dispatch({ type: 'train/start' });
    try {
      const payload = {
        base_weights: state.form.baseWeights,
        dataset_path: normalizedPath,
        run_name: state.form.runName.trim(),
        epochs: Number(state.form.epochs) || 50,
        imgsz: Number(state.form.imgsz) || 640,
        batch: Number(state.form.batch) || 16,
        device: state.form.device.trim() || 'auto',
        workers: Number(state.form.workers) || 4,
        patience: Number(state.form.patience) || 20,
        freeze: state.form.freeze === '' ? null : Number(state.form.freeze),
        cache: Boolean(state.form.cache),
      };
      await requestJson('/api/train/jobs/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await Promise.all([loadJobs(), loadRegistry()]);
      dispatch({ type: 'train/success' });
    } catch (error) {
      dispatch({ type: 'train/error', message: error.message || 'Could not start detector fine-tuning.' });
    }
  }

  async function handleStopJob(jobId) {
    dispatch({ type: 'job/stop/start', jobId });
    try {
      await requestJson(`/api/train/jobs/${jobId}/stop`, { method: 'POST' });
      await loadJobs();
    } catch (error) {
      dispatch({ type: 'jobs/error', message: error.message || 'Could not stop training job.' });
    } finally {
      dispatch({ type: 'job/stop/end' });
    }
  }

  async function handleActivateRun(runId) {
    dispatch({ type: 'run/activate/start', runId });
    try {
      await requestJson(`/api/train/runs/${runId}/activate`, { method: 'POST' });
      const [registry] = await Promise.all([loadRegistry(), loadJobs()]);
      dispatch({ type: 'registry/select', entryId: registry?.active_detector || 'soccana' });
      dispatch({ type: 'run/activate/end' });
    } catch (error) {
      dispatch({ type: 'registry/error', message: error.message || 'Could not activate detector.' });
      dispatch({ type: 'run/activate/end' });
    }
  }

  async function handleActivateRegistryEntry(entry) {
    dispatch({ type: 'detector/activate/start', detectorId: entry.id });
    try {
      await requestJson('/api/train/registry/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detector_id: entry.id }),
      });
      const registry = await loadRegistry();
      dispatch({ type: 'registry/select', entryId: registry?.active_detector || entry.id });
      dispatch({ type: 'detector/activate/end' });
    } catch (error) {
      dispatch({ type: 'registry/error', message: error.message || 'Could not activate detector.' });
      dispatch({ type: 'detector/activate/end' });
    }
  }

  const normalizedDatasetPath = normalizeDatasetPath(state.datasetPath);
  const scanMatchesCurrentPath = Boolean(state.datasetScan && state.scannedDatasetPath === normalizedDatasetPath);
  const enabledBaseWeights = state.trainingConfig?.available_base_weights || [{ id: 'soccana', label: 'soccana (football-pretrained)' }];
  const deviceOptions = state.trainingConfig?.device_options || [
    { id: 'auto', label: 'Auto' },
    { id: 'mps', label: 'Apple Silicon MPS' },
    { id: 'cpu', label: 'CPU only' },
  ];
  const activeRegistryId = state.registry?.active_detector || activeDetector || 'soccana';
  const trainDisabledReason = !normalizedDatasetPath
    ? 'Choose a dataset path first.'
    : !state.datasetScan || !scanMatchesCurrentPath
      ? 'Run a dataset scan for the current path before starting training.'
      : !state.datasetScan.can_start
        ? 'Fix the blocking dataset issues before starting a run.'
        : '';

  return (
    <section className="studio-shell">
      <StudioHeader
        trainingConfig={state.trainingConfig}
        registry={state.registry}
        activeDetector={activeDetector}
        helpIndex={helpIndex}
      />

      <StudioTabsNav
        studioTab={state.studioTab}
        onSelectTab={(tabId) => dispatch({ type: 'studioTab/set', value: tabId })}
      />

      {state.errors.global ? <div className="error-box">{state.errors.global}</div> : null}

      {state.studioTab === 'datasets' ? (
        <DatasetsTab
          helpIndex={helpIndex}
          datasetPath={state.datasetPath}
          onDatasetPathChange={(value) => dispatch({ type: 'datasetPath/set', value })}
          onScanDataset={() => scanDataset(state.datasetPath)}
          onOpenTrain={() => dispatch({ type: 'studioTab/set', value: 'train' })}
          isScanning={state.pending.scan}
          scanError={state.errors.scan}
          datasetScan={state.datasetScan}
          isScanStale={Boolean(state.datasetScan && !scanMatchesCurrentPath)}
        />
      ) : null}

      {state.studioTab === 'train' ? (
        <TrainTab
          helpIndex={helpIndex}
          datasetPath={state.datasetPath}
          datasetScan={state.datasetScan}
          scanMatchesCurrentPath={scanMatchesCurrentPath}
          form={state.form}
          enabledBaseWeights={enabledBaseWeights}
          deviceOptions={deviceOptions}
          trainingConfig={state.trainingConfig}
          isStarting={state.pending.start}
          trainDisabledReason={trainDisabledReason}
          trainError={state.errors.train}
          onFormChange={(patch) => dispatch({ type: 'form/patch', patch })}
          onStartTraining={handleStartTraining}
          onOpenDatasets={() => dispatch({ type: 'studioTab/set', value: 'datasets' })}
        />
      ) : null}

      {state.studioTab === 'jobs' ? (
        <JobsTab
          helpIndex={helpIndex}
          jobs={state.jobs}
          jobsError={state.errors.jobs}
          selectedJobId={state.selectedJobId}
          onSelectJob={(jobId) => dispatch({ type: 'job/select', jobId })}
          onOpenRegistry={() => dispatch({ type: 'studioTab/set', value: 'registry' })}
          onStopJob={handleStopJob}
          onActivateRun={handleActivateRun}
          pendingStopJobId={state.pending.stopJobId}
          pendingActivateRunId={state.pending.activateRunId}
        />
      ) : null}

      {state.studioTab === 'registry' ? (
        <RegistryTab
          helpIndex={helpIndex}
          registry={state.registry}
          registryError={state.errors.registry}
          selectedRegistryEntryId={state.selectedRegistryEntryId}
          activeRegistryId={activeRegistryId}
          pendingActivateDetectorId={state.pending.activateDetectorId}
          onSelectEntry={(entryId) => dispatch({ type: 'registry/select', entryId })}
          onActivateEntry={handleActivateRegistryEntry}
        />
      ) : null}
    </section>
  );
}
