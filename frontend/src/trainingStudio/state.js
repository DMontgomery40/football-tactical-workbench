import { readStoredJson, readStoredString } from './storage.js';

export const STUDIO_TABS = [
  { id: 'datasets', label: 'Datasets' },
  { id: 'train', label: 'Train' },
  { id: 'jobs', label: 'Jobs' },
  { id: 'registry', label: 'Registry' },
];

export const ACTIVE_JOB_STATUSES = new Set(['queued', 'running', 'stopping']);

export const STORAGE_KEYS = {
  datasetPath: 'fpw.trainingDatasetPath',
  studioTab: 'fpw.trainingStudioTab',
  formDraft: 'fpw.trainingFormDraft',
  selectedJobId: 'fpw.trainingSelectedJobId',
  selectedRegistryEntryId: 'fpw.trainingSelectedRegistryEntryId',
};

export const DEFAULT_FORM = {
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
};

export function normalizeDatasetPath(value) {
  return String(value || '').trim();
}

function sanitizeStoredForm(rawValue) {
  const raw = rawValue && typeof rawValue === 'object' ? rawValue : {};
  return {
    ...DEFAULT_FORM,
    ...raw,
    cache: Boolean(raw.cache),
    freeze: raw.freeze ?? '',
  };
}

function resolveSelectedJobId(currentId, jobs) {
  if (!Array.isArray(jobs) || jobs.length === 0) return '';
  if (currentId && jobs.some((job) => job.job_id === currentId)) {
    return currentId;
  }
  const activeJob = jobs.find((job) => ACTIVE_JOB_STATUSES.has(String(job?.status || '')));
  return activeJob?.job_id || jobs[0]?.job_id || '';
}

function resolveSelectedRegistryEntryId(currentId, registry) {
  const detectors = Array.isArray(registry?.detectors) ? registry.detectors : [];
  if (detectors.length === 0) return '';
  if (currentId && detectors.some((entry) => entry.id === currentId)) {
    return currentId;
  }
  if (registry?.active_detector && detectors.some((entry) => entry.id === registry.active_detector)) {
    return registry.active_detector;
  }
  return detectors[0]?.id || '';
}

export function createInitialStudioState() {
  return {
    studioTab: readStoredString(STORAGE_KEYS.studioTab, 'datasets'),
    datasetPath: readStoredString(STORAGE_KEYS.datasetPath, ''),
    scannedDatasetPath: '',
    trainingConfig: null,
    registry: null,
    jobs: [],
    datasetScan: null,
    form: sanitizeStoredForm(readStoredJson(STORAGE_KEYS.formDraft, DEFAULT_FORM)),
    selectedJobId: readStoredString(STORAGE_KEYS.selectedJobId, ''),
    selectedRegistryEntryId: readStoredString(STORAGE_KEYS.selectedRegistryEntryId, ''),
    pending: {
      bootstrap: false,
      scan: false,
      start: false,
      stopJobId: '',
      activateRunId: '',
      activateDetectorId: '',
    },
    errors: {
      global: '',
      operation: '',
      scan: '',
      jobs: '',
      registry: '',
      train: '',
    },
    bootstrapped: false,
  };
}

export function trainingStudioReducer(state, action) {
  switch (action.type) {
    case 'bootstrap/start':
      return {
        ...state,
        pending: { ...state.pending, bootstrap: true },
        errors: { ...state.errors, global: '', operation: '' },
      };
    case 'bootstrap/success': {
      const nextRegistry = action.registry || state.registry;
      const nextJobs = Array.isArray(action.jobs) ? action.jobs : state.jobs;
      return {
        ...state,
        trainingConfig: action.trainingConfig || state.trainingConfig,
        form: action.formPatch ? { ...state.form, ...action.formPatch } : state.form,
        registry: nextRegistry,
        jobs: nextJobs,
        selectedJobId: resolveSelectedJobId(state.selectedJobId, nextJobs),
        selectedRegistryEntryId: resolveSelectedRegistryEntryId(state.selectedRegistryEntryId, nextRegistry),
        pending: { ...state.pending, bootstrap: false },
        errors: { ...state.errors, global: '', operation: '' },
        bootstrapped: true,
      };
    }
    case 'bootstrap/error':
      return {
        ...state,
        pending: { ...state.pending, bootstrap: false },
        errors: { ...state.errors, global: action.message || 'Could not load the training studio.' },
        bootstrapped: true,
      };
    case 'studioTab/set':
      return { ...state, studioTab: action.value };
    case 'datasetPath/set':
      return {
        ...state,
        datasetPath: action.value,
        errors: { ...state.errors, scan: '', train: '' },
      };
    case 'scan/start':
      return {
        ...state,
        pending: { ...state.pending, scan: true },
        errors: { ...state.errors, scan: '' },
      };
    case 'scan/success':
      return {
        ...state,
        datasetScan: action.scan,
        scannedDatasetPath: normalizeDatasetPath(action.path),
        pending: { ...state.pending, scan: false },
        errors: { ...state.errors, scan: '' },
      };
    case 'scan/error':
      return {
        ...state,
        datasetScan: null,
        scannedDatasetPath: '',
        pending: { ...state.pending, scan: false },
        errors: { ...state.errors, scan: action.message || 'Could not scan dataset.' },
      };
    case 'form/patch':
      return {
        ...state,
        form: { ...state.form, ...action.patch },
      };
    case 'config/loaded':
      return {
        ...state,
        trainingConfig: action.trainingConfig,
        form: { ...state.form, ...action.formPatch },
      };
    case 'jobs/loaded': {
      const nextJobs = Array.isArray(action.jobs) ? action.jobs : [];
      return {
        ...state,
        jobs: nextJobs,
        selectedJobId: resolveSelectedJobId(state.selectedJobId, nextJobs),
        errors: { ...state.errors, jobs: '' },
      };
    }
    case 'jobs/error':
      return {
        ...state,
        errors: { ...state.errors, jobs: action.message || 'Could not refresh training jobs.' },
      };
    case 'registry/loaded': {
      const nextRegistry = action.registry || null;
      return {
        ...state,
        registry: nextRegistry,
        selectedRegistryEntryId: resolveSelectedRegistryEntryId(state.selectedRegistryEntryId, nextRegistry),
        errors: { ...state.errors, registry: '' },
      };
    }
    case 'registry/error':
      return {
        ...state,
        errors: { ...state.errors, registry: action.message || 'Could not refresh detector registry.' },
      };
    case 'job/select':
      return { ...state, selectedJobId: action.jobId };
    case 'registry/select':
      return { ...state, selectedRegistryEntryId: action.entryId };
    case 'operation/error':
      return {
        ...state,
        errors: { ...state.errors, operation: action.message || 'An operation completed with follow-up refresh errors.' },
      };
    case 'operation/clear':
      return {
        ...state,
        errors: { ...state.errors, operation: '' },
      };
    case 'train/start':
      return {
        ...state,
        pending: { ...state.pending, start: true },
        errors: { ...state.errors, train: '', operation: '' },
      };
    case 'train/success':
      return {
        ...state,
        pending: { ...state.pending, start: false },
        errors: { ...state.errors, train: '' },
        studioTab: 'jobs',
      };
    case 'train/error':
      return {
        ...state,
        pending: { ...state.pending, start: false },
        errors: { ...state.errors, train: action.message || 'Could not start detector fine-tuning.' },
        studioTab: action.tab || state.studioTab,
      };
    case 'job/stop/start':
      return {
        ...state,
        pending: { ...state.pending, stopJobId: action.jobId },
        errors: { ...state.errors, jobs: '', operation: '' },
      };
    case 'job/stop/end':
      return {
        ...state,
        pending: { ...state.pending, stopJobId: '' },
      };
    case 'run/activate/start':
      return {
        ...state,
        pending: { ...state.pending, activateRunId: action.runId },
        errors: { ...state.errors, registry: '', operation: '' },
      };
    case 'run/activate/end':
      return {
        ...state,
        pending: { ...state.pending, activateRunId: '' },
        studioTab: action.success ? 'registry' : state.studioTab,
      };
    case 'detector/activate/start':
      return {
        ...state,
        pending: { ...state.pending, activateDetectorId: action.detectorId },
        errors: { ...state.errors, registry: '', operation: '' },
      };
    case 'detector/activate/end':
      return {
        ...state,
        pending: { ...state.pending, activateDetectorId: '' },
      };
    default:
      return state;
  }
}
