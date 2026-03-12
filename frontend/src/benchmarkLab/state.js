import { readStoredJson, readStoredString } from '../trainingStudio/storage.js';

export const BENCHMARK_TABS = [
  { id: 'setup', label: 'Setup' },
  { id: 'leaderboard', label: 'Leaderboard' },
];

export const POLLING_BENCHMARK_STATUSES = new Set(['queued', 'running']);

export const STORAGE_KEYS = {
  tab: 'fpw.benchmarkTab',
  clipSourcePath: 'fpw.benchmarkClipSourcePath',
  selectedBenchmarkId: 'fpw.benchmarkSelectedId',
  selectedCandidateId: 'fpw.benchmarkSelectedCandidateId',
};

function resolveSelectedBenchmarkId(currentId, benchmarks) {
  if (!Array.isArray(benchmarks) || benchmarks.length === 0) return '';
  if (currentId && benchmarks.some((b) => b.benchmark_id === currentId)) {
    return currentId;
  }
  const active = benchmarks.find((b) => POLLING_BENCHMARK_STATUSES.has(String(b?.status || '')));
  return active?.benchmark_id || benchmarks[0]?.benchmark_id || '';
}

function resolveSelectedCandidateId(currentId, candidates) {
  if (!Array.isArray(candidates) || candidates.length === 0) return '';
  if (currentId && candidates.some((c) => c.id === currentId)) {
    return currentId;
  }
  return candidates[0]?.id || '';
}

export function createInitialBenchmarkState() {
  return {
    tab: readStoredString(STORAGE_KEYS.tab, 'setup'),
    clipSourcePath: readStoredString(STORAGE_KEYS.clipSourcePath, ''),
    clipStatus: null,
    runtimeProfile: null,
    candidates: [],
    benchmarks: [],
    selectedBenchmarkId: readStoredString(STORAGE_KEYS.selectedBenchmarkId, ''),
    selectedCandidateId: readStoredString(STORAGE_KEYS.selectedCandidateId, ''),
    pending: {
      bootstrap: false,
      ensureClip: false,
      importCandidate: false,
      runBenchmark: false,
    },
    errors: {
      global: '',
      operation: '',
      clip: '',
      candidates: '',
      benchmark: '',
    },
    bootstrapped: false,
  };
}

export function benchmarkLabReducer(state, action) {
  switch (action.type) {
    case 'bootstrap/start':
      return {
        ...state,
        pending: { ...state.pending, bootstrap: true },
        errors: { ...state.errors, global: '', operation: '' },
      };
    case 'bootstrap/success': {
      const nextCandidates = Array.isArray(action.candidates) ? action.candidates : state.candidates;
      const nextBenchmarks = Array.isArray(action.benchmarks) ? action.benchmarks : state.benchmarks;
      return {
        ...state,
        clipStatus: action.clipStatus || state.clipStatus,
        runtimeProfile: action.runtimeProfile || state.runtimeProfile,
        candidates: nextCandidates,
        benchmarks: nextBenchmarks,
        selectedBenchmarkId: resolveSelectedBenchmarkId(state.selectedBenchmarkId, nextBenchmarks),
        selectedCandidateId: resolveSelectedCandidateId(state.selectedCandidateId, nextCandidates),
        pending: { ...state.pending, bootstrap: false },
        errors: { ...state.errors, global: '' },
        bootstrapped: true,
      };
    }
    case 'bootstrap/error':
      return {
        ...state,
        pending: { ...state.pending, bootstrap: false },
        errors: { ...state.errors, global: action.message || 'Could not load Benchmark Lab.' },
        bootstrapped: true,
      };
    case 'tab/set':
      return { ...state, tab: action.value };
    case 'clipSourcePath/set':
      return {
        ...state,
        clipSourcePath: action.value,
        errors: { ...state.errors, clip: '' },
      };
    case 'clip/loaded':
      return {
        ...state,
        clipStatus: action.clipStatus,
        errors: { ...state.errors, clip: '' },
      };
    case 'clip/error':
      return {
        ...state,
        errors: { ...state.errors, clip: action.message || 'Could not check clip status.' },
      };
    case 'clip/ensure/start':
      return {
        ...state,
        pending: { ...state.pending, ensureClip: true },
        errors: { ...state.errors, clip: '' },
      };
    case 'clip/ensure/success':
      return {
        ...state,
        clipStatus: action.clipStatus,
        pending: { ...state.pending, ensureClip: false },
        errors: { ...state.errors, clip: '' },
      };
    case 'clip/ensure/error':
      return {
        ...state,
        pending: { ...state.pending, ensureClip: false },
        errors: { ...state.errors, clip: action.message || 'Could not prepare benchmark clip.' },
      };
    case 'candidates/loaded': {
      const nextCandidates = Array.isArray(action.candidates) ? action.candidates : [];
      return {
        ...state,
        candidates: nextCandidates,
        selectedCandidateId: resolveSelectedCandidateId(state.selectedCandidateId, nextCandidates),
        errors: { ...state.errors, candidates: '' },
      };
    }
    case 'candidates/error':
      return {
        ...state,
        errors: { ...state.errors, candidates: action.message || 'Could not load candidates.' },
      };
    case 'candidate/import/start':
      return {
        ...state,
        pending: { ...state.pending, importCandidate: true },
        errors: { ...state.errors, candidates: '' },
      };
    case 'candidate/import/success': {
      const merged = [...state.candidates];
      if (action.candidate && !merged.some((c) => c.id === action.candidate.id)) {
        merged.push(action.candidate);
      }
      return {
        ...state,
        candidates: merged,
        selectedCandidateId: action.candidate?.id || state.selectedCandidateId,
        pending: { ...state.pending, importCandidate: false },
        errors: { ...state.errors, candidates: '' },
      };
    }
    case 'candidate/import/error':
      return {
        ...state,
        pending: { ...state.pending, importCandidate: false },
        errors: { ...state.errors, candidates: action.message || 'Could not import candidate.' },
      };
    case 'candidate/select':
      return { ...state, selectedCandidateId: action.candidateId };
    case 'benchmark/run/start':
      return {
        ...state,
        pending: { ...state.pending, runBenchmark: true },
        errors: { ...state.errors, benchmark: '' },
      };
    case 'benchmark/run/success':
      return {
        ...state,
        pending: { ...state.pending, runBenchmark: false },
        errors: { ...state.errors, benchmark: '' },
        tab: 'leaderboard',
      };
    case 'benchmark/run/error':
      return {
        ...state,
        pending: { ...state.pending, runBenchmark: false },
        errors: { ...state.errors, benchmark: action.message || 'Could not start benchmark.' },
      };
    case 'benchmarks/loaded': {
      const nextBenchmarks = Array.isArray(action.benchmarks) ? action.benchmarks : [];
      return {
        ...state,
        benchmarks: nextBenchmarks,
        selectedBenchmarkId: resolveSelectedBenchmarkId(state.selectedBenchmarkId, nextBenchmarks),
        errors: { ...state.errors, benchmark: '' },
      };
    }
    case 'benchmarks/error':
      return {
        ...state,
        errors: { ...state.errors, benchmark: action.message || 'Could not load benchmark history.' },
      };
    case 'benchmark/select':
      return { ...state, selectedBenchmarkId: action.benchmarkId };
    case 'operation/error':
      return {
        ...state,
        errors: { ...state.errors, operation: action.message || 'An operation failed.' },
      };
    case 'operation/clear':
      return {
        ...state,
        errors: { ...state.errors, operation: '' },
      };
    default:
      return state;
  }
}
