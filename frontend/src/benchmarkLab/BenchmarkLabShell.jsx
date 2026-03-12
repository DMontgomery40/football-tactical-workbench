import { useEffect, useMemo, useReducer, useRef } from 'react';

import { buildHelpIndex, SectionTitleWithHelp } from '../helpUi';
import { writeStoredValue } from '../trainingStudio/storage';
import BenchmarkControls from './BenchmarkControls';
import CandidateDetail from './CandidateDetail';
import CandidateLibrary from './CandidateLibrary';
import ClipCard from './ClipCard';
import Leaderboard from './Leaderboard';
import {
  BENCHMARK_TABS,
  POLLING_BENCHMARK_STATUSES,
  STORAGE_KEYS,
  benchmarkLabReducer,
  createInitialBenchmarkState,
} from './state';

function getErrorMessage(error, fallback) {
  return error instanceof Error && error.message ? error.message : fallback;
}

function extractResponseError(payload, fallback) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return fallback;
  }
  if (typeof payload.detail === 'string' && payload.detail.trim()) {
    return payload.detail;
  }
  if (typeof payload.error === 'string' && payload.error.trim()) {
    return payload.error;
  }
  return fallback;
}

function SnGamestateReferenceCard({ helpIndex }) {
  return (
    <section className="card benchmark-reference-card">
      <SectionTitleWithHelp title="sn-gamestate baseline" entry={helpIndex.get('benchmark.sn_gamestate')} />
      <p className="field-note">
        SoccerNet&apos;s official Game State Reconstruction devkit is a separate TrackLab-based baseline and evaluator stack.
        It auto-downloads its dataset and weights on first run, and its real accuracy metric is GS-HOTA on labeled SoccerNetGS clips.
      </p>
      <p className="field-note">
        This workbench can review arbitrary clips, but that is not the same as official label-backed evaluation. Use the SoccerNet baseline when you need the published GSR scoring contract.
      </p>
      <div className="path-list">
        <a href="https://github.com/SoccerNet/sn-gamestate" target="_blank" rel="noreferrer">Repository</a>
        <a href="https://arxiv.org/abs/2404.11335" target="_blank" rel="noreferrer">Paper</a>
        <a href="https://www.soccer-net.org/tasks/new-game-state-reconstruction" target="_blank" rel="noreferrer">Task page</a>
      </div>
    </section>
  );
}

export default function BenchmarkLabShell({ apiBase, helpCatalog = [], activePipeline = 'classic', activeDetector = 'soccana' }) {
  const [state, dispatch] = useReducer(benchmarkLabReducer, undefined, createInitialBenchmarkState);
  const pollRef = useRef(null);
  const helpIndex = useMemo(() => buildHelpIndex(helpCatalog), [helpCatalog]);

  async function requestJson(path, options = {}) {
    const response = await fetch(`${apiBase}${path}`, options);
    const rawText = await response.text();
    let payload = {};

    if (rawText) {
      try {
        payload = JSON.parse(rawText);
      } catch (error) {
        if (!response.ok) {
          throw new Error(rawText.trim() || `Request failed (${response.status})`);
        }
        throw new Error(`Invalid JSON response from ${path}: ${getErrorMessage(error, 'Could not parse response body.')}`);
      }
    }

    if (!response.ok) {
      throw new Error(extractResponseError(payload, rawText.trim() || `Request failed (${response.status})`));
    }
    return payload;
  }

  async function loadCandidates() {
    const data = await requestJson('/api/benchmark/candidates');
    dispatch({ type: 'candidates/loaded', candidates: Array.isArray(data?.candidates) ? data.candidates : data });
    return data;
  }

  async function loadBenchmarks() {
    const data = await requestJson('/api/benchmark/history');
    dispatch({ type: 'benchmarks/loaded', benchmarks: Array.isArray(data) ? data : Array.isArray(data?.benchmarks) ? data.benchmarks : [] });
    return data;
  }

  // Bootstrap
  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      dispatch({ type: 'bootstrap/start' });
      try {
        const [config, history] = await Promise.all([
          requestJson('/api/benchmark/config'),
          requestJson('/api/benchmark/history'),
        ]);
        if (cancelled) return;
        dispatch({
          type: 'bootstrap/success',
          clipStatus: config.clip_status || null,
          runtimeProfile: config.runtime_profile || null,
          candidates: Array.isArray(config.candidates) ? config.candidates : [],
          benchmarks: Array.isArray(history) ? history : Array.isArray(history?.benchmarks) ? history.benchmarks : [],
        });
      } catch (error) {
        if (!cancelled) {
          dispatch({ type: 'bootstrap/error', message: getErrorMessage(error, 'Could not load Benchmark Lab.') });
        }
      }
    }

    bootstrap();
    return () => { cancelled = true; };
  }, []);

  // Persist tab
  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.tab, state.tab);
  }, [state.tab]);

  // Persist clip source path
  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.clipSourcePath, state.clipSourcePath);
  }, [state.clipSourcePath]);

  // Persist selections
  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.selectedBenchmarkId, state.selectedBenchmarkId);
  }, [state.selectedBenchmarkId]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.selectedCandidateId, state.selectedCandidateId);
  }, [state.selectedCandidateId]);

  // Poll active benchmarks
  const hasActiveBenchmark = useMemo(
    () => state.benchmarks.some((b) => POLLING_BENCHMARK_STATUSES.has(String(b?.status || ''))),
    [state.benchmarks],
  );

  useEffect(() => {
    if (!hasActiveBenchmark) {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return undefined;
    }

    async function poll() {
      try {
        await Promise.all([loadBenchmarks(), loadCandidates()]);
      } catch (error) {
        dispatch({ type: 'benchmarks/error', message: getErrorMessage(error, 'Could not refresh benchmarks.') });
      }
    }

    poll();
    pollRef.current = window.setInterval(poll, 3000);

    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [hasActiveBenchmark]);

  // --- Handlers ---

  async function handleEnsureClip(sourcePath) {
    dispatch({ type: 'clip/ensure/start' });
    try {
      const result = await requestJson('/api/benchmark/ensure-clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_path: sourcePath || '' }),
      });
      dispatch({ type: 'clip/ensure/success', clipStatus: result });
    } catch (error) {
      dispatch({ type: 'clip/ensure/error', message: getErrorMessage(error, 'Could not prepare clip.') });
    }
  }

  async function handleUploadClip(file) {
    dispatch({ type: 'clip/ensure/start' });
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${apiBase}/api/benchmark/ensure-clip-upload`, {
        method: 'POST',
        body: formData,
      });
      const rawText = await response.text();
      let payload = {};
      if (rawText) {
        try { payload = JSON.parse(rawText); } catch { payload = {}; }
      }
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || rawText.trim() || `Upload failed (${response.status})`);
      }
      dispatch({ type: 'clipSourcePath/set', value: payload.path || file.name });
      dispatch({ type: 'clip/ensure/success', clipStatus: payload });
    } catch (error) {
      dispatch({ type: 'clip/ensure/error', message: getErrorMessage(error, 'Could not upload clip.') });
    }
  }

  async function handleImportLocal(path) {
    dispatch({ type: 'candidate/import/start' });
    try {
      const result = await requestJson('/api/benchmark/candidates/import-local', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_path: path }),
      });
      dispatch({ type: 'candidate/import/success', candidate: result.candidate || result });
    } catch (error) {
      dispatch({ type: 'candidate/import/error', message: getErrorMessage(error, 'Could not import local weights.') });
    }
  }

  async function handleImportHf(repoId, filename) {
    dispatch({ type: 'candidate/import/start' });
    try {
      const body = { repo_id: repoId };
      if (filename) body.filename = filename;
      const result = await requestJson('/api/benchmark/candidates/import-hf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      dispatch({ type: 'candidate/import/success', candidate: result.candidate || result });
    } catch (error) {
      dispatch({ type: 'candidate/import/error', message: getErrorMessage(error, 'Could not import from HuggingFace.') });
    }
  }

  async function handleRunBenchmark(candidateIds) {
    dispatch({ type: 'benchmark/run/start' });
    try {
      const body = candidateIds ? { candidate_ids: candidateIds } : {};
      await requestJson('/api/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      await loadBenchmarks();
      dispatch({ type: 'benchmark/run/success' });
    } catch (error) {
      dispatch({ type: 'benchmark/run/error', message: getErrorMessage(error, 'Could not start benchmark.') });
    }
  }

  // --- Derived ---

  const selectedCandidate = useMemo(
    () => state.candidates.find((c) => c.id === state.selectedCandidateId) || null,
    [state.candidates, state.selectedCandidateId],
  );

  const activeBenchmark = useMemo(
    () => state.benchmarks.find((b) => b.benchmark_id === state.selectedBenchmarkId) || null,
    [state.benchmarks, state.selectedBenchmarkId],
  );

  const selectedCandidateResult = useMemo(() => {
    if (!activeBenchmark?.leaderboard || !state.selectedCandidateId) return null;
    return activeBenchmark.leaderboard.find((r) => r.candidate_id === state.selectedCandidateId) || null;
  }, [activeBenchmark, state.selectedCandidateId]);

  return (
    <section className="benchmark-shell">
      <div className="card benchmark-header">
        <div className="benchmark-header-copy">
          <div className="section-title">
            <span>Benchmark Lab</span>
          </div>
          <p className="studio-intro">
            Compare model outputs on a locked reference clip under identical runtime conditions.
            This surface currently produces a proxy clip-comparison score, not a label-backed accuracy benchmark.
          </p>
          {activePipeline !== (state.runtimeProfile?.pipeline || 'classic') ? (
            <p className="benchmark-pipeline-notice">
              Your Analysis Workspace uses the <strong>{activePipeline}</strong> pipeline.
              Benchmark Lab locks all runs to <strong>{state.runtimeProfile?.pipeline || 'classic'}</strong> for
              fair comparison. Results may not reflect your active pipeline configuration.
            </p>
          ) : null}
        </div>
      </div>

      <nav className="card studio-nav benchmark-nav" aria-label="Benchmark Lab tabs">
        {BENCHMARK_TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`studio-tab${state.tab === t.id ? ' active-studio-tab' : ''}`}
            onClick={() => dispatch({ type: 'tab/set', value: t.id })}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {state.errors.global ? <div className="error-box">{state.errors.global}</div> : null}
      {state.errors.operation ? <div className="error-box">{state.errors.operation}</div> : null}

      {state.tab === 'setup' ? (
        <div className="benchmark-setup-grid">
          <div className="benchmark-setup-left">
            <ClipCard
              clipStatus={state.clipStatus}
              runtimeProfile={state.runtimeProfile}
              clipSourcePath={state.clipSourcePath}
              onClipSourcePathChange={(value) => dispatch({ type: 'clipSourcePath/set', value })}
              isEnsuring={state.pending.ensureClip}
              clipError={state.errors.clip}
              onEnsureClip={handleEnsureClip}
              onUploadClip={handleUploadClip}
              helpIndex={helpIndex}
            />
            <CandidateLibrary
              candidates={state.candidates}
              selectedCandidateId={state.selectedCandidateId}
              onSelectCandidate={(id) => dispatch({ type: 'candidate/select', candidateId: id })}
              onImportLocal={handleImportLocal}
              onImportHf={handleImportHf}
              isImporting={state.pending.importCandidate}
              candidatesError={state.errors.candidates}
              helpIndex={helpIndex}
            />
            <BenchmarkControls
              clipReady={state.clipStatus?.ready}
              candidates={state.candidates}
              selectedCandidateId={state.selectedCandidateId}
              isRunning={state.pending.runBenchmark}
              benchmarkError={state.errors.benchmark}
              onRunBenchmark={handleRunBenchmark}
              helpIndex={helpIndex}
            />
            <SnGamestateReferenceCard helpIndex={helpIndex} />
          </div>
          <div className="benchmark-setup-right">
            <CandidateDetail
              apiBase={apiBase}
              candidate={selectedCandidate}
              benchmarkResult={selectedCandidateResult}
              candidateRunResult={activeBenchmark?.candidate_results?.[state.selectedCandidateId] || null}
              helpIndex={helpIndex}
            />
          </div>
        </div>
      ) : null}

      {state.tab === 'leaderboard' ? (
        <div className="benchmark-leaderboard-grid">
          <Leaderboard
            benchmarks={state.benchmarks}
            selectedBenchmarkId={state.selectedBenchmarkId}
            onSelectBenchmark={(id) => dispatch({ type: 'benchmark/select', benchmarkId: id })}
            onSelectCandidate={(id) => {
              dispatch({ type: 'candidate/select', candidateId: id });
              dispatch({ type: 'tab/set', value: 'setup' });
            }}
            helpIndex={helpIndex}
          />
          {selectedCandidateResult ? (
            <CandidateDetail
              apiBase={apiBase}
              candidate={selectedCandidate}
              benchmarkResult={selectedCandidateResult}
              candidateRunResult={activeBenchmark?.candidate_results?.[state.selectedCandidateId] || null}
              helpIndex={helpIndex}
            />
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
