import assert from 'node:assert/strict';
import test from 'node:test';

import { benchmarkLabReducer, createInitialBenchmarkState } from '../../frontend/src/benchmarkLab/state.js';

function createState(overrides = {}) {
  return {
    ...createInitialBenchmarkState(),
    bootstrapped: true,
    ...overrides,
  };
}

test('bootstrap/start sets pending flag and clears errors', () => {
  const state = createState({
    errors: { global: 'old', operation: 'old', clip: '', candidates: '', benchmark: '' },
  });
  const next = benchmarkLabReducer(state, { type: 'bootstrap/start' });

  assert.equal(next.pending.bootstrap, true);
  assert.equal(next.errors.global, '');
  assert.equal(next.errors.operation, '');
});

test('bootstrap/success resolves selected benchmark and candidate', () => {
  const state = createState();
  const candidates = [{ id: 'soccana' }, { id: 'custom_1' }];
  const benchmarks = [{ benchmark_id: 'b1', status: 'completed' }];
  const next = benchmarkLabReducer(state, {
    type: 'bootstrap/success',
    candidates,
    benchmarks,
    clipStatus: { ready: true },
    runtimeProfile: { tracker_mode: 'hybrid_reid' },
  });

  assert.equal(next.bootstrapped, true);
  assert.equal(next.pending.bootstrap, false);
  assert.equal(next.clipStatus.ready, true);
  assert.equal(next.candidates.length, 2);
  assert.equal(next.benchmarks.length, 1);
  assert.equal(next.selectedBenchmarkId, 'b1');
  assert.equal(next.selectedCandidateId, 'soccana');
});

test('bootstrap/error sets global error and marks bootstrapped', () => {
  const state = createState({ bootstrapped: false });
  const next = benchmarkLabReducer(state, {
    type: 'bootstrap/error',
    message: 'Network failed',
  });

  assert.equal(next.bootstrapped, true);
  assert.equal(next.pending.bootstrap, false);
  assert.equal(next.errors.global, 'Network failed');
});

test('tab/set changes the active tab', () => {
  const state = createState({ tab: 'setup' });
  const next = benchmarkLabReducer(state, { type: 'tab/set', value: 'leaderboard' });
  assert.equal(next.tab, 'leaderboard');
});

test('candidate/import/success merges new candidate and selects it', () => {
  const state = createState({
    candidates: [{ id: 'soccana' }],
    selectedCandidateId: 'soccana',
  });
  const next = benchmarkLabReducer(state, {
    type: 'candidate/import/success',
    candidate: { id: 'import_abc', label: 'My local checkpoint' },
  });

  assert.equal(next.candidates.length, 2);
  assert.equal(next.selectedCandidateId, 'import_abc');
  assert.equal(next.pending.importCandidate, false);
});

test('candidate/import/success does not duplicate an existing candidate', () => {
  const state = createState({
    candidates: [{ id: 'soccana' }, { id: 'import_abc' }],
  });
  const next = benchmarkLabReducer(state, {
    type: 'candidate/import/success',
    candidate: { id: 'import_abc', label: 'same' },
  });

  assert.equal(next.candidates.length, 2);
});

test('benchmark/run/success clears pending and switches to leaderboard', () => {
  const state = createState({
    tab: 'setup',
    pending: { ...createState().pending, runBenchmark: true },
  });
  const next = benchmarkLabReducer(state, { type: 'benchmark/run/success' });

  assert.equal(next.pending.runBenchmark, false);
  assert.equal(next.tab, 'leaderboard');
  assert.equal(next.errors.benchmark, '');
});

test('benchmark/run/error surfaces error and clears pending', () => {
  const state = createState({
    pending: { ...createState().pending, runBenchmark: true },
  });
  const next = benchmarkLabReducer(state, {
    type: 'benchmark/run/error',
    message: 'No clip ready',
  });

  assert.equal(next.pending.runBenchmark, false);
  assert.equal(next.errors.benchmark, 'No clip ready');
});

test('operation errors persist across benchmark list refreshes', () => {
  const state = benchmarkLabReducer(
    createState(),
    { type: 'operation/error', message: 'Import failed silently.' },
  );

  const afterRefresh = benchmarkLabReducer(state, {
    type: 'benchmarks/loaded',
    benchmarks: [{ benchmark_id: 'b1', status: 'completed' }],
  });

  assert.equal(afterRefresh.errors.operation, 'Import failed silently.');
});

test('operation/clear removes the operation error', () => {
  const state = createState({
    errors: { global: '', operation: 'stale error', clip: '', candidates: '', benchmark: '' },
  });
  const next = benchmarkLabReducer(state, { type: 'operation/clear' });
  assert.equal(next.errors.operation, '');
});

test('clip/ensure flow tracks pending state correctly', () => {
  let state = createState();

  state = benchmarkLabReducer(state, { type: 'clip/ensure/start' });
  assert.equal(state.pending.ensureClip, true);
  assert.equal(state.errors.clip, '');

  state = benchmarkLabReducer(state, {
    type: 'clip/ensure/success',
    clipStatus: { ready: true, path: '/tmp/clip.mp4' },
  });
  assert.equal(state.pending.ensureClip, false);
  assert.equal(state.clipStatus.ready, true);
});

test('benchmarks/loaded resolves selectedBenchmarkId to active benchmark', () => {
  const state = createState({ selectedBenchmarkId: 'stale_id' });
  const next = benchmarkLabReducer(state, {
    type: 'benchmarks/loaded',
    benchmarks: [
      { benchmark_id: 'b_done', status: 'completed' },
      { benchmark_id: 'b_active', status: 'running' },
    ],
  });

  // Should prefer the active (running) benchmark
  assert.equal(next.selectedBenchmarkId, 'b_active');
});
