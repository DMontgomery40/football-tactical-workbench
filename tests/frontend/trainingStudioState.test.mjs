import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_FORM, trainingStudioReducer } from '../../frontend/src/trainingStudio/state.js';

function createState(overrides = {}) {
  return {
    studioTab: 'jobs',
    datasetPath: '',
    scannedDatasetPath: '',
    trainingConfig: null,
    registry: null,
    jobs: [],
    datasetScan: null,
    form: { ...DEFAULT_FORM },
    selectedJobId: '',
    selectedRegistryEntryId: '',
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
    bootstrapped: true,
    ...overrides,
  };
}

test('operation errors persist across polling-driven jobs and registry refreshes', () => {
  const operationState = trainingStudioReducer(
    createState(),
    { type: 'operation/error', message: 'Detector activated, but the registry refresh failed.' },
  );

  const afterJobsRefresh = trainingStudioReducer(operationState, {
    type: 'jobs/loaded',
    jobs: [{ job_id: 'job-1', status: 'running' }],
  });
  const afterRegistryRefresh = trainingStudioReducer(afterJobsRefresh, {
    type: 'registry/loaded',
    registry: {
      active_detector: 'custom_live',
      detectors: [{ id: 'custom_live', label: 'Custom live detector' }],
    },
  });

  assert.equal(afterJobsRefresh.errors.operation, 'Detector activated, but the registry refresh failed.');
  assert.equal(afterRegistryRefresh.errors.operation, 'Detector activated, but the registry refresh failed.');
});

test('operation errors clear when the operator starts a fresh activation flow', () => {
  const operationState = createState({
    errors: {
      global: '',
      operation: 'Detector activated, but the registry refresh failed.',
      scan: '',
      jobs: '',
      registry: '',
      train: '',
    },
  });

  const nextState = trainingStudioReducer(operationState, {
    type: 'run/activate/start',
    runId: 'run-1',
  });

  assert.equal(nextState.errors.operation, '');
  assert.equal(nextState.pending.activateRunId, 'run-1');
});
