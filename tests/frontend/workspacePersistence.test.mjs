import assert from 'node:assert/strict';
import test from 'node:test';

import { STORAGE_KEYS as TRAINING_STORAGE_KEYS } from '../../frontend/src/trainingStudio/state.js';
import {
  APP_STORAGE_KEYS,
  buildWorkspaceResetErrorMessage,
  clearSavedWorkspace,
  getAllWorkspaceStorageKeys,
  resolveAppShellErrors,
} from '../../frontend/src/workspacePersistence.js';

test('workspace reset covers both analysis and training persistence keys', () => {
  const keys = getAllWorkspaceStorageKeys();

  assert.ok(keys.includes(APP_STORAGE_KEYS.appSpace));
  assert.ok(keys.includes(APP_STORAGE_KEYS.form));
  assert.ok(keys.includes(TRAINING_STORAGE_KEYS.datasetPath));
  assert.ok(keys.includes(TRAINING_STORAGE_KEYS.selectedJobId));
  assert.equal(keys.length, new Set(keys).size);
});

test('clearSavedWorkspace removes both analysis and training keys from storage', () => {
  const store = new Map(
    getAllWorkspaceStorageKeys().map((key) => [key, 'persisted-value']),
  );
  const storage = {
    removeItem(key) {
      store.delete(key);
    },
  };

  clearSavedWorkspace(storage);

  assert.equal(store.size, 0);
});

test('buildWorkspaceResetErrorMessage preserves the underlying storage failure detail', () => {
  const error = new Error('QuotaExceededError');

  assert.equal(
    buildWorkspaceResetErrorMessage(error),
    'Could not clear saved workspace. QuotaExceededError',
  );
});

test('resolveAppShellErrors keeps workspace reset failures visible alongside job errors', () => {
  assert.deepEqual(
    resolveAppShellErrors({
      workspaceError: 'Could not clear saved workspace. QuotaExceededError',
      jobError: 'Run failed to start',
    }),
    ['Could not clear saved workspace. QuotaExceededError', 'Run failed to start'],
  );
});
