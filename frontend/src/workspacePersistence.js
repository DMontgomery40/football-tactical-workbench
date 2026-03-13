import { STORAGE_KEYS as TRAINING_STORAGE_KEYS } from './trainingStudio/state.js';

const BENCHMARK_STORAGE_KEYS = {
  benchmarkLabel: 'fpw.benchmarkLabel',
  clipSourcePath: 'fpw.benchmarkClipSourcePath',
  selectedBenchmarkId: 'fpw.benchmarkSelectedBenchmarkId',
  selectedRecipeIds: 'fpw.benchmarkSelectedRecipeIds',
  selectedSuiteIds: 'fpw.benchmarkSelectedSuiteIds',
  detailSelection: 'fpw.benchmarkDetailSelection',
};

export const APP_STORAGE_KEYS = {
  appSpace: 'fpw.appSpace',
  themeMode: 'fpw.themeMode',
  form: 'fpw.form',
  soccerNetSplit: 'fpw.soccerNetSplit',
  soccerNetQuery: 'fpw.soccerNetQuery',
  soccerNetFiles: 'fpw.soccerNetFiles',
  folderPath: 'fpw.folderPath',
  localVideoPath: 'fpw.localVideoPath',
  analysisSidebarWidth: 'fpw.analysisSidebarWidth',
};

export function getAllWorkspaceStorageKeys() {
  return [...new Set([...Object.values(APP_STORAGE_KEYS), ...Object.values(TRAINING_STORAGE_KEYS), ...Object.values(BENCHMARK_STORAGE_KEYS)])];
}

export function clearSavedWorkspace(storage) {
  for (const key of getAllWorkspaceStorageKeys()) {
    storage.removeItem(key);
  }
}

export function buildWorkspaceResetErrorMessage(error) {
  return error instanceof Error && error.message
    ? `Could not clear saved workspace. ${error.message}`
    : 'Could not clear saved workspace.';
}

export function resolveAppShellErrors({ workspaceError = '', jobError = '' } = {}) {
  return [workspaceError, jobError].filter(Boolean);
}
