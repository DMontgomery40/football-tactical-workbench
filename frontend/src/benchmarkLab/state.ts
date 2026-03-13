import { readStoredJson, readStoredString } from '../trainingStudio/storage.js';
import type {
  BenchmarkFilterState,
  BenchmarkHistoryItem,
  BenchmarkRecipe,
  BenchmarkRunDetail,
  BenchmarkSortState,
  BenchmarkSuite,
  BenchmarkViewPreset,
} from './types';

export const ACTIVE_BENCHMARK_STATUSES = new Set(['queued', 'running']);

export const STORAGE_KEYS = {
  benchmarkLabel: 'fpw.benchmarkLabel',
  activeSuiteId: 'fpw.benchmarkActiveSuiteId',
  clipSourcePath: 'fpw.benchmarkClipSourcePath',
  filters: 'fpw.benchmarkFilters',
  selectedBenchmarkId: 'fpw.benchmarkSelectedBenchmarkId',
  selectedRecipeId: 'fpw.benchmarkSelectedRecipeId',
  selectedRecipeIds: 'fpw.benchmarkSelectedRecipeIds',
  selectedSuiteIds: 'fpw.benchmarkSelectedSuiteIds',
  detailSelection: 'fpw.benchmarkDetailSelection',
  sort: 'fpw.benchmarkSort',
  viewPreset: 'fpw.benchmarkViewPreset',
};

export const DEFAULT_FILTERS: BenchmarkFilterState = {
  search: '',
  suite: 'all',
  tier: 'all',
  provider: 'all',
  architecture: 'all',
  bundleMode: 'all',
  status: 'all',
  capability: 'all',
  supportsActiveSuite: 'all',
  hasNa: 'all',
};

export const DEFAULT_SORT: BenchmarkSortState = {
  column: '',
  direction: 'desc',
};

export const VIEW_PRESETS: BenchmarkViewPreset[] = [
  'Detection',
  'Spotting',
  'Localization',
  'Calibration',
  'Tracking',
  'Game State',
  'Operational',
];

export interface DetailSelection {
  suiteId: string;
  recipeId: string;
}

export function isActiveBenchmarkStatus(status: string): boolean {
  return ACTIVE_BENCHMARK_STATUSES.has(String(status || ''));
}

export function readStoredIdList(key: string): string[] {
  const raw = readStoredJson(key, []);
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .map((value) => String(value || '').trim())
    .filter(Boolean);
}

export function readStoredDetailSelection(): DetailSelection | null {
  const raw = readStoredJson(STORAGE_KEYS.detailSelection, null);
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const suiteId = String((raw as { suiteId?: unknown }).suiteId || '').trim();
  const recipeId = String((raw as { recipeId?: unknown }).recipeId || '').trim();
  return suiteId && recipeId ? { suiteId, recipeId } : null;
}

export function readStoredBenchmarkLabel(): string {
  return readStoredString(STORAGE_KEYS.benchmarkLabel, '');
}

export function readStoredActiveSuiteId(): string {
  return readStoredString(STORAGE_KEYS.activeSuiteId, '');
}

export function readStoredClipSourcePath(): string {
  return readStoredString(STORAGE_KEYS.clipSourcePath, '');
}

export function readStoredBenchmarkId(): string {
  return readStoredString(STORAGE_KEYS.selectedBenchmarkId, '');
}

export function readStoredBenchmarkFilters(): BenchmarkFilterState {
  const raw = readStoredJson(STORAGE_KEYS.filters, DEFAULT_FILTERS);
  return {
    ...DEFAULT_FILTERS,
    ...(raw && typeof raw === 'object' ? raw : {}),
  };
}

export function readStoredBenchmarkSort(): BenchmarkSortState {
  const raw = readStoredJson(STORAGE_KEYS.sort, DEFAULT_SORT);
  return {
    ...DEFAULT_SORT,
    ...(raw && typeof raw === 'object' ? raw : {}),
  };
}

export function readStoredSelectedSuites(): string[] {
  return readStoredIdList(STORAGE_KEYS.selectedSuiteIds);
}

export function readStoredSelectedRecipes(): string[] {
  return readStoredIdList(STORAGE_KEYS.selectedRecipeIds);
}

export function readStoredViewPreset(): BenchmarkViewPreset {
  const value = readStoredString(STORAGE_KEYS.viewPreset, 'Detection');
  return VIEW_PRESETS.includes(value as BenchmarkViewPreset) ? (value as BenchmarkViewPreset) : 'Detection';
}

export function sameStringArray(left: string[], right: string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((value, index) => value === right[index]);
}

export function sameDetailSelection(left: DetailSelection | null, right: DetailSelection | null): boolean {
  return left?.suiteId === right?.suiteId && left?.recipeId === right?.recipeId;
}

export function mergeUniqueStrings(existing: string[], next: string[]): string[] {
  const merged = [...existing];
  for (const value of next) {
    if (value && !merged.includes(value)) {
      merged.push(value);
    }
  }
  return merged;
}

export function resolveSelectedBenchmarkId(
  currentId: string,
  history: BenchmarkHistoryItem[],
): string {
  if (history.length === 0) {
    return '';
  }
  if (currentId && history.some((item) => item.benchmarkId === currentId)) {
    return currentId;
  }
  const activeItem = history.find((item) => isActiveBenchmarkStatus(item.status));
  return activeItem?.benchmarkId || history[0]?.benchmarkId || '';
}

export function resolveActiveSuiteId(
  currentId: string,
  suites: BenchmarkSuite[],
): string {
  if (suites.length === 0) {
    return '';
  }
  if (currentId && suites.some((suite) => suite.id === currentId)) {
    return currentId;
  }
  return suites[0]?.id || '';
}

export function resolveVisibleActiveSuiteId(
  currentId: string,
  visibleSuites: BenchmarkSuite[],
  allSuites: BenchmarkSuite[],
): string {
  if (visibleSuites.some((suite) => suite.id === currentId)) {
    return currentId;
  }
  if (visibleSuites.length > 0) {
    return visibleSuites[0]?.id || '';
  }
  if (allSuites.some((suite) => suite.id === currentId)) {
    return currentId;
  }
  return allSuites[0]?.id || '';
}

export function resolveSelectedSuiteIds(
  currentIds: string[],
  suites: BenchmarkSuite[],
  readySuiteIds: Set<string>,
): string[] {
  if (suites.length === 0) {
    return [];
  }
  const valid = currentIds.filter((suiteId) => suites.some((suite) => suite.id === suiteId));
  if (valid.length > 0) {
    return valid;
  }
  const operationalSuite = suites.find((suite) => suite.protocol === 'operational');
  if (operationalSuite) {
    return [operationalSuite.id];
  }
  const readySuite = suites.find((suite) => readySuiteIds.has(suite.id));
  if (readySuite) {
    return [readySuite.id];
  }
  return [suites[0]?.id].filter(Boolean);
}

export function resolveSelectedRecipeIds(
  currentIds: string[],
  recipes: BenchmarkRecipe[],
  preferredRecipeId: string,
): string[] {
  if (recipes.length === 0) {
    return [];
  }
  const valid = currentIds.filter((recipeId) => recipes.some((recipe) => recipe.id === recipeId));
  if (valid.length > 0) {
    return valid;
  }
  if (preferredRecipeId && recipes.some((recipe) => recipe.id === preferredRecipeId)) {
    return [preferredRecipeId];
  }
  const available = recipes.find((recipe) => recipe.available);
  return [available?.id || recipes[0]?.id].filter(Boolean);
}

export function resolveDetailSelection(
  currentSelection: DetailSelection | null,
  currentRun: BenchmarkRunDetail | null,
  selectedSuiteIds: string[],
  selectedRecipeIds: string[],
): DetailSelection | null {
  const suitePool = selectedSuiteIds.length > 0
    ? selectedSuiteIds
    : (currentRun?.suiteIds?.length ? currentRun.suiteIds : []);
  const recipePool = selectedRecipeIds.length > 0
    ? selectedRecipeIds
    : (currentRun?.recipeIds?.length ? currentRun.recipeIds : []);
  if (suitePool.length === 0 || recipePool.length === 0) {
    return null;
  }

  const currentIsValid = Boolean(
    currentSelection
      && suitePool.includes(currentSelection.suiteId)
      && recipePool.includes(currentSelection.recipeId),
  );
  if (currentIsValid) {
    return currentSelection;
  }

  if (currentRun) {
    for (const suiteId of suitePool) {
      for (const recipeId of recipePool) {
        return { suiteId, recipeId };
      }
    }
  }

  return {
    suiteId: suitePool[0],
    recipeId: recipePool[0],
  };
}
