import { startTransition, useEffect, useMemo, useState } from 'react';

import { buildHelpIndex } from '../helpUi';
import {
  ensureBenchmarkClip,
  fetchBenchmarkConfig,
  fetchBenchmarkHistory,
  fetchBenchmarkJob,
  importBenchmarkHf,
  importBenchmarkLocal,
  runBenchmark,
} from '../lib/api/contracts';
import { writeStoredJson, writeStoredValue } from '../trainingStudio/storage.js';
import AssetBrowser from './AssetBrowser';
import BenchmarkCharts from './BenchmarkCharts';
import {
  type BenchmarkChartMode,
  metricOptionsForSuite,
  reconcileFocusedRecipeIds,
  toggleFocusedRecipeId,
} from './chartData';
import DetailPanel from './DetailPanel';
import MatrixToolbar from './MatrixToolbar';
import OperationalReviewCard from './OperationalReviewCard';
import { cx, panelClass, sectionHeadingClass } from './ui';
import ResultsMatrix from './ResultsMatrix';
import RunControls from './RunControls';
import SuiteEvaluationCharts from './SuiteEvaluationCharts';
import {
  ACTIVE_BENCHMARK_STATUSES,
  STORAGE_KEYS,
  VIEW_PRESETS,
  mergeUniqueStrings,
  readStoredActiveSuiteId,
  readStoredBenchmarkFilters,
  readStoredBenchmarkId,
  readStoredBenchmarkLabel,
  readStoredBenchmarkSort,
  readStoredClipSourcePath,
  readStoredDetailSelection,
  readStoredIdList,
  resolveVisibleActiveSuiteId,
  readStoredViewPreset,
  resolveDetailSelection,
  resolveSelectedBenchmarkId,
  resolveSelectedRecipeIds,
  resolveSelectedSuiteIds,
  sameDetailSelection,
  sameStringArray,
  type DetailSelection,
} from './state';
import SuiteSelector from './SuiteSelector';
import type {
  BenchmarkConfigSnapshot,
  BenchmarkFilterState,
  BenchmarkHistoryItem,
  BenchmarkRecipe,
  BenchmarkRunDetail,
  BenchmarkRunResult,
  BenchmarkSortState,
  BenchmarkSuite,
  BenchmarkViewPreset,
  ClipStatus,
  ImportResponse,
} from './types';
import {
  buildCellKey,
  normalizeBenchmarkConfig,
  normalizeBenchmarkHistory,
  normalizeBenchmarkRunDetail,
  normalizeClipStatus,
  normalizeImportResponse,
  pickPreferredRecipeId,
} from './types';
import {
  collectMatrixFilterOptions,
  filterRecipesForMatrix,
  sortRecipesForMatrix,
  suiteFamilyForPreset,
} from './utils';

interface BenchmarkLabShellProps {
  apiBase: string;
  helpCatalog?: unknown[];
  activePipeline?: string;
  activeDetector?: string;
}

interface PreviewCell {
  status: string;
  reason: string;
}

function recommendedSortDirection(metricId: string): 'asc' | 'desc' {
  return /latency|_ms$/i.test(metricId) ? 'asc' : 'desc';
}

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function createFallbackSuite(suiteId: string): BenchmarkSuite {
  return {
    id: suiteId,
    label: suiteId,
    tier: '',
    family: '',
    protocol: '',
    primaryMetric: '',
    metricColumns: [],
    requiredCapabilities: [],
    datasetRoot: '',
    manifestPath: '',
    sourceUrl: '',
    license: '',
    notes: '',
    datasetSplit: null,
    dvcRequired: false,
    requiresClip: false,
    fallbackDatasetRoots: [],
  };
}

function patchConfigWithClipStatus(
  snapshot: BenchmarkConfigSnapshot | null,
  clipStatus: ClipStatus,
): BenchmarkConfigSnapshot | null {
  if (!snapshot) {
    return snapshot;
  }
  const nextStates = snapshot.datasetStates.map((state) => {
    if (state.suiteId !== 'ops.clip_review_v1') {
      return state;
    }
    return {
      ...state,
      ready: clipStatus.ready,
      datasetRoot: clipStatus.path,
      datasetExists: clipStatus.ready,
      datasetDvc: clipStatus.dvc,
      note: clipStatus.note,
    };
  });
  return {
    ...snapshot,
    datasetStates: nextStates,
    legacyClipStatus: clipStatus,
  };
}

function mergeImportedSnapshot(
  snapshot: BenchmarkConfigSnapshot | null,
  imported: ImportResponse,
): BenchmarkConfigSnapshot | null {
  if (!snapshot) {
    return snapshot;
  }

  const assetMap = new Map(snapshot.assets.map((asset) => [asset.assetId, asset]));
  if (imported.asset) {
    assetMap.set(imported.asset.assetId, imported.asset);
  }
  const recipeMap = new Map(snapshot.recipes.map((recipe) => [recipe.id, recipe]));
  for (const recipe of imported.recipes) {
    recipeMap.set(recipe.id, recipe);
  }

  return {
    ...snapshot,
    assets: Array.from(assetMap.values()),
    recipes: Array.from(recipeMap.values()),
  };
}

async function uploadBenchmarkClip(apiBase: string, file: File): Promise<ClipStatus> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${apiBase}/api/benchmark/ensure-clip-upload`, {
    method: 'POST',
    body: formData,
  });
  const rawText = await response.text();
  let payload: unknown = {};
  if (rawText) {
    try {
      payload = JSON.parse(rawText);
    } catch {
      payload = {};
    }
  }
  if (!response.ok) {
    const raw = asRecord(payload);
    throw new Error(
      String(raw.detail || raw.error || rawText.trim() || `Upload failed (${response.status})`),
    );
  }
  return normalizeClipStatus(payload);
}

function extractRunAcceptance(value: unknown): { benchmarkId: string; status: string } {
  const raw = asRecord(value);
  return {
    benchmarkId: String(raw.benchmark_id || '').trim(),
    status: String(raw.status || '').trim(),
  };
}

export default function BenchmarkLabShell({
  apiBase,
  helpCatalog = [],
  activePipeline = 'classic',
  activeDetector = 'soccana',
}: BenchmarkLabShellProps) {
  const helpIndex = useMemo(() => buildHelpIndex(helpCatalog), [helpCatalog]);

  const [config, setConfig] = useState<BenchmarkConfigSnapshot | null>(null);
  const [history, setHistory] = useState<BenchmarkHistoryItem[]>([]);
  const [currentRun, setCurrentRun] = useState<BenchmarkRunDetail | null>(null);
  const [selectedSuiteIds, setSelectedSuiteIds] = useState<string[]>(() => readStoredIdList(STORAGE_KEYS.selectedSuiteIds));
  const [selectedRecipeIds, setSelectedRecipeIds] = useState<string[]>(() => readStoredIdList(STORAGE_KEYS.selectedRecipeIds));
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string>(() => readStoredBenchmarkId());
  const [detailSelection, setDetailSelection] = useState<DetailSelection | null>(() => readStoredDetailSelection());
  const [clipSourcePath, setClipSourcePath] = useState<string>(() => readStoredClipSourcePath());
  const [benchmarkLabel, setBenchmarkLabel] = useState<string>(() => readStoredBenchmarkLabel());
  const [activeSuiteId, setActiveSuiteId] = useState<string>(() => readStoredActiveSuiteId());
  const [filters, setFilters] = useState<BenchmarkFilterState>(() => readStoredBenchmarkFilters());
  const [sort, setSort] = useState<BenchmarkSortState>(() => readStoredBenchmarkSort());
  const [viewPreset, setViewPreset] = useState<BenchmarkViewPreset>(() => readStoredViewPreset());
  const [railTab, setRailTab] = useState<'suites' | 'recipes'>('suites');
  const [chartMode, setChartMode] = useState<BenchmarkChartMode>('bar');
  const [chartMetricId, setChartMetricId] = useState('');
  const [focusedRecipeIds, setFocusedRecipeIds] = useState<string[]>([]);
  const [highlightedRecipeId, setHighlightedRecipeId] = useState('');

  const [bootstrapping, setBootstrapping] = useState(true);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [refreshingSnapshot, setRefreshingSnapshot] = useState(false);
  const [loadingBenchmarkDetail, setLoadingBenchmarkDetail] = useState(false);
  const [preparingClip, setPreparingClip] = useState(false);
  const [importingLocal, setImportingLocal] = useState(false);
  const [importingHf, setImportingHf] = useState(false);
  const [startingRun, setStartingRun] = useState(false);

  const [globalError, setGlobalError] = useState('');
  const [clipError, setClipError] = useState('');
  const [assetError, setAssetError] = useState('');
  const [runError, setRunError] = useState('');
  const [operationMessage, setOperationMessage] = useState('');

  const suiteById = useMemo(
    () => new Map((config?.suites || []).map((suite) => [suite.id, suite])),
    [config],
  );
  const datasetStateById = useMemo(
    () => new Map((config?.datasetStates || []).map((state) => [state.suiteId, state])),
    [config],
  );
  const assetById = useMemo(
    () => new Map((config?.assets || []).map((asset) => [asset.assetId, asset])),
    [config],
  );
  const configRecipeById = useMemo(
    () => new Map((config?.recipes || []).map((recipe) => [recipe.id, recipe])),
    [config],
  );
  const runRecipeById = useMemo(
    () => new Map((currentRun?.recipes || []).map((recipe) => [recipe.id, recipe])),
    [currentRun],
  );
  const mergedAssetById = useMemo(
    () => new Map([
      ...assetById.entries(),
      ...(currentRun?.assets || []).map((asset) => [asset.assetId, asset] as const),
    ]),
    [assetById, currentRun?.assets],
  );

  const readySuiteIds = useMemo(
    () => new Set((config?.datasetStates || []).filter((state) => state.ready).map((state) => state.suiteId)),
    [config],
  );

  const selectedSuites = useMemo(
    () => selectedSuiteIds.map((suiteId) => suiteById.get(suiteId)).filter(Boolean) as BenchmarkSuite[],
    [selectedSuiteIds, suiteById],
  );
  const preferredRecipeId = useMemo(
    () => pickPreferredRecipeId(config?.recipes || [], selectedSuites, activeDetector),
    [activeDetector, config?.recipes, selectedSuites],
  );
  const selectedRecipes = useMemo(
    () => selectedRecipeIds.map((recipeId) => configRecipeById.get(recipeId)).filter(Boolean) as BenchmarkRecipe[],
    [configRecipeById, selectedRecipeIds],
  );
  const selectedRecipePipelines = useMemo(
    () => Array.from(new Set(selectedRecipes.map((recipe) => recipe.pipeline || 'classic'))),
    [selectedRecipes],
  );

  const matrixSuiteIds = currentRun?.suiteIds?.length ? currentRun.suiteIds : selectedSuiteIds;
  const matrixRecipeIds = currentRun?.recipeIds?.length ? currentRun.recipeIds : selectedRecipeIds;
  const matrixSuites = useMemo(
    () => matrixSuiteIds.map((suiteId) => suiteById.get(suiteId) || createFallbackSuite(suiteId)),
    [matrixSuiteIds, suiteById],
  );
  const matrixRecipes = useMemo(
    () => matrixRecipeIds
      .map((recipeId) => runRecipeById.get(recipeId) || configRecipeById.get(recipeId))
      .filter(Boolean) as BenchmarkRecipe[],
    [configRecipeById, matrixRecipeIds, runRecipeById],
  );

  const selectionPreviewCells = useMemo<Record<string, PreviewCell>>(() => {
    const nextCells: Record<string, PreviewCell> = {};
    for (const suite of selectedSuites) {
      const datasetState = datasetStateById.get(suite.id);
      const suiteReady = Boolean(datasetState?.ready || (suite.requiresClip && config?.legacyClipStatus?.ready));
      for (const recipe of selectedRecipes) {
        let status = 'ready';
        let reason = 'Runnable now';
        if (!recipe.available) {
          status = 'unavailable';
          reason = 'Recipe assets are unavailable.';
        } else if (!recipe.compatibleSuiteIds.includes(suite.id)) {
          status = 'not_supported';
          reason = 'Recipe does not satisfy this suite capability contract.';
        } else if (!suiteReady) {
          const readinessNote = String(datasetState?.note || '');
          status = /still blocked|does not yet/i.test(readinessNote) ? 'blocked' : 'dataset_missing';
          reason = datasetState?.note || 'Benchmark dataset or clip is unavailable.';
        }
        nextCells[buildCellKey(suite.id, recipe.id)] = { status, reason };
      }
    }
    return nextCells;
  }, [config?.legacyClipStatus?.ready, datasetStateById, selectedRecipes, selectedSuites]);

  const matrixPreviewCells = useMemo<Record<string, PreviewCell>>(() => {
    const nextCells: Record<string, PreviewCell> = {};
    for (const suite of matrixSuites) {
      const datasetState = datasetStateById.get(suite.id);
      const suiteReady = Boolean(datasetState?.ready || (suite.requiresClip && config?.legacyClipStatus?.ready));
      for (const recipe of matrixRecipes) {
        let status = 'ready';
        let reason = 'Runnable now';
        if (!recipe.available) {
          status = 'unavailable';
          reason = 'Recipe assets are unavailable.';
        } else if (!recipe.compatibleSuiteIds.includes(suite.id)) {
          status = 'not_supported';
          reason = 'Recipe does not satisfy this suite capability contract.';
        } else if (!suiteReady) {
          status = 'dataset_missing';
          reason = datasetState?.note || 'Benchmark dataset or clip is unavailable.';
        }
        nextCells[buildCellKey(suite.id, recipe.id)] = { status, reason };
      }
    }
    return nextCells;
  }, [config?.legacyClipStatus?.ready, datasetStateById, matrixRecipes, matrixSuites]);

  const preflightSummary = useMemo(() => {
    let runnableCells = 0;
    let unsupportedCells = 0;
    let missingDatasetCells = 0;
    let unavailableCells = 0;

    for (const cell of Object.values(selectionPreviewCells)) {
      if (cell.status === 'ready') {
        runnableCells += 1;
      } else if (cell.status === 'not_supported') {
        unsupportedCells += 1;
      } else if (cell.status === 'dataset_missing' || cell.status === 'blocked') {
        missingDatasetCells += 1;
      } else if (cell.status === 'unavailable') {
        unavailableCells += 1;
      }
    }

    const totalCells = selectedSuites.length * selectedRecipes.length;
    return {
      totalCells,
      runnableCells,
      unsupportedCells,
      missingDatasetCells,
      unavailableCells,
      blockedCells: unsupportedCells + missingDatasetCells + unavailableCells,
    };
  }, [selectionPreviewCells, selectedRecipes.length, selectedSuites.length]);

  const presetFamily = useMemo(() => suiteFamilyForPreset(viewPreset), [viewPreset]);

  const visibleSuites = useMemo(() => {
    const nextSuites = matrixSuites.filter((suite) => {
      if (presetFamily && suite.family !== presetFamily) {
        return false;
      }
      if (filters.tier !== 'all' && suite.tier !== filters.tier) {
        return false;
      }
      if (filters.suite !== 'all' && suite.id !== filters.suite) {
        return false;
      }
      return true;
    });
    return nextSuites;
  }, [filters.suite, filters.tier, matrixSuites, presetFamily]);

  const effectiveActiveSuite = useMemo(() => {
    const resolvedSuiteId = resolveVisibleActiveSuiteId(activeSuiteId, visibleSuites, matrixSuites);
    return visibleSuites.find((suite) => suite.id === resolvedSuiteId)
      || matrixSuites.find((suite) => suite.id === resolvedSuiteId)
      || null;
  }, [activeSuiteId, matrixSuites, visibleSuites]);

  const matrixFilterOptions = useMemo(
    () => collectMatrixFilterOptions(matrixRecipes, effectiveActiveSuite, currentRun, matrixPreviewCells, mergedAssetById),
    [currentRun, effectiveActiveSuite, matrixPreviewCells, matrixRecipes, mergedAssetById],
  );
  const availableTiers = useMemo(
    () => Array.from(new Set(matrixSuites.map((suite) => suite.tier).filter(Boolean))).sort(),
    [matrixSuites],
  );
  const filteredMatrixRecipes = useMemo(
    () => filterRecipesForMatrix(
      matrixRecipes,
      effectiveActiveSuite,
      currentRun,
      matrixPreviewCells,
      filters,
      mergedAssetById,
    ),
    [currentRun, effectiveActiveSuite, filters, matrixPreviewCells, matrixRecipes, mergedAssetById],
  );
  const sortedMatrixRecipes = useMemo(
    () => sortRecipesForMatrix(
      filteredMatrixRecipes,
      effectiveActiveSuite,
      currentRun,
      matrixPreviewCells,
      sort,
      mergedAssetById,
    ),
    [currentRun, effectiveActiveSuite, filteredMatrixRecipes, matrixPreviewCells, mergedAssetById, sort],
  );
  const reconciledFocusedRecipeIds = useMemo(
    () => reconcileFocusedRecipeIds(focusedRecipeIds, sortedMatrixRecipes.map((recipe) => recipe.id)),
    [focusedRecipeIds, sortedMatrixRecipes],
  );
  const displayedMatrixRecipes = useMemo(() => {
    const visibleSet = new Set(reconciledFocusedRecipeIds);
    return sortedMatrixRecipes.filter((recipe) => visibleSet.has(recipe.id));
  }, [reconciledFocusedRecipeIds, sortedMatrixRecipes]);

  const selectedSuite = detailSelection
    ? visibleSuites.find((suite) => suite.id === detailSelection.suiteId)
      || suiteById.get(detailSelection.suiteId)
      || createFallbackSuite(detailSelection.suiteId)
    : effectiveActiveSuite;
  const selectedRecipe = detailSelection
    ? runRecipeById.get(detailSelection.recipeId) || configRecipeById.get(detailSelection.recipeId) || null
    : displayedMatrixRecipes[0] || null;
  const selectedResult: BenchmarkRunResult | null = detailSelection
    ? currentRun?.suiteResults?.[detailSelection.suiteId]?.[detailSelection.recipeId] || null
    : (selectedSuite && selectedRecipe ? currentRun?.suiteResults?.[selectedSuite.id]?.[selectedRecipe.id] || null : null);

  async function refreshConfigSnapshot(): Promise<BenchmarkConfigSnapshot> {
    const snapshot = normalizeBenchmarkConfig(await fetchBenchmarkConfig(apiBase));
    startTransition(() => {
      setConfig(snapshot);
    });
    return snapshot;
  }

  async function refreshHistorySnapshot(): Promise<BenchmarkHistoryItem[]> {
    const snapshot = normalizeBenchmarkHistory(await fetchBenchmarkHistory(apiBase, 40));
    startTransition(() => {
      setHistory(snapshot);
    });
    return snapshot;
  }

  async function loadBenchmarkDetail(benchmarkId: string): Promise<BenchmarkRunDetail> {
    const detail = normalizeBenchmarkRunDetail(await fetchBenchmarkJob(apiBase, benchmarkId));
    startTransition(() => {
      setCurrentRun(detail);
    });
    return detail;
  }

  async function refreshAll() {
    setRefreshingSnapshot(true);
    setOperationMessage('');
    const problems: string[] = [];

    try {
      await refreshConfigSnapshot();
      setGlobalError('');
    } catch (error) {
      problems.push(`config refresh failed: ${getErrorMessage(error, 'Could not refresh benchmark config.')}`);
    }

    try {
      await refreshHistorySnapshot();
    } catch (error) {
      problems.push(`history refresh failed: ${getErrorMessage(error, 'Could not refresh benchmark history.')}`);
    }

    if (selectedBenchmarkId) {
      try {
        await loadBenchmarkDetail(selectedBenchmarkId);
        setRunError('');
      } catch (error) {
        problems.push(`run detail refresh failed: ${getErrorMessage(error, 'Could not refresh benchmark detail.')}`);
      }
    }

    if (problems.length > 0) {
      setOperationMessage(problems.join(' '));
    }
    setRefreshingSnapshot(false);
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setBootstrapping(true);
      setGlobalError('');
      try {
        const [configSnapshot, historySnapshot] = await Promise.all([
          fetchBenchmarkConfig(apiBase),
          fetchBenchmarkHistory(apiBase, 40),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setConfig(normalizeBenchmarkConfig(configSnapshot));
          setHistory(normalizeBenchmarkHistory(historySnapshot));
        });
        setGlobalError('');
      } catch (error) {
        if (!cancelled) {
          setGlobalError(getErrorMessage(error, 'Could not load Benchmark Lab.'));
        }
      } finally {
        if (!cancelled) {
          setBootstrapping(false);
          setBootstrapped(true);
        }
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  useEffect(() => {
    if (!config) {
      return;
    }
    setSelectedSuiteIds((currentIds) => {
      const nextIds = resolveSelectedSuiteIds(currentIds, config.suites, readySuiteIds);
      return sameStringArray(currentIds, nextIds) ? currentIds : nextIds;
    });
  }, [config, readySuiteIds]);

  useEffect(() => {
    if (!config) {
      return;
    }
    setSelectedRecipeIds((currentIds) => {
      const nextIds = resolveSelectedRecipeIds(currentIds, config.recipes, preferredRecipeId);
      return sameStringArray(currentIds, nextIds) ? currentIds : nextIds;
    });
  }, [config, preferredRecipeId]);

  useEffect(() => {
    const nextActiveSuiteId = effectiveActiveSuite?.id || '';
    setActiveSuiteId((currentId) => (currentId === nextActiveSuiteId ? currentId : nextActiveSuiteId));
  }, [effectiveActiveSuite?.id]);

  useEffect(() => {
    if (!effectiveActiveSuite) {
      return;
    }
    const allowedColumns = new Set([
      'label',
      'status',
      'provider',
      'architecture',
      'bundleMode',
      ...effectiveActiveSuite.metricColumns,
    ]);
    if (!sort.column || !allowedColumns.has(sort.column)) {
      setSort({
        column: effectiveActiveSuite.primaryMetric || 'label',
        direction: recommendedSortDirection(effectiveActiveSuite.primaryMetric),
      });
    }
  }, [effectiveActiveSuite, sort.column]);

  useEffect(() => {
    setSelectedBenchmarkId((currentId) => {
      const nextId = resolveSelectedBenchmarkId(currentId, history);
      return currentId === nextId ? currentId : nextId;
    });
  }, [history]);

  useEffect(() => {
    setDetailSelection((currentSelection) => {
      const nextSelection = resolveDetailSelection(
        currentSelection,
        currentRun,
        visibleSuites.map((suite) => suite.id),
        displayedMatrixRecipes.map((recipe) => recipe.id),
      );
      return sameDetailSelection(currentSelection, nextSelection) ? currentSelection : nextSelection;
    });
  }, [currentRun, displayedMatrixRecipes, visibleSuites]);

  useEffect(() => {
    if (!effectiveActiveSuite) {
      setChartMetricId('');
      return;
    }
    const allowedMetricIds = metricOptionsForSuite(effectiveActiveSuite).map((option) => option.id);
    setChartMetricId((currentMetricId) => (
      currentMetricId && allowedMetricIds.includes(currentMetricId)
        ? currentMetricId
        : (allowedMetricIds[0] || '')
    ));
  }, [effectiveActiveSuite]);

  useEffect(() => {
    if (!bootstrapped || !selectedBenchmarkId || (history.length > 0 && !history.some((item) => item.benchmarkId === selectedBenchmarkId))) {
      setCurrentRun(null);
      setRunError('');
      return;
    }
    let cancelled = false;
    setLoadingBenchmarkDetail(true);
    setRunError('');
    setCurrentRun(null);

    fetchBenchmarkJob(apiBase, selectedBenchmarkId)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setCurrentRun(normalizeBenchmarkRunDetail(payload));
        });
      })
      .catch((error) => {
        if (!cancelled) {
          setRunError(getErrorMessage(error, 'Could not load benchmark detail.'));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingBenchmarkDetail(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [apiBase, bootstrapped, history, selectedBenchmarkId]);

  const selectedHistoryItem = useMemo(
    () => history.find((item) => item.benchmarkId === selectedBenchmarkId) || null,
    [history, selectedBenchmarkId],
  );
  const shouldPollActiveBenchmark = Boolean(
    selectedBenchmarkId
      && (ACTIVE_BENCHMARK_STATUSES.has(currentRun?.status || '')
        || ACTIVE_BENCHMARK_STATUSES.has(selectedHistoryItem?.status || '')),
  );

  useEffect(() => {
    if (!shouldPollActiveBenchmark || !selectedBenchmarkId) {
      return undefined;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const [historySnapshot, detailSnapshot] = await Promise.all([
          fetchBenchmarkHistory(apiBase, 40),
          fetchBenchmarkJob(apiBase, selectedBenchmarkId),
        ]);
        startTransition(() => {
          setHistory(normalizeBenchmarkHistory(historySnapshot));
          setCurrentRun(normalizeBenchmarkRunDetail(detailSnapshot));
        });
      } catch (error) {
        setOperationMessage(`Automatic benchmark refresh is stale. ${getErrorMessage(error, 'Could not poll the active benchmark.')}`);
      }
    }, 3000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [apiBase, selectedBenchmarkId, shouldPollActiveBenchmark]);

  useEffect(() => {
    writeStoredJson(STORAGE_KEYS.selectedSuiteIds, selectedSuiteIds);
  }, [selectedSuiteIds]);

  useEffect(() => {
    writeStoredJson(STORAGE_KEYS.selectedRecipeIds, selectedRecipeIds);
  }, [selectedRecipeIds]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.selectedBenchmarkId, selectedBenchmarkId);
  }, [selectedBenchmarkId]);

  useEffect(() => {
    writeStoredJson(STORAGE_KEYS.detailSelection, detailSelection);
  }, [detailSelection]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.clipSourcePath, clipSourcePath);
  }, [clipSourcePath]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.benchmarkLabel, benchmarkLabel);
  }, [benchmarkLabel]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.activeSuiteId, activeSuiteId);
  }, [activeSuiteId]);

  useEffect(() => {
    writeStoredJson(STORAGE_KEYS.filters, filters);
  }, [filters]);

  useEffect(() => {
    writeStoredJson(STORAGE_KEYS.sort, sort);
  }, [sort]);

  useEffect(() => {
    writeStoredValue(STORAGE_KEYS.viewPreset, viewPreset);
  }, [viewPreset]);

  async function handlePrepareClip(sourcePath: string) {
    setPreparingClip(true);
    setClipError('');
    setOperationMessage('');
    try {
      const clipStatus = normalizeClipStatus(await ensureBenchmarkClip(apiBase, sourcePath));
      startTransition(() => {
        setConfig((currentConfig) => patchConfigWithClipStatus(currentConfig, clipStatus));
      });
      try {
        await refreshConfigSnapshot();
      } catch (error) {
        setOperationMessage(`Prepared the operational clip, but the suite snapshot did not refresh. ${getErrorMessage(error, 'Could not refresh benchmark config.')}`);
      }
    } catch (error) {
      setClipError(getErrorMessage(error, 'Could not prepare the benchmark clip.'));
    } finally {
      setPreparingClip(false);
    }
  }

  async function handleUploadClip(file: File) {
    setPreparingClip(true);
    setClipError('');
    setOperationMessage('');
    try {
      const clipStatus = await uploadBenchmarkClip(apiBase, file);
      startTransition(() => {
        setConfig((currentConfig) => patchConfigWithClipStatus(currentConfig, clipStatus));
        setClipSourcePath('');
      });
      try {
        await refreshConfigSnapshot();
      } catch (error) {
        setOperationMessage(`Uploaded the operational clip, but the suite snapshot did not refresh. ${getErrorMessage(error, 'Could not refresh benchmark config.')}`);
      }
    } catch (error) {
      setClipError(getErrorMessage(error, 'Could not upload the benchmark clip.'));
    } finally {
      setPreparingClip(false);
    }
  }

  async function handleImportLocal(payload: { checkpointPath: string; label?: string }) {
    setImportingLocal(true);
    setAssetError('');
    setOperationMessage('');
    try {
      const imported = normalizeImportResponse(await importBenchmarkLocal(apiBase, {
        checkpoint_path: payload.checkpointPath,
        label: payload.label,
      }));
      startTransition(() => {
        setConfig((currentConfig) => mergeImportedSnapshot(currentConfig, imported));
        setSelectedRecipeIds((currentIds) => mergeUniqueStrings(currentIds, imported.recipes.map((recipe) => recipe.id)));
      });
      try {
        await refreshConfigSnapshot();
      } catch (error) {
        setOperationMessage(`Imported the local checkpoint, but the asset catalog did not fully refresh. ${getErrorMessage(error, 'Could not refresh benchmark config.')}`);
      }
    } catch (error) {
      setAssetError(getErrorMessage(error, 'Could not import the local checkpoint.'));
    } finally {
      setImportingLocal(false);
    }
  }

  async function handleImportHf(payload: { repoId: string; filename?: string; label?: string }) {
    setImportingHf(true);
    setAssetError('');
    setOperationMessage('');
    try {
      const imported = normalizeImportResponse(await importBenchmarkHf(apiBase, {
        repo_id: payload.repoId,
        filename: payload.filename,
        label: payload.label,
      }));
      startTransition(() => {
        setConfig((currentConfig) => mergeImportedSnapshot(currentConfig, imported));
        setSelectedRecipeIds((currentIds) => mergeUniqueStrings(currentIds, imported.recipes.map((recipe) => recipe.id)));
      });
      try {
        await refreshConfigSnapshot();
      } catch (error) {
        setOperationMessage(`Imported the Hugging Face checkpoint, but the asset catalog did not fully refresh. ${getErrorMessage(error, 'Could not refresh benchmark config.')}`);
      }
    } catch (error) {
      setAssetError(getErrorMessage(error, 'Could not import the Hugging Face checkpoint.'));
    } finally {
      setImportingHf(false);
    }
  }

  async function handleRunSelectedMatrix() {
    setStartingRun(true);
    setRunError('');
    setOperationMessage('');
    try {
      const accepted = extractRunAcceptance(await runBenchmark(apiBase, {
        suite_ids: selectedSuiteIds,
        recipe_ids: selectedRecipeIds,
        label: benchmarkLabel.trim() || undefined,
      }));
      if (accepted.benchmarkId) {
        setSelectedBenchmarkId(accepted.benchmarkId);
      }

      const problems: string[] = [];

      try {
        await refreshHistorySnapshot();
      } catch (error) {
        problems.push(`history refresh failed: ${getErrorMessage(error, 'Could not refresh benchmark history.')}`);
      }

      if (accepted.benchmarkId) {
        try {
          await loadBenchmarkDetail(accepted.benchmarkId);
        } catch (error) {
          problems.push(`run detail refresh failed: ${getErrorMessage(error, 'Could not refresh benchmark detail.')}`);
        }
      }

      if (problems.length > 0) {
        setOperationMessage(`Benchmark started, but ${problems.join(' ')}`);
      }
    } catch (error) {
      setRunError(getErrorMessage(error, 'Could not start the benchmark matrix.'));
    } finally {
      setStartingRun(false);
    }
  }

  function handleSetViewPreset(nextPreset: BenchmarkViewPreset) {
    setViewPreset(nextPreset);
    const family = suiteFamilyForPreset(nextPreset);
    const matchingSuites = matrixSuites.filter((suite) => !family || suite.family === family);
    const nextSuiteId = matchingSuites[0]?.id || matrixSuites[0]?.id || '';
    setFilters((currentFilters) => ({
      ...currentFilters,
      tier: 'all',
      suite: nextSuiteId || currentFilters.suite,
    }));
    if (nextSuiteId) {
      setActiveSuiteId(nextSuiteId);
      const nextSuite = matchingSuites[0] || matrixSuites.find((suite) => suite.id === nextSuiteId);
      setSort({
        column: nextSuite?.primaryMetric || 'label',
        direction: recommendedSortDirection(nextSuite?.primaryMetric || ''),
      });
    }
  }

  function handleUpdateFilters(patch: Partial<BenchmarkFilterState>) {
    setFilters((currentFilters) => ({ ...currentFilters, ...patch }));
    if (patch.suite && patch.suite !== 'all') {
      setActiveSuiteId(patch.suite);
      const nextSuite = matrixSuites.find((suite) => suite.id === patch.suite);
      if (nextSuite) {
        setSort({
          column: nextSuite.primaryMetric || 'label',
          direction: recommendedSortDirection(nextSuite.primaryMetric),
        });
      }
    }
  }

  function handleResetFilters() {
    setFilters({
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
    });
  }

  function handleResetSort() {
    if (effectiveActiveSuite) {
      setSort({
        column: effectiveActiveSuite.primaryMetric || 'label',
        direction: recommendedSortDirection(effectiveActiveSuite.primaryMetric),
      });
      return;
    }
    setSort({ column: 'label', direction: 'desc' });
  }

  function handleSetActiveSuiteFocus(suiteId: string) {
    setActiveSuiteId(suiteId);
    setFilters((currentFilters) => ({ ...currentFilters, suite: suiteId }));
    setDetailSelection((currentSelection) => (
      currentSelection
        ? { suiteId, recipeId: currentSelection.recipeId }
        : currentSelection
    ));
    const nextSuite = matrixSuites.find((suite) => suite.id === suiteId);
    if (nextSuite) {
      setSort({
        column: nextSuite.primaryMetric || 'label',
        direction: recommendedSortDirection(nextSuite.primaryMetric),
      });
    }
  }

  function handleSelectRecipeFromChart(recipeId: string) {
    if (!effectiveActiveSuite) {
      return;
    }
    setDetailSelection({ suiteId: effectiveActiveSuite.id, recipeId });
    setHighlightedRecipeId(recipeId);
  }

  const shellHeader = (
    <header className="relative overflow-hidden rounded-[1.9rem] border border-[color:var(--line)] bg-[var(--surface)] shadow-[0_30px_80px_rgba(15,23,42,0.08)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(31,95,146,0.18),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(15,23,42,0.1),transparent_42%)]" />
      <div className="absolute inset-x-0 top-0 h-px bg-[linear-gradient(90deg,transparent,rgba(31,95,146,0.65),transparent)]" />
      <div className="relative grid gap-6 p-6 xl:grid-cols-[minmax(0,1fr)_auto] xl:p-8">
        <div className="space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full border border-[color:var(--line)] bg-[var(--surface)]/85 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-muted)]">
            Benchmark Lab
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
            suite / assets / recipes
          </div>
          <div className="max-w-4xl space-y-3">
            <h1 className="text-3xl font-semibold tracking-[-0.04em] text-[var(--text-strong)] [font-family:'Iowan_Old_Style','Palatino_Linotype','Book_Antiqua',Palatino,serif] md:text-4xl">
              Build a benchmark matrix the backend can actually execute.
            </h1>
            <p className="max-w-3xl text-base leading-7 text-[var(--text-muted)]">
              The refactored lab is recipe-first and suite-aware: pick protocols, bind assets through recipes, then
              inspect the resulting matrix, detail view, and operational overlay evidence without leaving the same
              workspace.
            </p>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:w-[24rem]">
          <div className="rounded-[1.25rem] border border-[color:var(--line)] bg-[var(--surface)]/85 p-4">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Catalog</div>
            <div className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">
              {config?.recipes.length || 0}
            </div>
            <div className="mt-1 text-sm text-[var(--text-muted)]">recipes across {config?.assets.length || 0} assets</div>
          </div>
          <div className="rounded-[1.25rem] border border-[color:var(--line)] bg-[var(--surface)]/85 p-4">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Protocol surface</div>
            <div className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">
              {config?.suites.length || 0}
            </div>
            <div className="mt-1 text-sm text-[var(--text-muted)]">benchmark suites currently exposed</div>
          </div>
        </div>
      </div>
    </header>
  );

  if (!bootstrapped && bootstrapping) {
    return (
      <section className="space-y-6 overflow-x-hidden">
        {shellHeader}
        <div className="rounded-[1.5rem] border border-[color:var(--line)] bg-[var(--surface)] p-6 text-sm text-[var(--text-muted)]">
          Loading Benchmark Lab…
        </div>
      </section>
    );
  }

  return (
    <section className="space-y-6 overflow-x-hidden text-[var(--text)]">
      {shellHeader}

      {globalError ? <div className="error-box">{globalError}</div> : null}

      {!config ? (
        <div className="rounded-[1.5rem] border border-[color:var(--line)] bg-[var(--surface)] p-6">
          <p className="text-sm leading-6 text-[var(--text-muted)]">
            Benchmark Lab could not finish booting. Try a manual refresh once the backend is available.
          </p>
          <button className="mt-4 inline-flex rounded-full border border-[color:var(--accent)] bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white" type="button" onClick={() => refreshAll()}>
            Retry snapshot
          </button>
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(20rem,24rem)_minmax(0,1fr)] min-[1800px]:grid-cols-[minmax(20rem,24rem)_minmax(0,1fr)_minmax(21rem,28rem)]">
          <div className="space-y-6 xl:sticky xl:top-4 xl:self-start">
            <section className={cx(panelClass, 'space-y-4')}>
              <div className="space-y-2">
                <div className={sectionHeadingClass}>Workspace Rail</div>
                <p className="text-sm leading-6 text-[var(--text-muted)]">
                  Choose the benchmark evidence first, then bind the recipe catalog without carrying both surfaces open at once.
                </p>
              </div>

              <div className="flex items-center gap-6 border-b border-[color:var(--line)]">
                <button
                  className={cx(
                    'border-b-2 pb-3 text-sm font-semibold transition',
                    railTab === 'suites'
                      ? 'border-[color:var(--accent)] text-[var(--accent-strong)]'
                      : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
                  )}
                  type="button"
                  onClick={() => setRailTab('suites')}
                >
                  Suites ({selectedSuiteIds.length})
                </button>
                <button
                  className={cx(
                    'border-b-2 pb-3 text-sm font-semibold transition',
                    railTab === 'recipes'
                      ? 'border-[color:var(--accent)] text-[var(--accent-strong)]'
                      : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
                  )}
                  type="button"
                  onClick={() => setRailTab('recipes')}
                >
                  Recipes ({selectedRecipeIds.length})
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-3 text-xs font-medium uppercase tracking-[0.14em] text-[var(--text-muted)]">
                <span>{selectedSuites.length} suite selections</span>
                <span>{selectedRecipes.length} recipe selections</span>
                <span>{config.assets.length} catalog assets</span>
              </div>
            </section>

            {railTab === 'suites' ? (
              <SuiteSelector
                suites={config.suites}
                datasetStates={config.datasetStates}
                selectedSuiteIds={selectedSuiteIds}
                clipStatus={config.legacyClipStatus}
                clipSourcePath={clipSourcePath}
                isPreparingClip={preparingClip}
                error={clipError}
                helpIndex={helpIndex}
                onClipSourcePathChange={setClipSourcePath}
                onToggleSuite={(suiteId) => {
                  setSelectedSuiteIds((currentIds) => (
                    currentIds.includes(suiteId)
                      ? currentIds.filter((value) => value !== suiteId)
                      : [...currentIds, suiteId]
                  ));
                }}
                onPrepareClip={handlePrepareClip}
                onUploadClip={handleUploadClip}
              />
            ) : (
              <AssetBrowser
                assets={config.assets}
                recipes={config.recipes}
                selectedSuiteIds={selectedSuiteIds}
                selectedRecipeIds={selectedRecipeIds}
                isImportingLocal={importingLocal}
                isImportingHf={importingHf}
                error={assetError}
                helpIndex={helpIndex}
                onToggleRecipe={(recipeId) => {
                  setSelectedRecipeIds((currentIds) => (
                    currentIds.includes(recipeId)
                      ? currentIds.filter((value) => value !== recipeId)
                      : [...currentIds, recipeId]
                  ));
                }}
                onImportLocal={handleImportLocal}
                onImportHf={handleImportHf}
              />
            )}
          </div>

          <div className="min-w-0 space-y-6">
            <RunControls
              benchmarkLabel={benchmarkLabel}
              history={history}
              selectedBenchmarkId={selectedBenchmarkId}
              currentRun={currentRun}
              selectedRecipePipelines={selectedRecipePipelines}
              activePipeline={activePipeline}
              selectedSuiteCount={selectedSuites.length}
              selectedRecipeCount={selectedRecipes.length}
              totalCells={preflightSummary.totalCells}
              runnableCells={preflightSummary.runnableCells}
              blockedCells={preflightSummary.blockedCells}
              unsupportedCells={preflightSummary.unsupportedCells}
              missingDatasetCells={preflightSummary.missingDatasetCells}
              unavailableCells={preflightSummary.unavailableCells}
              isRunning={startingRun}
              isRefreshing={refreshingSnapshot || loadingBenchmarkDetail}
              error={runError}
              operationMessage={operationMessage}
              helpIndex={helpIndex}
              onBenchmarkLabelChange={setBenchmarkLabel}
              onSelectBenchmark={setSelectedBenchmarkId}
              onRun={handleRunSelectedMatrix}
              onRefresh={refreshAll}
            />

            <MatrixToolbar
              suites={visibleSuites.length > 0 ? visibleSuites : matrixSuites}
              availableTiers={availableTiers}
              activeSuite={effectiveActiveSuite}
              visibleRecipeCount={displayedMatrixRecipes.length}
              totalRecipeCount={sortedMatrixRecipes.length}
              viewPreset={viewPreset}
              viewPresets={VIEW_PRESETS}
              filterState={filters}
              filterOptions={matrixFilterOptions}
              sortState={sort}
              helpIndex={helpIndex}
              onActiveSuiteChange={handleSetActiveSuiteFocus}
              onFilterChange={handleUpdateFilters}
              onResetFilters={handleResetFilters}
              onResetSort={handleResetSort}
              onSortChange={setSort}
              onViewPresetChange={handleSetViewPreset}
            />

            <ResultsMatrix
              activeSuite={effectiveActiveSuite}
              recipes={displayedMatrixRecipes}
              currentRun={currentRun}
              selectedCell={detailSelection}
              previewCells={matrixPreviewCells}
              helpIndex={helpIndex}
              onSelectCell={(selection) => {
                setDetailSelection(selection);
                setHighlightedRecipeId(selection.recipeId);
              }}
            />

            <BenchmarkCharts
              suites={visibleSuites.length > 0 ? visibleSuites : matrixSuites}
              activeSuite={effectiveActiveSuite}
              recipes={sortedMatrixRecipes}
              currentRun={currentRun}
              previewCells={matrixPreviewCells}
              chartMode={chartMode}
              selectedMetricId={chartMetricId}
              focusedRecipeIds={reconciledFocusedRecipeIds}
              highlightedRecipeId={highlightedRecipeId || selectedRecipe?.id || ''}
              selectedRecipeId={selectedRecipe?.id || ''}
              helpIndex={helpIndex}
              onActiveSuiteChange={handleSetActiveSuiteFocus}
              onChartModeChange={setChartMode}
              onMetricChange={setChartMetricId}
              onRecipeHighlight={setHighlightedRecipeId}
              onRecipeHighlightClear={() => setHighlightedRecipeId('')}
              onRecipeSelect={handleSelectRecipeFromChart}
              onRecipeToggle={(recipeId) => {
                setFocusedRecipeIds((currentIds) => toggleFocusedRecipeId(currentIds, recipeId, sortedMatrixRecipes.map((recipe) => recipe.id)));
              }}
              onResetRecipeFocus={() => {
                setFocusedRecipeIds(sortedMatrixRecipes.map((recipe) => recipe.id));
                setHighlightedRecipeId('');
              }}
            />

            <SuiteEvaluationCharts
              suites={visibleSuites.length > 0 ? visibleSuites : matrixSuites}
              recipes={displayedMatrixRecipes}
              currentRun={currentRun}
              previewCells={matrixPreviewCells}
              datasetStateById={datasetStateById}
              activeSuiteId={effectiveActiveSuite?.id || null}
              helpIndex={helpIndex}
              onActiveSuiteChange={handleSetActiveSuiteFocus}
            />
          </div>

          <div className="min-w-0 space-y-6 xl:col-start-2 xl:row-start-2 min-[1800px]:col-start-auto min-[1800px]:row-start-auto min-[1800px]:sticky min-[1800px]:top-4 min-[1800px]:self-start">
            <DetailPanel
              suite={selectedSuite}
              recipe={selectedRecipe}
              result={selectedResult}
              benchmark={currentRun}
              datasetState={selectedSuite ? datasetStateById.get(selectedSuite.id) || null : null}
              assetById={mergedAssetById}
              helpIndex={helpIndex}
            />

            <OperationalReviewCard
              apiBase={apiBase}
              suite={selectedSuite}
              recipe={selectedRecipe}
              result={selectedResult}
              benchmark={currentRun}
              clipStatus={config.legacyClipStatus}
              helpIndex={helpIndex}
            />
          </div>
        </div>
      )}
    </section>
  );
}
