import type {
  BenchmarkDatasetState,
  BenchmarkRecipe,
  BenchmarkRunDetail,
  BenchmarkSuite,
} from './types';
import {
  getMetricCell,
  isLatencyMetric,
  metricDisplay,
  metricLabel,
  resolveMatrixCellState,
  type PreviewCell,
} from './utils.js';

export type BenchmarkChartMode = 'bar' | 'line';

export interface ChartMetricOption {
  id: string;
  label: string;
  kind: 'primary' | 'metric' | 'runtime';
}

export interface RecipeLegendEntry {
  recipeId: string;
  label: string;
  color: string;
  visible: boolean;
  selected: boolean;
  highlighted: boolean;
  status: string;
}

export interface RecipeComparisonDatum {
  recipeId: string;
  label: string;
  displayValue: string;
  value: number | null;
  status: string;
  note: string;
  color: string;
  visible: boolean;
  selected: boolean;
  highlighted: boolean;
}

export interface RecipeMetricProfileDatum {
  metricId: string;
  label: string;
  [recipeId: string]: string | number | null;
}

export interface SuiteEvaluationDatum {
  suiteId: string;
  label: string;
  family: string;
  primaryMetricId: string;
  primaryMetricLabel: string;
  runtimeMetricId: string;
  runtimeMetricLabel: string;
  comparableRecipes: number;
  avgPrimaryMetric: number | null;
  primaryMetricSpread: number | null;
  primaryMetricDispersion: number | null;
  avgRuntime: number | null;
  blockedRate: number;
  unavailableRate: number;
  failureRate: number;
  materializationCoverage: number;
  rankCorrelation: number | null;
  readyState: number;
}

const CHART_PALETTE = [
  '#0F4C81',
  '#C76B2B',
  '#2F6F4E',
  '#B33A3A',
  '#2F7F8F',
  '#7A5C2E',
  '#365C8D',
  '#6B4E71',
];

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function normalizeRate(value: number, total: number): number {
  if (total <= 0) {
    return 0;
  }
  return Number(((value / total) * 100).toFixed(1));
}

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const total = values.reduce((sum, value) => sum + value, 0);
  return Number((total / values.length).toFixed(3));
}

function standardDeviation(values: number[]): number | null {
  if (values.length < 2) {
    return null;
  }
  const avg = average(values);
  if (avg === null) {
    return null;
  }
  const variance = values.reduce((sum, value) => sum + ((value - avg) ** 2), 0) / values.length;
  return Number(Math.sqrt(variance).toFixed(3));
}

function range(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return Number((Math.max(...values) - Math.min(...values)).toFixed(3));
}

function sortRecipeIdsByMetric(
  recipeIds: string[],
  valuesByRecipeId: Map<string, number>,
): string[] {
  return [...recipeIds].sort((left, right) => {
    const leftValue = valuesByRecipeId.get(left);
    const rightValue = valuesByRecipeId.get(right);
    if (leftValue === undefined && rightValue === undefined) {
      return left.localeCompare(right);
    }
    if (leftValue === undefined) {
      return 1;
    }
    if (rightValue === undefined) {
      return -1;
    }
    if (rightValue !== leftValue) {
      return rightValue - leftValue;
    }
    return left.localeCompare(right);
  });
}

function buildRankLookup(
  recipeIds: string[],
  valuesByRecipeId: Map<string, number>,
): Map<string, number> {
  const comparableRecipeIds = recipeIds.filter((recipeId) => valuesByRecipeId.has(recipeId));
  const ordered = sortRecipeIdsByMetric(comparableRecipeIds, valuesByRecipeId);
  return new Map(ordered.map((recipeId, index) => [recipeId, index + 1]));
}

function spearmanCorrelation(
  leftRanks: Map<string, number>,
  rightRanks: Map<string, number>,
): number | null {
  const sharedIds = [...leftRanks.keys()].filter((recipeId) => rightRanks.has(recipeId));
  if (sharedIds.length < 2) {
    return null;
  }
  const leftValues = sharedIds.map((recipeId) => leftRanks.get(recipeId) ?? 0);
  const rightValues = sharedIds.map((recipeId) => rightRanks.get(recipeId) ?? 0);
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const leftMean = mean(leftValues);
  const rightMean = mean(rightValues);
  let numerator = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < sharedIds.length; index += 1) {
    const leftDelta = leftValues[index] - leftMean;
    const rightDelta = rightValues[index] - rightMean;
    numerator += leftDelta * rightDelta;
    leftVariance += leftDelta ** 2;
    rightVariance += rightDelta ** 2;
  }
  if (leftVariance <= 0 || rightVariance <= 0) {
    return 1;
  }
  return Number((numerator / Math.sqrt(leftVariance * rightVariance)).toFixed(3));
}

function datasetCoverageScore(state: BenchmarkDatasetState | null | undefined): number {
  if (!state) {
    return 0;
  }
  const checks = [
    state.datasetExists ? 1 : 0,
    state.manifestExists ? 1 : 0,
    state.ready ? 1 : 0,
  ];
  return Number(((checks.reduce((sum, value) => sum + value, 0) / checks.length) * 100).toFixed(1));
}

export function chartColor(index: number): string {
  return CHART_PALETTE[index % CHART_PALETTE.length];
}

export function reconcileFocusedRecipeIds(currentIds: string[], allIds: string[]): string[] {
  if (allIds.length === 0) {
    return [];
  }
  if (currentIds.length === 0) {
    return [...allIds];
  }
  const allowed = new Set(allIds);
  const next = currentIds.filter((recipeId) => allowed.has(recipeId));
  return next.length > 0 ? next : [...allIds];
}

export function toggleFocusedRecipeId(currentIds: string[], recipeId: string, allIds: string[]): string[] {
  const normalized = reconcileFocusedRecipeIds(currentIds, allIds);
  if (normalized.length === 0) {
    return [];
  }
  const visible = new Set(normalized);
  if (visible.has(recipeId)) {
    visible.delete(recipeId);
  } else {
    visible.add(recipeId);
  }
  const next = allIds.filter((candidateId) => visible.has(candidateId));
  return next.length > 0 ? next : [...allIds];
}

export function resolveRuntimeMetricId(suite: BenchmarkSuite | null): string {
  if (!suite) {
    return '';
  }
  const candidates = (suite.metricColumns || []).filter((metricId) => {
    const normalized = metricId.toLowerCase();
    return normalized.includes('per_second') || normalized === 'fps' || isLatencyMetric(metricId);
  });
  return candidates[0] || '';
}

export function metricOptionsForSuite(suite: BenchmarkSuite | null): ChartMetricOption[] {
  if (!suite) {
    return [];
  }
  const runtimeMetricId = resolveRuntimeMetricId(suite);
  return (suite.metricColumns.length > 0 ? suite.metricColumns : [suite.primaryMetric])
    .filter(Boolean)
    .map((metricId) => ({
      id: metricId,
      label: metricLabel(metricId),
      kind: metricId === suite.primaryMetric ? 'primary' : (metricId === runtimeMetricId ? 'runtime' : 'metric'),
    }));
}

export function buildRecipeLegendEntries(
  recipes: BenchmarkRecipe[],
  suite: BenchmarkSuite | null,
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  focusedRecipeIds: string[],
  highlightedRecipeId: string,
  selectedRecipeId: string,
): RecipeLegendEntry[] {
  const visibleSet = new Set(reconcileFocusedRecipeIds(focusedRecipeIds, recipes.map((recipe) => recipe.id)));
  return recipes.map((recipe, index) => {
    const cellState = suite
      ? resolveMatrixCellState(detail, suite, recipe, previewCells)
      : { status: 'pending', note: 'Not run yet', result: null };
    return {
      recipeId: recipe.id,
      label: recipe.label,
      color: chartColor(index),
      visible: visibleSet.has(recipe.id),
      selected: selectedRecipeId === recipe.id,
      highlighted: highlightedRecipeId === recipe.id,
      status: cellState.status,
    };
  });
}

export function buildRecipeComparisonData(
  recipes: BenchmarkRecipe[],
  suite: BenchmarkSuite | null,
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  metricId: string,
  focusedRecipeIds: string[],
  highlightedRecipeId: string,
  selectedRecipeId: string,
): RecipeComparisonDatum[] {
  const visibleSet = new Set(reconcileFocusedRecipeIds(focusedRecipeIds, recipes.map((recipe) => recipe.id)));
  return recipes
    .filter((recipe) => visibleSet.has(recipe.id))
    .map((recipe, index) => {
      const cellState = suite
        ? resolveMatrixCellState(detail, suite, recipe, previewCells)
        : { status: 'pending', note: 'Not run yet', result: null };
      const metric = getMetricCell(cellState.result, metricId);
      return {
        recipeId: recipe.id,
        label: recipe.label,
        displayValue: metricDisplay(metric),
        value: finiteNumber(metric?.sortValue ?? metric?.value ?? null),
        status: cellState.status,
        note: cellState.note,
        color: chartColor(index),
        visible: true,
        selected: recipe.id === selectedRecipeId,
        highlighted: recipe.id === highlightedRecipeId,
      };
    });
}

export function buildRecipeMetricProfileData(
  recipes: BenchmarkRecipe[],
  suite: BenchmarkSuite | null,
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  focusedRecipeIds: string[],
): RecipeMetricProfileDatum[] {
  if (!suite) {
    return [];
  }
  const visibleRecipes = recipes.filter((recipe) => reconcileFocusedRecipeIds(focusedRecipeIds, recipes.map((item) => item.id)).includes(recipe.id));
  const metricIds = (suite.metricColumns.length > 0 ? suite.metricColumns : [suite.primaryMetric]).filter(Boolean);

  return metricIds.map((metricId) => {
    const values = visibleRecipes.map((recipe) => {
      const cellState = resolveMatrixCellState(detail, suite, recipe, previewCells);
      const metric = getMetricCell(cellState.result, metricId);
      return finiteNumber(metric?.sortValue ?? metric?.value ?? null);
    });
    const finiteValues = values.filter((value): value is number => value !== null);
    const minValue = finiteValues.length > 0 ? Math.min(...finiteValues) : null;
    const maxValue = finiteValues.length > 0 ? Math.max(...finiteValues) : null;
    const datum: RecipeMetricProfileDatum = {
      metricId,
      label: metricLabel(metricId),
    };
    visibleRecipes.forEach((recipe, recipeIndex) => {
      const rawValue = values[recipeIndex];
      if (rawValue === null || minValue === null || maxValue === null) {
        datum[recipe.id] = null;
        return;
      }
      if (maxValue === minValue) {
        datum[recipe.id] = 100;
        return;
      }
      const normalized = isLatencyMetric(metricId)
        ? ((maxValue - rawValue) / (maxValue - minValue)) * 100
        : ((rawValue - minValue) / (maxValue - minValue)) * 100;
      datum[recipe.id] = Number(normalized.toFixed(1));
    });
    return datum;
  });
}

export function buildSuiteEvaluationData(
  suites: BenchmarkSuite[],
  recipes: BenchmarkRecipe[],
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  datasetStateById: Map<string, BenchmarkDatasetState>,
  referenceSuiteId: string | null,
): SuiteEvaluationDatum[] {
  const referenceSummary = new Map<string, number>();
  const recipeIds = recipes.map((recipe) => recipe.id);
  const results = suites.map((suite) => {
    const primaryMetricId = suite.primaryMetric;
    const runtimeMetricId = resolveRuntimeMetricId(suite);
    const states = recipes.map((recipe) => ({
      recipe,
      cellState: resolveMatrixCellState(detail, suite, recipe, previewCells),
    }));
    const comparable = states.filter(({ cellState }) => {
      const metric = getMetricCell(cellState.result, primaryMetricId);
      return finiteNumber(metric?.sortValue ?? metric?.value ?? null) !== null;
    });
    const primaryValues = comparable
      .map(({ cellState }) => finiteNumber(getMetricCell(cellState.result, primaryMetricId)?.sortValue
        ?? getMetricCell(cellState.result, primaryMetricId)?.value ?? null))
      .filter((value): value is number => value !== null);
    const runtimeValues = comparable
      .map(({ cellState }) => finiteNumber(getMetricCell(cellState.result, runtimeMetricId)?.sortValue
        ?? getMetricCell(cellState.result, runtimeMetricId)?.value ?? null))
      .filter((value): value is number => value !== null);

    const blockedCount = states.filter(({ cellState }) => ['blocked', 'dataset_missing', 'unavailable', 'not_supported', 'failed', 'error'].includes(cellState.status)).length;
    const unavailableCount = states.filter(({ cellState }) => ['unavailable', 'not_supported'].includes(cellState.status)).length;
    const failureCount = states.filter(({ cellState }) => ['failed', 'error'].includes(cellState.status)).length;
    const metricValuesByRecipeId = new Map<string, number>();
    comparable.forEach(({ recipe, cellState }) => {
      const metric = getMetricCell(cellState.result, primaryMetricId);
      const numeric = finiteNumber(metric?.sortValue ?? metric?.value ?? null);
      if (numeric !== null) {
        metricValuesByRecipeId.set(recipe.id, numeric);
      }
    });

    if (suite.id === referenceSuiteId) {
      referenceSummary.clear();
      metricValuesByRecipeId.forEach((value, recipeId) => {
        referenceSummary.set(recipeId, value);
      });
    }

    return {
      suiteId: suite.id,
      label: suite.label,
      family: suite.family,
      primaryMetricId,
      primaryMetricLabel: metricLabel(primaryMetricId),
      runtimeMetricId,
      runtimeMetricLabel: runtimeMetricId ? metricLabel(runtimeMetricId) : 'Runtime',
      comparableRecipes: comparable.length,
      avgPrimaryMetric: average(primaryValues),
      primaryMetricSpread: range(primaryValues),
      primaryMetricDispersion: standardDeviation(primaryValues),
      avgRuntime: average(runtimeValues),
      blockedRate: normalizeRate(blockedCount, states.length),
      unavailableRate: normalizeRate(unavailableCount, states.length),
      failureRate: normalizeRate(failureCount, states.length),
      materializationCoverage: datasetCoverageScore(datasetStateById.get(suite.id)),
      rankCorrelation: null,
      readyState: datasetStateById.get(suite.id)?.ready ? 100 : 0,
      _metricValuesByRecipeId: metricValuesByRecipeId,
      _recipeIds: recipeIds,
    } as SuiteEvaluationDatum & {
      _metricValuesByRecipeId: Map<string, number>;
      _recipeIds: string[];
    };
  });

  const referenceEntry = results.find((suite) => suite.suiteId === referenceSuiteId) as (SuiteEvaluationDatum & {
    _metricValuesByRecipeId?: Map<string, number>;
    _recipeIds?: string[];
  }) | undefined;
  const referenceRanks = referenceEntry?._metricValuesByRecipeId
    ? buildRankLookup(referenceEntry._recipeIds || [], referenceEntry._metricValuesByRecipeId)
    : null;

  return results.map((suiteEntry) => {
    const rawEntry = suiteEntry as SuiteEvaluationDatum & {
      _metricValuesByRecipeId?: Map<string, number>;
      _recipeIds?: string[];
    };
    const rankCorrelation = referenceRanks && rawEntry._metricValuesByRecipeId
      ? spearmanCorrelation(referenceRanks, buildRankLookup(rawEntry._recipeIds || [], rawEntry._metricValuesByRecipeId))
      : null;
    return {
      suiteId: suiteEntry.suiteId,
      label: suiteEntry.label,
      family: suiteEntry.family,
      primaryMetricId: suiteEntry.primaryMetricId,
      primaryMetricLabel: suiteEntry.primaryMetricLabel,
      runtimeMetricId: suiteEntry.runtimeMetricId,
      runtimeMetricLabel: suiteEntry.runtimeMetricLabel,
      comparableRecipes: suiteEntry.comparableRecipes,
      avgPrimaryMetric: suiteEntry.avgPrimaryMetric,
      primaryMetricSpread: suiteEntry.primaryMetricSpread,
      primaryMetricDispersion: suiteEntry.primaryMetricDispersion,
      avgRuntime: suiteEntry.avgRuntime,
      blockedRate: suiteEntry.blockedRate,
      unavailableRate: suiteEntry.unavailableRate,
      failureRate: suiteEntry.failureRate,
      materializationCoverage: suiteEntry.materializationCoverage,
      rankCorrelation: suiteEntry.suiteId === referenceSuiteId ? 1 : rankCorrelation,
      readyState: suiteEntry.readyState,
    };
  });
}
