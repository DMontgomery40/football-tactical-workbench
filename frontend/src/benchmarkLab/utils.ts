import type {
  BenchmarkAsset,
  BenchmarkDatasetState,
  BenchmarkFilterState,
  BenchmarkMetricCell,
  BenchmarkRecipe,
  BenchmarkRunDetail,
  BenchmarkRunResult,
  BenchmarkSortState,
  BenchmarkSuite,
  BenchmarkViewPreset,
} from './types';

export const SUITE_HELP_IDS: Record<string, string> = {
  'det.ball_quick_v1': 'benchmark.suite.det_ball_quick',
  'det.roles_quick_v1': 'benchmark.suite.det_roles_quick',
  'loc.synloc_quick_v1': 'benchmark.suite.synloc_quick',
  'spot.team_bas_quick_v1': 'benchmark.suite.team_bas_quick',
  'calib.sn_calib_medium_v1': 'benchmark.suite.sn_calib_medium',
  'track.sn_tracking_medium_v1': 'benchmark.suite.sn_tracking_medium',
  'spot.pcbas_medium_v1': 'benchmark.suite.pcbas_medium',
  'gsr.medium_v1': 'benchmark.suite.gsr_medium',
  'gsr.long_v1': 'benchmark.suite.gsr_long',
  'ops.clip_review_v1': 'benchmark.suite.operational_review',
};

export const METRIC_HELP_IDS: Record<string, string> = {
  ap_50_95: 'benchmark.metric.map50_95',
  ap_ball_50_95: 'benchmark.metric.map50_95',
  map_locsim: 'benchmark.metric.map_locsim',
  team_map_at_1: 'benchmark.metric.team_map_at_1',
  map_at_1: 'benchmark.metric.team_map_at_1',
  completeness_x_jac_5: 'benchmark.metric.jac5',
  hota: 'benchmark.metric.hota',
  gs_hota: 'benchmark.metric.gs_hota',
  f1_at_15: 'benchmark.metric.f1_at_15',
  fps: 'benchmark.metric.throughput',
  images_per_second: 'benchmark.metric.throughput',
  avg_image_latency_ms: 'benchmark.metric.throughput',
  frames_per_second: 'benchmark.metric.throughput',
  clips_per_second: 'benchmark.metric.throughput',
  track_stability: 'benchmark.metric.track_stability',
  calibration: 'benchmark.metric.calibration_health',
  coverage: 'benchmark.metric.coverage',
};

export interface PreviewCell {
  status: string;
  reason: string;
}

export interface MatrixFilterOptions {
  architectures: string[];
  bundleModes: string[];
  capabilities: string[];
  providers: string[];
  statuses: string[];
  tiers: string[];
}

export interface BenchmarkChartRow {
  recipeId: string;
  recipeLabel: string;
  status: string;
  durationSeconds: number | null;
  [metricId: string]: number | string | null;
}

export interface BenchmarkSeriesDefinition {
  id: string;
  label: string;
  helpId: string | null;
}

export interface SuiteEvaluationRow {
  suiteId: string;
  suiteLabel: string;
  readinessPct: number;
  comparableRecipes: number;
  comparableRatePct: number;
  avgPrimaryMetric: number | null;
  primaryMetricSpread: number | null;
  metricDispersion: number | null;
  avgRuntimeSeconds: number | null;
  blockedRatePct: number;
  unavailableRatePct: number;
  failedRatePct: number;
  sampleSize: number | null;
  materializationPct: number;
  avgRankCorrelation: number | null;
}

export function asSuites(value: BenchmarkSuite[] | Record<string, unknown>[] | undefined | null): BenchmarkSuite[] {
  return Array.isArray(value) ? (value as BenchmarkSuite[]) : [];
}

export function asDatasetStates(value: BenchmarkDatasetState[] | Record<string, unknown>[] | undefined | null): BenchmarkDatasetState[] {
  return Array.isArray(value) ? (value as BenchmarkDatasetState[]) : [];
}

export function asAssets(value: BenchmarkAsset[] | Record<string, unknown>[] | undefined | null): BenchmarkAsset[] {
  return Array.isArray(value) ? (value as BenchmarkAsset[]) : [];
}

export function asRecipes(value: BenchmarkRecipe[] | Record<string, unknown>[] | undefined | null): BenchmarkRecipe[] {
  return Array.isArray(value) ? (value as BenchmarkRecipe[]) : [];
}

export function getSuiteResult(detail: BenchmarkRunDetail | null, suiteId: string, recipeId: string): BenchmarkRunResult | null {
  const suiteResults = (detail?.suiteResults || {}) as Record<string, Record<string, BenchmarkRunResult | undefined>>;
  return suiteResults[suiteId]?.[recipeId] || null;
}

export function getMetricCell(result: BenchmarkRunResult | null, metricId: string): BenchmarkMetricCell | null {
  const raw = result?.metrics?.[metricId] as BenchmarkMetricCell | undefined;
  return raw || null;
}

export function metricDisplay(cell: BenchmarkMetricCell | null): string {
  if (!cell) return '—';
  if (cell.displayValue) return String(cell.displayValue);
  if (cell.isNa) return 'N/A';
  if (typeof cell.value === 'number') return String(cell.value);
  return '—';
}

export function metricSortValue(cell: BenchmarkMetricCell | null): number {
  if (!cell) return Number.NEGATIVE_INFINITY;
  if (typeof cell.sortValue === 'number') return cell.sortValue;
  if (typeof cell.value === 'number') return cell.value;
  return Number.NEGATIVE_INFINITY;
}

export function recipeSupportsSuite(recipe: BenchmarkRecipe, suiteId: string): boolean {
  return Array.isArray(recipe.compatibleSuiteIds) && recipe.compatibleSuiteIds.includes(suiteId);
}

export function recipeHasAnyNaForSuite(recipe: BenchmarkRecipe, suite: BenchmarkSuite, detail: BenchmarkRunDetail | null): boolean {
  const result = getSuiteResult(detail, suite.id, recipe.id);
  return (suite.metricColumns || []).some((metricId) => getMetricCell(result, metricId)?.isNa);
}

export function resolveMatrixCellState(
  detail: BenchmarkRunDetail | null,
  suite: BenchmarkSuite,
  recipe: BenchmarkRecipe,
  previewCells: Record<string, PreviewCell>,
): { result: BenchmarkRunResult | null; status: string; note: string } {
  const result = getSuiteResult(detail, suite.id, recipe.id);
  if (result) {
    return {
      result,
      status: result.status,
      note: metricDisplay(getMetricCell(result, result.primaryMetric || suite.primaryMetric)),
    };
  }

  const preview = previewCells[`${suite.id}::${recipe.id}`];
  if (detail && ['queued', 'running'].includes(detail.status)) {
    return {
      result: null,
      status: preview?.status === 'ready' ? 'queued' : preview?.status || 'queued',
      note: preview?.status === 'ready' ? 'Pending execution' : preview?.reason || 'Waiting for execution',
    };
  }

  return {
    result: null,
    status: preview?.status || 'pending',
    note: preview?.reason || 'Not run yet',
  };
}

export function suiteFamilyForPreset(preset: BenchmarkViewPreset): string | null {
  switch (preset) {
    case 'Detection':
      return 'detection';
    case 'Spotting':
      return 'spotting';
    case 'Localization':
      return 'localization';
    case 'Calibration':
      return 'calibration';
    case 'Tracking':
      return 'tracking';
    case 'Game State':
      return 'game_state';
    case 'Operational':
      return 'operational';
    default:
      return null;
  }
}

export function buildRecipeAssetMap(
  assets: BenchmarkAsset[],
): Map<string, BenchmarkAsset> {
  return new Map(assets.map((asset) => [asset.assetId, asset]));
}

function sourceAssetsForRecipe(
  recipe: BenchmarkRecipe,
  assetById: Map<string, BenchmarkAsset>,
): BenchmarkAsset[] {
  return recipe.sourceAssetIds
    .map((assetId) => assetById.get(assetId))
    .filter(Boolean) as BenchmarkAsset[];
}

function recipeSearchBlob(
  recipe: BenchmarkRecipe,
  assetById: Map<string, BenchmarkAsset>,
): string {
  const sourceAssets = sourceAssetsForRecipe(recipe, assetById);
  return [
    recipe.id,
    recipe.label,
    recipe.kind,
    recipe.pipeline,
    recipe.bundleMode,
    recipe.runtimeBinding,
    ...sourceAssets.flatMap((asset) => [
      asset.assetId,
      asset.label,
      asset.kind,
      asset.provider,
      asset.architecture,
    ]),
  ]
    .join(' ')
    .toLowerCase();
}

function recipeProvider(
  recipe: BenchmarkRecipe,
  assetById: Map<string, BenchmarkAsset>,
): string {
  return sourceAssetsForRecipe(recipe, assetById)[0]?.provider || '';
}

function recipeArchitecture(
  recipe: BenchmarkRecipe,
  assetById: Map<string, BenchmarkAsset>,
): string {
  return sourceAssetsForRecipe(recipe, assetById)[0]?.architecture || '';
}

function recipeStatusRank(status: string): number {
  switch (String(status || '').toLowerCase()) {
    case 'completed':
    case 'ready':
      return 5;
    case 'running':
      return 4;
    case 'queued':
      return 3;
    case 'pending':
      return 2;
    case 'blocked':
    case 'dataset_missing':
      return 1;
    case 'unavailable':
    case 'not_supported':
      return 0;
    case 'failed':
    case 'error':
      return -1;
    default:
      return -2;
  }
}

export function isLatencyMetric(metricId: string): boolean {
  const normalized = String(metricId || '').toLowerCase();
  return normalized.includes('latency') || normalized.endsWith('_ms');
}

export function collectMatrixFilterOptions(
  recipes: BenchmarkRecipe[],
  activeSuite: BenchmarkSuite | null,
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  assetById: Map<string, BenchmarkAsset>,
): MatrixFilterOptions {
  const providers = new Set<string>();
  const architectures = new Set<string>();
  const bundleModes = new Set<string>();
  const capabilities = new Set<string>();
  const statuses = new Set<string>();

  for (const recipe of recipes) {
    const sourceAssets = sourceAssetsForRecipe(recipe, assetById);
    for (const asset of sourceAssets) {
      if (asset.provider) {
        providers.add(asset.provider);
      }
      if (asset.architecture) {
        architectures.add(asset.architecture);
      }
    }
    if (recipe.bundleMode) {
      bundleModes.add(recipe.bundleMode);
    }
    for (const [capability, enabled] of Object.entries(recipe.capabilities || {})) {
      if (enabled) {
        capabilities.add(capability);
      }
    }
    if (activeSuite) {
      statuses.add(resolveMatrixCellState(detail, activeSuite, recipe, previewCells).status);
    }
  }

  const sortStrings = (values: Set<string>): string[] => Array.from(values).sort((left, right) => left.localeCompare(right));
  return {
    providers: sortStrings(providers),
    architectures: sortStrings(architectures),
    bundleModes: sortStrings(bundleModes),
    capabilities: sortStrings(capabilities),
    statuses: sortStrings(statuses),
    tiers: [],
  };
}

export function filterRecipesForMatrix(
  recipes: BenchmarkRecipe[],
  activeSuite: BenchmarkSuite | null,
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  filters: BenchmarkFilterState,
  assetById: Map<string, BenchmarkAsset>,
): BenchmarkRecipe[] {
  return recipes.filter((recipe) => {
    const cellState = activeSuite
      ? resolveMatrixCellState(detail, activeSuite, recipe, previewCells)
      : { status: 'pending', reason: '', result: null, note: '' };
    const supportsActiveSuite = activeSuite ? recipeSupportsSuite(recipe, activeSuite.id) : true;
    const hasNa = activeSuite ? recipeHasAnyNaForSuite(recipe, activeSuite, detail) : false;
    const search = filters.search.trim().toLowerCase();

    if (search && !recipeSearchBlob(recipe, assetById).includes(search)) {
      return false;
    }
    if (filters.provider !== 'all' && recipeProvider(recipe, assetById) !== filters.provider) {
      return false;
    }
    if (filters.architecture !== 'all' && recipeArchitecture(recipe, assetById) !== filters.architecture) {
      return false;
    }
    if (filters.bundleMode !== 'all' && recipe.bundleMode !== filters.bundleMode) {
      return false;
    }
    if (filters.status !== 'all' && cellState.status !== filters.status) {
      return false;
    }
    if (filters.capability !== 'all' && !Boolean(recipe.capabilities?.[filters.capability])) {
      return false;
    }
    if (filters.supportsActiveSuite === 'supported' && !supportsActiveSuite) {
      return false;
    }
    if (filters.supportsActiveSuite === 'unsupported' && supportsActiveSuite) {
      return false;
    }
    if (filters.hasNa === 'has_na' && !hasNa) {
      return false;
    }
    if (filters.hasNa === 'no_na' && hasNa) {
      return false;
    }
    return true;
  });
}

export function resolveSortColumn(
  activeSuite: BenchmarkSuite | null,
  sort: BenchmarkSortState,
): string {
  if (sort.column) {
    return sort.column;
  }
  return activeSuite?.primaryMetric || 'label';
}

export function sortRecipesForMatrix(
  recipes: BenchmarkRecipe[],
  activeSuite: BenchmarkSuite | null,
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
  sort: BenchmarkSortState,
  assetById: Map<string, BenchmarkAsset>,
): BenchmarkRecipe[] {
  const column = resolveSortColumn(activeSuite, sort);
  const direction = sort.direction === 'asc' ? 1 : -1;

  return [...recipes].sort((left, right) => {
    const leftState = activeSuite ? resolveMatrixCellState(detail, activeSuite, left, previewCells) : null;
    const rightState = activeSuite ? resolveMatrixCellState(detail, activeSuite, right, previewCells) : null;
    let forceMissingLast = false;

    const compareString = (leftValue: string, rightValue: string): number => leftValue.localeCompare(rightValue);
    const compareNumber = (leftValue: number | null, rightValue: number | null): number => {
      const leftMissing = leftValue === null || !Number.isFinite(leftValue);
      const rightMissing = rightValue === null || !Number.isFinite(rightValue);
      if (leftMissing && rightMissing) {
        return 0;
      }
      if (leftMissing) {
        return 1;
      }
      if (rightMissing) {
        return -1;
      }
      return leftValue - rightValue;
    };

    let comparison = 0;
    switch (column) {
      case 'label':
        comparison = compareString(left.label, right.label);
        break;
      case 'status':
        comparison = recipeStatusRank(leftState?.status || '') - recipeStatusRank(rightState?.status || '');
        break;
      case 'provider':
        comparison = compareString(recipeProvider(left, assetById), recipeProvider(right, assetById));
        break;
      case 'architecture':
        comparison = compareString(recipeArchitecture(left, assetById), recipeArchitecture(right, assetById));
        break;
      case 'bundleMode':
        comparison = compareString(left.bundleMode, right.bundleMode);
        break;
      default: {
        const leftCell = getMetricCell(leftState?.result || null, column);
        const rightCell = getMetricCell(rightState?.result || null, column);
        const leftMetric = leftCell
          ? (typeof leftCell.sortValue === 'number' ? leftCell.sortValue : leftCell.value)
          : null;
        const rightMetric = rightCell
          ? (typeof rightCell.sortValue === 'number' ? rightCell.sortValue : rightCell.value)
          : null;
        forceMissingLast = !Number.isFinite(leftMetric) || !Number.isFinite(rightMetric);
        comparison = compareNumber(leftMetric, rightMetric);
        break;
      }
    }

    if (comparison === 0) {
      comparison = left.label.localeCompare(right.label);
    }

    if (forceMissingLast) {
      return comparison;
    }
    if (column !== 'label' && column !== 'status' && isLatencyMetric(column)) {
      return comparison;
    }

    return comparison * direction;
  });
}

export function metricLabel(metricId: string): string {
  const labelMap: Record<string, string> = {
    ap_50_95: 'AP@[.50:.95]',
    ap_ball_50_95: 'AP Ball@[.50:.95]',
    ap_50: 'AP50',
    ap_75: 'AP75',
    precision: 'Precision',
    recall: 'Recall',
    images_per_second: 'Images/s',
    avg_image_latency_ms: 'Latency (ms)',
    map_locsim: 'mAP-LocSim',
    team_map_at_1: 'Team-mAP@1',
    map_at_1: 'mAP@1',
    completeness_x_jac_5: 'Completeness x JaC@5',
    completeness: 'Completeness',
    jac_5: 'JaC@5',
    hota: 'HOTA',
    deta: 'DetA',
    assa: 'AssA',
    gs_hota: 'GS-HOTA',
    f1_at_15: 'F1@15%',
    precision_at_15: 'Precision@15%',
    recall_at_15: 'Recall@15%',
    fps: 'FPS',
    track_stability: 'Track Stability',
    calibration: 'Calibration',
    coverage: 'Coverage',
    frames_per_second: 'Frames/s',
    clips_per_second: 'Clips/s',
  };
  return labelMap[metricId] || metricId;
}

export function statusLabel(status: string | null | undefined): string {
  const normalized = String(status || 'idle');
  if (normalized === 'not_supported') return 'N/A';
  if (normalized === 'blocked') return 'Blocked';
  if (normalized === 'dataset_missing') return 'Dataset missing';
  if (normalized === 'unavailable') return 'Unavailable';
  return normalized;
}

export function datasetStateBySuite(states: BenchmarkDatasetState[]): Record<string, BenchmarkDatasetState> {
  return Object.fromEntries(states.map((state) => [state.suiteId, state]));
}

function metricNumber(metric: BenchmarkMetricCell | null): number | null {
  if (!metric) {
    return null;
  }
  if (typeof metric.sortValue === 'number') {
    return metric.sortValue;
  }
  if (typeof metric.value === 'number') {
    return metric.value;
  }
  return null;
}

function durationSeconds(result: BenchmarkRunResult | null): number | null {
  const duration = result?.runtimeContext?.durationSeconds;
  return typeof duration === 'number' && Number.isFinite(duration) ? duration : null;
}

export function buildBenchmarkSeries(activeSuite: BenchmarkSuite | null): BenchmarkSeriesDefinition[] {
  if (!activeSuite) {
    return [];
  }
  return activeSuite.metricColumns.map((metricId) => ({
    id: metricId,
    label: metricLabel(metricId),
    helpId: METRIC_HELP_IDS[metricId] || null,
  }));
}

export function runtimeSeriesDefinition(): BenchmarkSeriesDefinition {
  return {
    id: 'durationSeconds',
    label: 'Cell runtime (s)',
    helpId: 'benchmark.metric.runtime_burden',
  };
}

export function buildBenchmarkChartRows(
  activeSuite: BenchmarkSuite | null,
  recipes: BenchmarkRecipe[],
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
): BenchmarkChartRow[] {
  if (!activeSuite) {
    return [];
  }

  return recipes.map((recipe) => {
    const state = resolveMatrixCellState(detail, activeSuite, recipe, previewCells);
    const row: BenchmarkChartRow = {
      recipeId: recipe.id,
      recipeLabel: recipe.label,
      status: state.status,
      durationSeconds: durationSeconds(state.result),
    };

    for (const metricId of activeSuite.metricColumns) {
      row[metricId] = metricNumber(getMetricCell(state.result, metricId));
    }
    return row;
  });
}

function statusCount(rows: Array<{ status: string }>, statuses: string[]): number {
  const wanted = new Set(statuses);
  return rows.filter((row) => wanted.has(row.status)).length;
}

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[]): number | null {
  if (values.length < 2) {
    return values.length === 1 ? 0 : null;
  }
  const mean = average(values) ?? 0;
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
  return Math.sqrt(variance);
}

function rankMap(values: Array<{ recipeId: string; value: number }>, descending: boolean): Map<string, number> {
  const ordered = [...values].sort((left, right) => (
    descending ? right.value - left.value : left.value - right.value
  ));
  const ranks = new Map<string, number>();
  ordered.forEach((entry, index) => {
    ranks.set(entry.recipeId, index + 1);
  });
  return ranks;
}

function spearmanCorrelation(
  left: Array<{ recipeId: string; value: number }>,
  right: Array<{ recipeId: string; value: number }>,
  descending: boolean,
): number | null {
  const sharedIds = left
    .map((entry) => entry.recipeId)
    .filter((recipeId) => right.some((candidate) => candidate.recipeId === recipeId));
  if (sharedIds.length < 2) {
    return null;
  }
  const leftRanks = rankMap(left.filter((entry) => sharedIds.includes(entry.recipeId)), descending);
  const rightRanks = rankMap(right.filter((entry) => sharedIds.includes(entry.recipeId)), descending);
  const count = sharedIds.length;
  const deltaSquared = sharedIds.reduce((sum, recipeId) => {
    const leftRank = leftRanks.get(recipeId) ?? 0;
    const rightRank = rightRanks.get(recipeId) ?? 0;
    return sum + ((leftRank - rightRank) ** 2);
  }, 0);
  return 1 - ((6 * deltaSquared) / (count * ((count ** 2) - 1)));
}

function suitePrimaryValues(
  suite: BenchmarkSuite,
  recipes: BenchmarkRecipe[],
  detail: BenchmarkRunDetail | null,
  previewCells: Record<string, PreviewCell>,
): Array<{ recipeId: string; value: number }> {
  const values: Array<{ recipeId: string; value: number }> = [];
  for (const recipe of recipes) {
    const state = resolveMatrixCellState(detail, suite, recipe, previewCells);
    if (state.status !== 'completed') {
      continue;
    }
    const value = metricNumber(getMetricCell(state.result, suite.primaryMetric));
    if (typeof value === 'number') {
      values.push({ recipeId: recipe.id, value });
    }
  }
  return values;
}

function materializationPct(state: BenchmarkDatasetState | null | undefined): number {
  if (!state) {
    return 0;
  }
  if (state.ready) {
    return 100;
  }
  if (state.datasetExists && state.manifestExists) {
    return 75;
  }
  if (state.datasetExists || state.manifestExists) {
    return 50;
  }
  return 0;
}

export function buildSuiteEvaluationRows(
  suites: BenchmarkSuite[],
  recipes: BenchmarkRecipe[],
  detail: BenchmarkRunDetail | null,
  datasetStateMap: Map<string, BenchmarkDatasetState>,
  previewCells: Record<string, PreviewCell>,
): SuiteEvaluationRow[] {
  const rows: SuiteEvaluationRow[] = suites.map((suite) => {
    const recipeRows = recipes.map((recipe) => resolveMatrixCellState(detail, suite, recipe, previewCells));
    const completedValues = suitePrimaryValues(suite, recipes, detail, previewCells);
    const primaryValues = completedValues.map((entry) => entry.value);
    const runtimeValues = recipes
      .map((recipe) => durationSeconds(getSuiteResult(detail, suite.id, recipe.id)))
      .filter((value): value is number => typeof value === 'number');
    const totalRecipes = Math.max(recipes.length, 1);
    const datasetState = datasetStateMap.get(suite.id);

    return {
      suiteId: suite.id,
      suiteLabel: suite.label,
      readinessPct: datasetState?.ready ? 100 : 0,
      comparableRecipes: completedValues.length,
      comparableRatePct: (completedValues.length / totalRecipes) * 100,
      avgPrimaryMetric: average(primaryValues),
      primaryMetricSpread: primaryValues.length > 1 ? Math.max(...primaryValues) - Math.min(...primaryValues) : (primaryValues.length === 1 ? 0 : null),
      metricDispersion: standardDeviation(primaryValues),
      avgRuntimeSeconds: average(runtimeValues),
      blockedRatePct: (statusCount(recipeRows, ['blocked', 'dataset_missing']) / totalRecipes) * 100,
      unavailableRatePct: (statusCount(recipeRows, ['unavailable', 'not_supported']) / totalRecipes) * 100,
      failedRatePct: (statusCount(recipeRows, ['failed', 'error']) / totalRecipes) * 100,
      sampleSize: datasetState?.manifestSummary?.itemCount ?? null,
      materializationPct: materializationPct(datasetState),
      avgRankCorrelation: null,
    };
  });

  for (const row of rows) {
    const sourceSuite = suites.find((suite) => suite.id === row.suiteId);
    if (!sourceSuite) {
      continue;
    }
    const descending = !isLatencyMetric(sourceSuite.primaryMetric);
    const sourceValues = suitePrimaryValues(sourceSuite, recipes, detail, previewCells);
    const correlations = rows
      .filter((candidate) => candidate.suiteId !== row.suiteId)
      .map((candidate) => {
        const candidateSuite = suites.find((suite) => suite.id === candidate.suiteId);
        if (!candidateSuite) {
          return null;
        }
        return spearmanCorrelation(
          sourceValues,
          suitePrimaryValues(candidateSuite, recipes, detail, previewCells),
          descending,
        );
      })
      .filter((value): value is number => typeof value === 'number');
    row.avgRankCorrelation = average(correlations);
  }

  return rows;
}

export function suiteEvaluationSeries(): Record<string, BenchmarkSeriesDefinition[]> {
  return {
    performance: [
      { id: 'primaryMetricSpread', label: 'Primary spread', helpId: 'benchmark.metric.discriminative_power' },
      { id: 'avgPrimaryMetric', label: 'Average primary', helpId: 'benchmark.metric.suite_average' },
      { id: 'metricDispersion', label: 'Dispersion', helpId: 'benchmark.metric.metric_dispersion' },
    ],
    operations: [
      { id: 'avgRuntimeSeconds', label: 'Avg runtime (s)', helpId: 'benchmark.metric.runtime_burden' },
      { id: 'blockedRatePct', label: 'Blocked %', helpId: 'benchmark.metric.blocked_rate' },
      { id: 'unavailableRatePct', label: 'Unavailable %', helpId: 'benchmark.metric.unavailable_rate' },
      { id: 'failedRatePct', label: 'Failed %', helpId: 'benchmark.metric.failed_rate' },
      { id: 'materializationPct', label: 'Materialized %', helpId: 'benchmark.metric.materialization_coverage' },
      { id: 'sampleSize', label: 'Sample size', helpId: 'benchmark.metric.sample_size' },
    ],
    agreement: [
      { id: 'comparableRecipes', label: 'Comparable recipes', helpId: 'benchmark.metric.comparable_recipes' },
      { id: 'comparableRatePct', label: 'Comparable %', helpId: 'benchmark.metric.comparable_recipes' },
      { id: 'avgRankCorrelation', label: 'Avg rank corr.', helpId: 'benchmark.metric.rank_correlation' },
      { id: 'readinessPct', label: 'Ready %', helpId: 'benchmark.metric.materialization_coverage' },
    ],
  };
}
