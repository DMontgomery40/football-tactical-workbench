import type { components } from '@contracts/generated/schema';

type BenchmarkConfigApi = components['schemas']['BenchmarkConfigResponse'];
type BenchmarkHistoryItemApi = components['schemas']['BenchmarkHistoryItemResponse'];
type BenchmarkRunDetailApi = components['schemas']['BenchmarkRunDetailResponse'];

type UnknownRecord = Record<string, unknown>;

export type DvcPayload = UnknownRecord | null;

export interface BenchmarkManifestSummary {
  kind: string | null;
  split: string | null;
  selection: string | null;
  itemCount: number | null;
  classCount: number | null;
  taskCoverage: string[];
}

export interface BenchmarkRuntimeContext {
  startedAt: string | null;
  finishedAt: string | null;
  durationSeconds: number | null;
}

export interface BenchmarkCapabilityMap {
  detection: boolean;
  tracking: boolean;
  reid: boolean;
  calibration: boolean;
  team_id: boolean;
  role_id: boolean;
  jersey_ocr: boolean;
  event_spotting: boolean;
  [key: string]: boolean | unknown;
}

export interface BenchmarkSuite {
  id: string;
  label: string;
  tier: string;
  family: string;
  protocol: string;
  primaryMetric: string;
  metricColumns: string[];
  requiredCapabilities: string[];
  datasetRoot: string;
  manifestPath: string;
  sourceUrl: string;
  license: string;
  notes: string;
  datasetSplit: string | null;
  dvcRequired: boolean;
  requiresClip: boolean;
  fallbackDatasetRoots: string[];
}

export interface BenchmarkDatasetState {
  suiteId: string;
  datasetRoot: string | null;
  datasetExists: boolean;
  datasetDvc: DvcPayload;
  manifestPath: string | null;
  manifestExists: boolean;
  manifestDvc: DvcPayload;
  conversionRoot: string;
  ready: boolean;
  readinessStatus: string;
  requiresClip: boolean;
  dvcRequired: boolean;
  dvcRuntime: DvcPayload;
  note: string | null;
  blockers: string[];
  manifestSummary: BenchmarkManifestSummary;
}

export interface BenchmarkAsset {
  assetId: string;
  kind: string;
  provider: string;
  source: string;
  label: string;
  version: string;
  architecture: string;
  artifactPath: string;
  bundleMode: string;
  runtimeBinding: string;
  available: boolean;
  capabilities: BenchmarkCapabilityMap;
  classMapping: UnknownRecord;
  artifactDvc: DvcPayload;
  availabilityError: string | null;
  trainingRunId: string | null;
  metrics: UnknownRecord | null;
  importOrigin: string | null;
  importedAt: string | null;
}

export interface BenchmarkRecipe {
  id: string;
  label: string;
  kind: string;
  assetId: string;
  sourceAssetIds: string[];
  pipeline: string;
  detectorAssetId: string | null;
  trackerAssetId: string | null;
  requestedTrackerMode: string | null;
  keypointModel: string | null;
  bundleMode: string;
  runtimeBinding: string;
  available: boolean;
  artifactPath: string;
  capabilities: BenchmarkCapabilityMap;
  classMapping: UnknownRecord;
  compatibleSuiteIds: string[];
}

export interface BenchmarkMetricValue {
  label: string;
  value: number | null;
  displayValue: string;
  unit: string;
  sortValue: number | null;
  isNa: boolean;
  raw: unknown;
}

export interface BenchmarkRunResult {
  suiteId: string;
  recipeId: string;
  status: string;
  error: string | null;
  metrics: Record<string, BenchmarkMetricValue>;
  flattenedMetrics: UnknownRecord;
  primaryMetric: string | null;
  artifacts: UnknownRecord;
  blockers: string[];
  runtimeContext: BenchmarkRuntimeContext;
  rawResult: UnknownRecord;
  legacyRecord: boolean;
}

export interface BenchmarkRunDetail {
  benchmarkId: string;
  schemaVersion: number;
  legacyRecord: boolean;
  label: string;
  status: string;
  createdAt: string;
  primarySuiteId: string | null;
  suiteIds: string[];
  recipeIds: string[];
  assets: BenchmarkAsset[];
  recipes: BenchmarkRecipe[];
  suiteResults: Record<string, Record<string, BenchmarkRunResult>>;
  progress: number;
  logs: string[];
  error: string | null;
  dvcRuntime: DvcPayload;
  legacyClipStatus: ClipStatus | null;
}

export interface BenchmarkHistoryItem {
  benchmarkId: string;
  label: string;
  status: string;
  createdAt: string;
  primarySuiteId: string | null;
  suiteIds: string[];
  recipeCount: number;
  legacyRecord: boolean;
}

export interface BenchmarkConfigSnapshot {
  schemaVersion: number;
  suites: BenchmarkSuite[];
  datasetStates: BenchmarkDatasetState[];
  assets: BenchmarkAsset[];
  recipes: BenchmarkRecipe[];
  dvcRuntime: DvcPayload;
  legacyClipStatus: ClipStatus | null;
  benchmarksDir: string;
}

export type BenchmarkConfig = BenchmarkConfigSnapshot;
export type BenchmarkMetricCell = BenchmarkMetricValue;

export type BenchmarkViewPreset =
  | 'Detection'
  | 'Spotting'
  | 'Localization'
  | 'Calibration'
  | 'Tracking'
  | 'Game State'
  | 'Operational';

export interface BenchmarkFilterState {
  search: string;
  suite: string;
  tier: string;
  provider: string;
  architecture: string;
  bundleMode: string;
  status: string;
  capability: string;
  supportsActiveSuite: 'all' | 'supported' | 'unsupported';
  hasNa: 'all' | 'has_na' | 'no_na';
}

export interface BenchmarkSortState {
  column: string;
  direction: 'asc' | 'desc';
}

export interface ClipStatus {
  ready: boolean;
  path: string | null;
  sizeMb: number | null;
  cacheDir: string | null;
  expectedFilename: string | null;
  dvc: DvcPayload;
  suiteId: string | null;
  note: string | null;
  error: string | null;
}

export interface ImportResponse {
  asset: BenchmarkAsset | null;
  recipes: BenchmarkRecipe[];
}

function isRecord(value: unknown): value is UnknownRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function asRecord(value: unknown): UnknownRecord {
  return isRecord(value) ? value : {};
}

function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') {
    return value;
  }
  if (value === null || value === undefined) {
    return fallback;
  }
  return String(value);
}

function asNullableString(value: unknown): string | null {
  const normalized = asString(value, '').trim();
  return normalized ? normalized : null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function asBoolean(value: unknown): boolean {
  return Boolean(value);
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => asString(item, '').trim())
    .filter(Boolean);
}

function normalizeDvcPayload(value: unknown): DvcPayload {
  return isRecord(value) ? value : null;
}

function normalizeCapabilityMap(value: unknown): BenchmarkCapabilityMap {
  const raw = asRecord(value);
  return {
    detection: asBoolean(raw.detection),
    tracking: asBoolean(raw.tracking),
    reid: asBoolean(raw.reid),
    calibration: asBoolean(raw.calibration),
    team_id: asBoolean(raw.team_id),
    role_id: asBoolean(raw.role_id),
    jersey_ocr: asBoolean(raw.jersey_ocr),
    event_spotting: asBoolean(raw.event_spotting),
    ...raw,
  };
}

function normalizeMetricValue(metricId: string, value: unknown): BenchmarkMetricValue {
  if (isRecord(value)) {
    const numericValue = asNumber(value.value);
    const displayValue = asNullableString(value.display_value)
      ?? (numericValue === null ? 'N/A' : String(numericValue));
    return {
      label: asString(value.label, metricId),
      value: numericValue,
      displayValue,
      unit: asString(value.unit, ''),
      sortValue: asNumber(value.sort_value),
      isNa: asBoolean(value.is_na) || numericValue === null,
      raw: value,
    };
  }

  const numericValue = asNumber(value);
  return {
    label: metricId,
    value: numericValue,
    displayValue: numericValue === null ? asString(value, 'N/A') : String(numericValue),
    unit: '',
    sortValue: numericValue,
    isNa: numericValue === null,
    raw: value,
  };
}

function normalizeMetrics(value: unknown): Record<string, BenchmarkMetricValue> {
  const raw = asRecord(value);
  return Object.fromEntries(
    Object.entries(raw).map(([metricId, metricValue]) => [metricId, normalizeMetricValue(metricId, metricValue)]),
  );
}

function normalizeSuite(value: unknown): BenchmarkSuite {
  const raw = asRecord(value);
  return {
    id: asString(raw.id),
    label: asString(raw.label),
    tier: asString(raw.tier),
    family: asString(raw.family),
    protocol: asString(raw.protocol),
    primaryMetric: asString(raw.primary_metric),
    metricColumns: asStringArray(raw.metric_columns),
    requiredCapabilities: asStringArray(raw.required_capabilities),
    datasetRoot: asString(raw.dataset_root),
    manifestPath: asString(raw.manifest_path),
    sourceUrl: asString(raw.source_url),
    license: asString(raw.license),
    notes: asString(raw.notes),
    datasetSplit: asNullableString(raw.dataset_split),
    dvcRequired: asBoolean(raw.dvc_required),
    requiresClip: asBoolean(raw.requires_clip),
    fallbackDatasetRoots: asStringArray(raw.fallback_dataset_roots),
  };
}

function normalizeDatasetState(value: unknown): BenchmarkDatasetState {
  const raw = asRecord(value);
  const manifestSummary = asRecord(raw.manifest_summary);
  return {
    suiteId: asString(raw.suite_id),
    datasetRoot: asNullableString(raw.dataset_root),
    datasetExists: asBoolean(raw.dataset_exists),
    datasetDvc: normalizeDvcPayload(raw.dataset_dvc),
    manifestPath: asNullableString(raw.manifest_path),
    manifestExists: asBoolean(raw.manifest_exists),
    manifestDvc: normalizeDvcPayload(raw.manifest_dvc),
    conversionRoot: asString(raw.conversion_root),
    ready: asBoolean(raw.ready),
    readinessStatus: asString(raw.readiness_status, asBoolean(raw.ready) ? 'ready' : 'blocked'),
    requiresClip: asBoolean(raw.requires_clip),
    dvcRequired: asBoolean(raw.dvc_required),
    dvcRuntime: normalizeDvcPayload(raw.dvc_runtime),
    note: asNullableString(raw.note),
    blockers: asStringArray(raw.blockers),
    manifestSummary: {
      kind: asNullableString(manifestSummary.kind),
      split: asNullableString(manifestSummary.split),
      selection: asNullableString(manifestSummary.selection),
      itemCount: asNumber(manifestSummary.item_count),
      classCount: asNumber(manifestSummary.class_count),
      taskCoverage: asStringArray(manifestSummary.task_coverage),
    },
  };
}

function normalizeAsset(value: unknown): BenchmarkAsset {
  const raw = asRecord(value);
  return {
    assetId: asString(raw.asset_id),
    kind: asString(raw.kind),
    provider: asString(raw.provider),
    source: asString(raw.source),
    label: asString(raw.label),
    version: asString(raw.version),
    architecture: asString(raw.architecture),
    artifactPath: asString(raw.artifact_path),
    bundleMode: asString(raw.bundle_mode),
    runtimeBinding: asString(raw.runtime_binding),
    available: asBoolean(raw.available),
    capabilities: normalizeCapabilityMap(raw.capabilities),
    classMapping: asRecord(raw.class_mapping),
    artifactDvc: normalizeDvcPayload(raw.artifact_dvc),
    availabilityError: asNullableString(raw.availability_error),
    trainingRunId: asNullableString(raw.training_run_id),
    metrics: isRecord(raw.metrics) ? raw.metrics : null,
    importOrigin: asNullableString(raw.import_origin),
    importedAt: asNullableString(raw.imported_at),
  };
}

function normalizeRecipe(value: unknown): BenchmarkRecipe {
  const raw = asRecord(value);
  return {
    id: asString(raw.id),
    label: asString(raw.label),
    kind: asString(raw.kind),
    assetId: asString(raw.asset_id),
    sourceAssetIds: asStringArray(raw.source_asset_ids),
    pipeline: asString(raw.pipeline),
    detectorAssetId: asNullableString(raw.detector_asset_id),
    trackerAssetId: asNullableString(raw.tracker_asset_id),
    requestedTrackerMode: asNullableString(raw.requested_tracker_mode),
    keypointModel: asNullableString(raw.keypoint_model),
    bundleMode: asString(raw.bundle_mode),
    runtimeBinding: asString(raw.runtime_binding),
    available: asBoolean(raw.available),
    artifactPath: asString(raw.artifact_path),
    capabilities: normalizeCapabilityMap(raw.capabilities),
    classMapping: asRecord(raw.class_mapping),
    compatibleSuiteIds: asStringArray(raw.compatible_suite_ids),
  };
}

function normalizeRunResult(value: unknown): BenchmarkRunResult {
  const raw = asRecord(value);
  const runtimeContext = asRecord(raw.runtime_context);
  return {
    suiteId: asString(raw.suite_id),
    recipeId: asString(raw.recipe_id),
    status: asString(raw.status),
    error: asNullableString(raw.error),
    metrics: normalizeMetrics(raw.metrics),
    flattenedMetrics: asRecord(raw.flattened_metrics),
    primaryMetric: asNullableString(raw.primary_metric),
    artifacts: asRecord(raw.artifacts),
    blockers: asStringArray(raw.blockers),
    runtimeContext: {
      startedAt: asNullableString(runtimeContext.started_at),
      finishedAt: asNullableString(runtimeContext.finished_at),
      durationSeconds: asNumber(runtimeContext.duration_seconds),
    },
    rawResult: asRecord(raw.raw_result),
    legacyRecord: asBoolean(raw.legacy_record),
  };
}

export function normalizeClipStatus(value: unknown): ClipStatus {
  const raw = asRecord(value);
  return {
    ready: asBoolean(raw.ready),
    path: asNullableString(raw.path),
    sizeMb: asNumber(raw.size_mb),
    cacheDir: asNullableString(raw.cache_dir),
    expectedFilename: asNullableString(raw.expected_filename),
    dvc: normalizeDvcPayload(raw.dvc),
    suiteId: asNullableString(raw.suite_id),
    note: asNullableString(raw.note),
    error: asNullableString(raw.error),
  };
}

export function normalizeBenchmarkConfig(value: BenchmarkConfigApi): BenchmarkConfigSnapshot {
  const raw = asRecord(value);
  return {
    schemaVersion: asNumber(raw.schema_version) ?? 2,
    suites: (Array.isArray(raw.suites) ? raw.suites : []).map(normalizeSuite).filter((suite) => Boolean(suite.id)),
    datasetStates: (Array.isArray(raw.dataset_states) ? raw.dataset_states : [])
      .map(normalizeDatasetState)
      .filter((state) => Boolean(state.suiteId)),
    assets: (Array.isArray(raw.assets) ? raw.assets : []).map(normalizeAsset).filter((asset) => Boolean(asset.assetId)),
    recipes: (Array.isArray(raw.recipes) ? raw.recipes : []).map(normalizeRecipe).filter((recipe) => Boolean(recipe.id)),
    dvcRuntime: normalizeDvcPayload(raw.dvc_runtime),
    legacyClipStatus: raw.legacy_clip_status ? normalizeClipStatus(raw.legacy_clip_status) : null,
    benchmarksDir: asString(raw.benchmarks_dir),
  };
}

export function normalizeBenchmarkHistory(value: BenchmarkHistoryItemApi[] | unknown): BenchmarkHistoryItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      const raw = asRecord(item);
      return {
        benchmarkId: asString(raw.benchmark_id),
        label: asString(raw.label),
        status: asString(raw.status),
        createdAt: asString(raw.created_at),
        primarySuiteId: asNullableString(raw.primary_suite_id),
        suiteIds: asStringArray(raw.suite_ids),
        recipeCount: asNumber(raw.recipe_count) ?? 0,
        legacyRecord: asBoolean(raw.legacy_record),
      };
    })
    .filter((item) => Boolean(item.benchmarkId));
}

export function normalizeBenchmarkRunDetail(value: BenchmarkRunDetailApi): BenchmarkRunDetail {
  const raw = asRecord(value);
  const suiteResultsRaw = asRecord(raw.suite_results);
  const suiteResults: Record<string, Record<string, BenchmarkRunResult>> = {};

  for (const [suiteId, resultMapValue] of Object.entries(suiteResultsRaw)) {
    const normalizedResultMap: Record<string, BenchmarkRunResult> = {};
    for (const [recipeId, resultValue] of Object.entries(asRecord(resultMapValue))) {
      normalizedResultMap[recipeId] = normalizeRunResult(resultValue);
    }
    suiteResults[suiteId] = normalizedResultMap;
  }

  return {
    benchmarkId: asString(raw.benchmark_id),
    schemaVersion: asNumber(raw.schema_version) ?? 2,
    legacyRecord: asBoolean(raw.legacy_record),
    label: asString(raw.label),
    status: asString(raw.status),
    createdAt: asString(raw.created_at),
    primarySuiteId: asNullableString(raw.primary_suite_id),
    suiteIds: asStringArray(raw.suite_ids),
    recipeIds: asStringArray(raw.recipe_ids),
    assets: (Array.isArray(raw.assets) ? raw.assets : []).map(normalizeAsset).filter((asset) => Boolean(asset.assetId)),
    recipes: (Array.isArray(raw.recipes) ? raw.recipes : []).map(normalizeRecipe).filter((recipe) => Boolean(recipe.id)),
    suiteResults,
    progress: asNumber(raw.progress) ?? 0,
    logs: asStringArray(raw.logs),
    error: asNullableString(raw.error),
    dvcRuntime: normalizeDvcPayload(raw.dvc_runtime),
    legacyClipStatus: raw.legacy_clip_status ? normalizeClipStatus(raw.legacy_clip_status) : null,
  };
}

export function normalizeImportResponse(value: unknown): ImportResponse {
  const raw = asRecord(value);
  return {
    asset: raw.asset ? normalizeAsset(raw.asset) : null,
    recipes: (Array.isArray(raw.recipes) ? raw.recipes : []).map(normalizeRecipe).filter((recipe) => Boolean(recipe.id)),
  };
}

export function buildCellKey(suiteId: string, recipeId: string): string {
  return `${suiteId}::${recipeId}`;
}

export function recipeSupportsSuite(recipe: BenchmarkRecipe, suiteId: string): boolean {
  return recipe.compatibleSuiteIds.includes(suiteId);
}

export function recipeSupportsAllSuites(recipe: BenchmarkRecipe, suiteIds: string[]): boolean {
  if (suiteIds.length === 0) {
    return true;
  }
  return suiteIds.every((suiteId) => recipe.compatibleSuiteIds.includes(suiteId));
}

export function formatCapabilityLabel(value: string): string {
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export function formatResultStatus(status: string): string {
  return formatCapabilityLabel(status || 'unknown');
}

export function formatMetricDisplay(metric: BenchmarkMetricValue | undefined): string {
  if (!metric) {
    return 'N/A';
  }
  return metric.displayValue || (metric.value === null ? 'N/A' : String(metric.value));
}

export function getMetricNumericValue(metric: BenchmarkMetricValue | undefined): number | null {
  if (!metric) {
    return null;
  }
  return metric.sortValue ?? metric.value;
}

export function getPrimaryMetricId(
  result: BenchmarkRunResult | null | undefined,
  suite: BenchmarkSuite | null | undefined,
): string {
  return result?.primaryMetric || suite?.primaryMetric || '';
}

export function getPrimaryMetricDisplay(
  result: BenchmarkRunResult | null | undefined,
  suite: BenchmarkSuite | null | undefined,
): string {
  const metricId = getPrimaryMetricId(result, suite);
  return formatMetricDisplay(metricId ? result?.metrics[metricId] : undefined);
}

export function getPrimaryMetricValue(
  result: BenchmarkRunResult | null | undefined,
  suite: BenchmarkSuite | null | undefined,
): number | null {
  const metricId = getPrimaryMetricId(result, suite);
  return getMetricNumericValue(metricId ? result?.metrics[metricId] : undefined);
}

export function pickPreferredRecipeId(
  recipes: BenchmarkRecipe[],
  suites: BenchmarkSuite[],
  activeDetector: string,
): string {
  const availableRecipes = recipes.filter((recipe) => recipe.available);
  const selectedSuiteIds = suites.map((suite) => suite.id);
  const compatiblePool = availableRecipes.filter((recipe) => recipeSupportsAllSuites(recipe, selectedSuiteIds));
  const pool = compatiblePool.length > 0 ? compatiblePool : (availableRecipes.length > 0 ? availableRecipes : recipes);
  if (pool.length === 0) {
    return '';
  }

  const needsTrackingLikeRecipe = suites.some((suite) => {
    const capabilitySet = new Set(suite.requiredCapabilities);
    return (
      suite.protocol === 'operational'
      || capabilitySet.has('tracking')
      || capabilitySet.has('calibration')
      || capabilitySet.has('team_id')
    );
  });
  const needsGameStateBaseline = suites.some((suite) => {
    const capabilitySet = new Set(suite.requiredCapabilities);
    return capabilitySet.has('jersey_ocr') || capabilitySet.has('role_id');
  });

  const preferredIds = [
    needsGameStateBaseline ? 'pipeline:sn-gamestate-tracklab' : '',
    needsTrackingLikeRecipe && activeDetector ? `tracker:${activeDetector}+hybrid_reid+soccana_keypoint` : '',
    activeDetector ? `detector:${activeDetector}` : '',
    'tracker:soccana+hybrid_reid+soccana_keypoint',
    'detector:soccana',
    'pipeline:soccermaster',
    'pipeline:sn-gamestate-tracklab',
  ].filter(Boolean);

  for (const preferredId of preferredIds) {
    if (pool.some((recipe) => recipe.id === preferredId)) {
      return preferredId;
    }
  }
  return pool[0]?.id || '';
}

export function basenameFromPath(value: string | null | undefined): string {
  if (!value) {
    return '';
  }
  const parts = String(value).split(/[\\/]+/).filter(Boolean);
  return parts.pop() || String(value);
}
