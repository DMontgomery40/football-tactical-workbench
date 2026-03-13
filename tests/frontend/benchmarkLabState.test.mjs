import assert from 'node:assert/strict';
import test from 'node:test';

import {
  mergeUniqueStrings,
  resolveVisibleActiveSuiteId,
  resolveDetailSelection,
  resolveSelectedBenchmarkId,
  resolveSelectedRecipeIds,
  resolveSelectedSuiteIds,
} from '../../frontend/src/benchmarkLab/state.ts';
import {
  normalizeBenchmarkConfig,
  normalizeBenchmarkHistory,
  normalizeBenchmarkRunDetail,
} from '../../frontend/src/benchmarkLab/types.ts';
import {
  collectMatrixFilterOptions,
  filterRecipesForMatrix,
  sortRecipesForMatrix,
} from '../../frontend/src/benchmarkLab/utils.ts';

test('normalizeBenchmarkConfig converts snake_case payload into TS-friendly shape', () => {
  const normalized = normalizeBenchmarkConfig({
    schema_version: 2,
    suites: [
      {
        id: 'det.roles_quick_v1',
        label: 'Detection Roles Quick',
        tier: 'quick',
        family: 'detection',
        protocol: 'coco_detection',
        primary_metric: 'ap_50_95',
        metric_columns: ['ap_50_95'],
        required_capabilities: ['detection'],
        dataset_root: '/tmp/dataset',
        manifest_path: '/tmp/manifest.json',
        source_url: 'https://example.com',
        license: 'cc-by-4.0',
        notes: 'quick',
        dataset_split: 'test',
        dvc_required: true,
      },
    ],
    dataset_states: [],
    assets: [],
    recipes: [],
    dvc_runtime: null,
    legacy_clip_status: null,
    benchmarks_dir: '/tmp/benchmarks',
  });

  assert.equal(normalized.suites[0].primaryMetric, 'ap_50_95');
  assert.equal(normalized.suites[0].metricColumns[0], 'ap_50_95');
  assert.equal(normalized.benchmarksDir, '/tmp/benchmarks');
});

test('normalizeBenchmarkHistory converts benchmark ids and status fields', () => {
  const history = normalizeBenchmarkHistory([
    {
      benchmark_id: 'b1',
      label: 'run',
      status: 'completed',
      created_at: '2026-03-12T00:00:00Z',
      primary_suite_id: 'det.roles_quick_v1',
      suite_ids: ['det.roles_quick_v1'],
      recipe_count: 2,
      legacy_record: false,
    },
  ]);

  assert.equal(history[0].benchmarkId, 'b1');
  assert.equal(history[0].primarySuiteId, 'det.roles_quick_v1');
});

test('normalizeBenchmarkRunDetail maps suiteResults and metrics', () => {
  const detail = normalizeBenchmarkRunDetail({
    benchmark_id: 'b1',
    schema_version: 2,
    legacy_record: false,
    label: 'Benchmark',
    status: 'completed',
    created_at: '2026-03-12T00:00:00Z',
    primary_suite_id: 'det.roles_quick_v1',
    suite_ids: ['det.roles_quick_v1'],
    recipe_ids: ['detector:soccana'],
    assets: [],
    recipes: [],
    suite_results: {
      'det.roles_quick_v1': {
        'detector:soccana': {
          suite_id: 'det.roles_quick_v1',
          recipe_id: 'detector:soccana',
          status: 'completed',
          error: null,
          metrics: {
            ap_50_95: { value: 0.61, display_value: '0.6100', sort_value: 0.61, is_na: false },
          },
          flattened_metrics: { ap_50_95: 0.61 },
          primary_metric: 'ap_50_95',
          artifacts: {},
          raw_result: {},
        },
      },
    },
    progress: 100,
    logs: [],
    error: null,
    dvc_runtime: null,
    legacy_clip_status: null,
  });

  assert.equal(detail.suiteResults['det.roles_quick_v1']['detector:soccana'].metrics.ap_50_95.value, 0.61);
  assert.equal(detail.suiteResults['det.roles_quick_v1']['detector:soccana'].metrics.ap_50_95.displayValue, '0.6100');
});

test('resolveSelectedBenchmarkId prefers the active benchmark when current is stale', () => {
  const next = resolveSelectedBenchmarkId('stale', [
    { benchmarkId: 'done', label: 'done', status: 'completed', createdAt: '', primarySuiteId: null, suiteIds: [], recipeCount: 1, legacyRecord: false },
    { benchmarkId: 'active', label: 'active', status: 'running', createdAt: '', primarySuiteId: null, suiteIds: [], recipeCount: 1, legacyRecord: false },
  ]);

  assert.equal(next, 'active');
});

test('resolveSelectedSuiteIds falls back to the operational suite when current selection is invalid', () => {
  const next = resolveSelectedSuiteIds(
    ['missing'],
    [
      { id: 'ops.clip_review_v1', label: 'Operational', tier: 'operational', family: 'operational', protocol: 'operational', primaryMetric: 'fps', metricColumns: [], requiredCapabilities: [], datasetRoot: '', manifestPath: '', sourceUrl: '', license: '', notes: '', datasetSplit: null, dvcRequired: false, requiresClip: true, fallbackDatasetRoots: [] },
      { id: 'det.roles_quick_v1', label: 'Detection', tier: 'quick', family: 'detection', protocol: 'coco_detection', primaryMetric: 'ap_50_95', metricColumns: [], requiredCapabilities: [], datasetRoot: '', manifestPath: '', sourceUrl: '', license: '', notes: '', datasetSplit: null, dvcRequired: false, requiresClip: false, fallbackDatasetRoots: [] },
    ],
    new Set(['det.roles_quick_v1']),
  );

  assert.deepEqual(next, ['ops.clip_review_v1']);
});

test('resolveSelectedRecipeIds prefers the requested recipe when it exists', () => {
  const next = resolveSelectedRecipeIds(
    [],
    [
      { id: 'detector:soccana', label: 'soccana', kind: 'detector_recipe', assetId: 'detector.soccana', sourceAssetIds: [], pipeline: 'classic', detectorAssetId: null, trackerAssetId: null, requestedTrackerMode: null, keypointModel: null, bundleMode: 'separable', runtimeBinding: 'replace_component', available: true, artifactPath: '/tmp/model.pt', capabilities: { detection: true, tracking: false, reid: false, calibration: false, team_id: false, role_id: false, jersey_ocr: false, event_spotting: false }, classMapping: {}, compatibleSuiteIds: [] },
    ],
    'detector:soccana',
  );

  assert.deepEqual(next, ['detector:soccana']);
});

test('resolveDetailSelection prefers the current valid suite/recipe pair', () => {
  const detail = {
    suiteIds: ['det.roles_quick_v1'],
    recipeIds: ['detector:soccana'],
  };
  const next = resolveDetailSelection(
    { suiteId: 'det.roles_quick_v1', recipeId: 'detector:soccana' },
    detail,
    ['det.roles_quick_v1'],
    ['detector:soccana'],
  );

  assert.deepEqual(next, { suiteId: 'det.roles_quick_v1', recipeId: 'detector:soccana' });
});

test('resolveVisibleActiveSuiteId snaps back to the visible suite pool before stale benchmark suites', () => {
  const next = resolveVisibleActiveSuiteId(
    'ops.clip_review_v1',
    [
      { id: 'track.sn_tracking_medium_v1', label: 'Tracking', tier: 'medium', family: 'tracking', protocol: 'tracking', primaryMetric: 'hota', metricColumns: [], requiredCapabilities: [], datasetRoot: '', manifestPath: '', sourceUrl: '', license: '', notes: '', datasetSplit: null, dvcRequired: false, requiresClip: false, fallbackDatasetRoots: [] },
    ],
    [
      { id: 'ops.clip_review_v1', label: 'Operational', tier: 'operational', family: 'operational', protocol: 'operational', primaryMetric: 'fps', metricColumns: [], requiredCapabilities: [], datasetRoot: '', manifestPath: '', sourceUrl: '', license: '', notes: '', datasetSplit: null, dvcRequired: false, requiresClip: true, fallbackDatasetRoots: [] },
      { id: 'track.sn_tracking_medium_v1', label: 'Tracking', tier: 'medium', family: 'tracking', protocol: 'tracking', primaryMetric: 'hota', metricColumns: [], requiredCapabilities: [], datasetRoot: '', manifestPath: '', sourceUrl: '', license: '', notes: '', datasetSplit: null, dvcRequired: false, requiresClip: false, fallbackDatasetRoots: [] },
    ],
  );

  assert.equal(next, 'track.sn_tracking_medium_v1');
});

test('resolveDetailSelection prefers the current visible suite and recipe pool over stale run-wide selections', () => {
  const detail = {
    suiteIds: ['det.roles_quick_v1', 'track.sn_tracking_medium_v1', 'ops.clip_review_v1'],
    recipeIds: ['detector:soccana', 'tracker:soccana+hybrid_reid+soccana_keypoint'],
  };

  const next = resolveDetailSelection(
    { suiteId: 'det.roles_quick_v1', recipeId: 'detector:soccana' },
    detail,
    ['track.sn_tracking_medium_v1'],
    ['tracker:soccana+hybrid_reid+soccana_keypoint'],
  );

  assert.deepEqual(next, {
    suiteId: 'track.sn_tracking_medium_v1',
    recipeId: 'tracker:soccana+hybrid_reid+soccana_keypoint',
  });
});

test('mergeUniqueStrings appends only unseen values', () => {
  assert.deepEqual(mergeUniqueStrings(['a', 'b'], ['b', 'c']), ['a', 'b', 'c']);
});

function createMatrixFixture() {
  const suite = {
    id: 'det.roles_quick_v1',
    label: 'Detection Roles Quick',
    tier: 'quick',
    family: 'detection',
    protocol: 'coco_detection',
    primaryMetric: 'ap_50_95',
    metricColumns: ['ap_50_95', 'avg_image_latency_ms'],
    requiredCapabilities: ['detection'],
    datasetRoot: '',
    manifestPath: '',
    sourceUrl: '',
    license: '',
    notes: '',
    datasetSplit: 'test',
    dvcRequired: true,
    requiresClip: false,
    fallbackDatasetRoots: [],
  };
  const assets = new Map([
    ['detector.soccana', {
      assetId: 'detector.soccana',
      kind: 'detector',
      provider: 'local',
      source: 'pretrained',
      label: 'soccana',
      version: 'pretrained',
      architecture: 'ultralytics_yolo',
      artifactPath: '/tmp/soccana.pt',
      bundleMode: 'separable',
      runtimeBinding: 'replace_component',
      available: true,
      capabilities: { detection: true, tracking: false, reid: false, calibration: false, team_id: false, role_id: true, jersey_ocr: false, event_spotting: false },
      classMapping: {},
      artifactDvc: null,
      availabilityError: null,
      trainingRunId: null,
      metrics: null,
      importOrigin: null,
      importedAt: null,
    }],
    ['detector.custom', {
      assetId: 'detector.custom',
      kind: 'detector',
      provider: 'huggingface',
      source: 'huggingface',
      label: 'custom',
      version: 'v1',
      architecture: 'rtdetr',
      artifactPath: '/tmp/custom.pt',
      bundleMode: 'separable',
      runtimeBinding: 'replace_component',
      available: true,
      capabilities: { detection: true, tracking: false, reid: false, calibration: false, team_id: false, role_id: true, jersey_ocr: false, event_spotting: false },
      classMapping: {},
      artifactDvc: null,
      availabilityError: null,
      trainingRunId: null,
      metrics: null,
      importOrigin: null,
      importedAt: null,
    }],
  ]);
  const recipes = [
    {
      id: 'detector:soccana',
      label: 'soccana',
      kind: 'detector_recipe',
      assetId: 'detector.soccana',
      sourceAssetIds: ['detector.soccana'],
      pipeline: 'classic',
      detectorAssetId: 'detector.soccana',
      trackerAssetId: null,
      requestedTrackerMode: null,
      keypointModel: null,
      bundleMode: 'separable',
      runtimeBinding: 'replace_component',
      available: true,
      artifactPath: '/tmp/soccana.pt',
      capabilities: { detection: true, tracking: false, reid: false, calibration: false, team_id: false, role_id: true, jersey_ocr: false, event_spotting: false },
      classMapping: {},
      compatibleSuiteIds: ['det.roles_quick_v1'],
    },
    {
      id: 'detector:custom',
      label: 'custom',
      kind: 'detector_recipe',
      assetId: 'detector.custom',
      sourceAssetIds: ['detector.custom'],
      pipeline: 'classic',
      detectorAssetId: 'detector.custom',
      trackerAssetId: null,
      requestedTrackerMode: null,
      keypointModel: null,
      bundleMode: 'separable',
      runtimeBinding: 'replace_component',
      available: true,
      artifactPath: '/tmp/custom.pt',
      capabilities: { detection: true, tracking: false, reid: false, calibration: false, team_id: false, role_id: true, jersey_ocr: false, event_spotting: false },
      classMapping: {},
      compatibleSuiteIds: ['det.roles_quick_v1'],
    },
    {
      id: 'pipeline:soccermaster',
      label: 'SoccerMaster',
      kind: 'pipeline_recipe',
      assetId: 'pipeline.soccermaster',
      sourceAssetIds: ['detector.soccana'],
      pipeline: 'soccermaster',
      detectorAssetId: null,
      trackerAssetId: null,
      requestedTrackerMode: null,
      keypointModel: null,
      bundleMode: 'bundled',
      runtimeBinding: 'full_pipeline',
      available: true,
      artifactPath: '/tmp/soccermaster',
      capabilities: { detection: true, tracking: true, reid: false, calibration: true, team_id: true, role_id: false, jersey_ocr: false, event_spotting: false },
      classMapping: {},
      compatibleSuiteIds: [],
    },
  ];
  const detail = {
    benchmarkId: 'b1',
    schemaVersion: 2,
    legacyRecord: false,
    label: 'matrix',
    status: 'completed',
    createdAt: '2026-03-12T00:00:00Z',
    primarySuiteId: 'det.roles_quick_v1',
    suiteIds: ['det.roles_quick_v1'],
    recipeIds: recipes.map((recipe) => recipe.id),
    assets: [],
    recipes,
    suiteResults: {
      'det.roles_quick_v1': {
        'detector:soccana': {
          suiteId: 'det.roles_quick_v1',
          recipeId: 'detector:soccana',
          status: 'completed',
          error: null,
          metrics: {
            ap_50_95: { label: 'AP@[.50:.95]', value: 0.81, displayValue: '0.8100', unit: '', sortValue: 0.81, isNa: false, raw: null },
            avg_image_latency_ms: { label: 'Latency', value: 18.2, displayValue: '18.20', unit: 'ms', sortValue: 18.2, isNa: false, raw: null },
          },
          flattenedMetrics: {},
          primaryMetric: 'ap_50_95',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
        'detector:custom': {
          suiteId: 'det.roles_quick_v1',
          recipeId: 'detector:custom',
          status: 'completed',
          error: null,
          metrics: {
            ap_50_95: { label: 'AP@[.50:.95]', value: 0.67, displayValue: '0.6700', unit: '', sortValue: 0.67, isNa: false, raw: null },
            avg_image_latency_ms: { label: 'Latency', value: 9.7, displayValue: '9.70', unit: 'ms', sortValue: 9.7, isNa: false, raw: null },
          },
          flattenedMetrics: {},
          primaryMetric: 'ap_50_95',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
      },
    },
    progress: 100,
    logs: [],
    error: null,
    dvcRuntime: null,
    legacyClipStatus: null,
  };
  const previewCells = {
    'det.roles_quick_v1::detector:soccana': { status: 'ready', reason: 'Runnable now' },
    'det.roles_quick_v1::detector:custom': { status: 'ready', reason: 'Runnable now' },
    'det.roles_quick_v1::pipeline:soccermaster': { status: 'not_supported', reason: 'Recipe does not satisfy this suite capability contract.' },
  };
  return { suite, assets, recipes, detail, previewCells };
}

test('collectMatrixFilterOptions surfaces provider, architecture, capability, and status options for the active suite', () => {
  const { suite, assets, recipes, detail, previewCells } = createMatrixFixture();
  const options = collectMatrixFilterOptions(recipes, suite, detail, previewCells, assets);

  assert.deepEqual(options.providers, ['huggingface', 'local']);
  assert.deepEqual(options.architectures, ['rtdetr', 'ultralytics_yolo']);
  assert.ok(options.capabilities.includes('detection'));
  assert.ok(options.statuses.includes('completed'));
  assert.ok(options.statuses.includes('not_supported'));
});

test('filterRecipesForMatrix applies provider and supportsActiveSuite filters without hiding unsupported rows by default', () => {
  const { suite, assets, recipes, detail, previewCells } = createMatrixFixture();
  const allVisible = filterRecipesForMatrix(
    recipes,
    suite,
    detail,
    previewCells,
    { search: '', suite: 'all', tier: 'all', provider: 'all', architecture: 'all', bundleMode: 'all', status: 'all', capability: 'all', supportsActiveSuite: 'all', hasNa: 'all' },
    assets,
  );
  const supportedOnly = filterRecipesForMatrix(
    recipes,
    suite,
    detail,
    previewCells,
    { search: '', suite: 'all', tier: 'all', provider: 'all', architecture: 'all', bundleMode: 'all', status: 'all', capability: 'all', supportsActiveSuite: 'supported', hasNa: 'all' },
    assets,
  );
  const hfOnly = filterRecipesForMatrix(
    recipes,
    suite,
    detail,
    previewCells,
    { search: '', suite: 'all', tier: 'all', provider: 'huggingface', architecture: 'all', bundleMode: 'all', status: 'all', capability: 'all', supportsActiveSuite: 'all', hasNa: 'all' },
    assets,
  );

  assert.equal(allVisible.length, 3);
  assert.deepEqual(supportedOnly.map((recipe) => recipe.id), ['detector:soccana', 'detector:custom']);
  assert.deepEqual(hfOnly.map((recipe) => recipe.id), ['detector:custom']);
});

test('sortRecipesForMatrix sorts by metric descending and latency ascending while leaving missing metrics last', () => {
  const { suite, assets, recipes, detail, previewCells } = createMatrixFixture();

  const byAp = sortRecipesForMatrix(
    recipes,
    suite,
    detail,
    previewCells,
    { column: 'ap_50_95', direction: 'desc' },
    assets,
  );
  const byLatency = sortRecipesForMatrix(
    recipes,
    suite,
    detail,
    previewCells,
    { column: 'avg_image_latency_ms', direction: 'asc' },
    assets,
  );

  assert.deepEqual(byAp.map((recipe) => recipe.id), ['detector:soccana', 'detector:custom', 'pipeline:soccermaster']);
  assert.deepEqual(byLatency.map((recipe) => recipe.id), ['detector:custom', 'detector:soccana', 'pipeline:soccermaster']);
});
