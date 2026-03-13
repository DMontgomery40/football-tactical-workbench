import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildRecipeComparisonData,
  buildSuiteEvaluationData,
  metricOptionsForSuite,
  toggleFocusedRecipeId,
} from '../../frontend/src/benchmarkLab/chartData.ts';
import { createMatrixFixture } from './benchmarkLabFixtures.mjs';

function metric(label, value) {
  return {
    label,
    value,
    displayValue: value === null ? 'N/A' : String(value),
    unit: '',
    sortValue: value,
    isNa: value === null,
    raw: null,
  };
}

function createSuiteEvaluationFixture() {
  const suites = [
    {
      id: 'track.sn_tracking_medium_v1',
      label: 'Tracking Medium',
      tier: 'medium',
      family: 'tracking',
      protocol: 'tracking',
      primaryMetric: 'hota',
      metricColumns: ['hota', 'frames_per_second'],
      requiredCapabilities: ['detection', 'tracking'],
      datasetRoot: '/tmp/SoccerNetMOT',
      manifestPath: '/tmp/track.json',
      sourceUrl: '',
      license: '',
      notes: '',
      datasetSplit: 'valid',
      dvcRequired: true,
      requiresClip: false,
      fallbackDatasetRoots: [],
    },
    {
      id: 'gsr.medium_v1',
      label: 'Game State Medium',
      tier: 'medium',
      family: 'game_state',
      protocol: 'gamestate',
      primaryMetric: 'gs_hota',
      metricColumns: ['gs_hota', 'frames_per_second'],
      requiredCapabilities: ['detection', 'tracking', 'calibration'],
      datasetRoot: '/tmp/SoccerNetGS',
      manifestPath: '/tmp/gsr.json',
      sourceUrl: '',
      license: '',
      notes: '',
      datasetSplit: 'valid',
      dvcRequired: true,
      requiresClip: false,
      fallbackDatasetRoots: [],
    },
  ];
  const recipes = [
    {
      id: 'tracker:soccana+hybrid_reid+soccana_keypoint',
      label: 'Classic + Hybrid ReID',
      kind: 'tracking_recipe',
      assetId: 'detector.soccana',
      sourceAssetIds: [],
      pipeline: 'classic',
      detectorAssetId: null,
      trackerAssetId: null,
      requestedTrackerMode: 'hybrid_reid',
      keypointModel: 'soccana_keypoint',
      bundleMode: 'separable',
      runtimeBinding: 'replace_component',
      available: true,
      artifactPath: '/tmp/a.pt',
      capabilities: { detection: true, tracking: true, reid: true, calibration: true, team_id: true, role_id: false, jersey_ocr: false, event_spotting: false },
      classMapping: {},
      compatibleSuiteIds: suites.map((suite) => suite.id),
    },
    {
      id: 'pipeline:soccermaster',
      label: 'SoccerMaster',
      kind: 'pipeline_recipe',
      assetId: 'pipeline.soccermaster',
      sourceAssetIds: [],
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
      compatibleSuiteIds: suites.map((suite) => suite.id),
    },
    {
      id: 'pipeline:sn-gamestate-tracklab',
      label: 'sn-gamestate / TrackLab',
      kind: 'pipeline_recipe',
      assetId: 'pipeline.tracklab_sn_gamestate',
      sourceAssetIds: [],
      pipeline: 'tracklab_gamestate',
      detectorAssetId: null,
      trackerAssetId: null,
      requestedTrackerMode: null,
      keypointModel: null,
      bundleMode: 'bundled',
      runtimeBinding: 'full_pipeline',
      available: true,
      artifactPath: '/tmp/tracklab',
      capabilities: { detection: true, tracking: true, reid: true, calibration: true, team_id: true, role_id: true, jersey_ocr: true, event_spotting: false },
      classMapping: {},
      compatibleSuiteIds: suites.map((suite) => suite.id),
    },
  ];
  const detail = {
    benchmarkId: 'suite-eval',
    schemaVersion: 2,
    legacyRecord: false,
    label: 'suite evaluation',
    status: 'completed',
    createdAt: '2026-03-12T00:00:00Z',
    primarySuiteId: 'track.sn_tracking_medium_v1',
    suiteIds: suites.map((suite) => suite.id),
    recipeIds: recipes.map((recipe) => recipe.id),
    assets: [],
    recipes,
    suiteResults: {
      'track.sn_tracking_medium_v1': {
        'tracker:soccana+hybrid_reid+soccana_keypoint': {
          suiteId: 'track.sn_tracking_medium_v1',
          recipeId: 'tracker:soccana+hybrid_reid+soccana_keypoint',
          status: 'completed',
          error: null,
          metrics: {
            hota: metric('HOTA', 0.72),
            frames_per_second: metric('Frames/s', 12.5),
          },
          flattenedMetrics: {},
          primaryMetric: 'hota',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
        'pipeline:soccermaster': {
          suiteId: 'track.sn_tracking_medium_v1',
          recipeId: 'pipeline:soccermaster',
          status: 'completed',
          error: null,
          metrics: {
            hota: metric('HOTA', 0.63),
            frames_per_second: metric('Frames/s', 18.4),
          },
          flattenedMetrics: {},
          primaryMetric: 'hota',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
        'pipeline:sn-gamestate-tracklab': {
          suiteId: 'track.sn_tracking_medium_v1',
          recipeId: 'pipeline:sn-gamestate-tracklab',
          status: 'completed',
          error: null,
          metrics: {
            hota: metric('HOTA', 0.58),
            frames_per_second: metric('Frames/s', 9.8),
          },
          flattenedMetrics: {},
          primaryMetric: 'hota',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
      },
      'gsr.medium_v1': {
        'tracker:soccana+hybrid_reid+soccana_keypoint': {
          suiteId: 'gsr.medium_v1',
          recipeId: 'tracker:soccana+hybrid_reid+soccana_keypoint',
          status: 'completed',
          error: null,
          metrics: {
            gs_hota: metric('GS-HOTA', 0.37),
            frames_per_second: metric('Frames/s', 6.1),
          },
          flattenedMetrics: {},
          primaryMetric: 'gs_hota',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
        'pipeline:soccermaster': {
          suiteId: 'gsr.medium_v1',
          recipeId: 'pipeline:soccermaster',
          status: 'completed',
          error: null,
          metrics: {
            gs_hota: metric('GS-HOTA', 0.41),
            frames_per_second: metric('Frames/s', 7.2),
          },
          flattenedMetrics: {},
          primaryMetric: 'gs_hota',
          artifacts: {},
          rawResult: {},
          legacyRecord: false,
        },
        'pipeline:sn-gamestate-tracklab': {
          suiteId: 'gsr.medium_v1',
          recipeId: 'pipeline:sn-gamestate-tracklab',
          status: 'blocked',
          error: 'Benchmark execution is still blocked.',
          metrics: {
            gs_hota: metric('GS-HOTA', null),
            frames_per_second: metric('Frames/s', null),
          },
          flattenedMetrics: {},
          primaryMetric: 'gs_hota',
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
  const previewCells = {};
  const datasetStateById = new Map([
    ['track.sn_tracking_medium_v1', {
      suiteId: 'track.sn_tracking_medium_v1',
      datasetRoot: '/tmp/SoccerNetMOT',
      datasetExists: true,
      datasetDvc: null,
      manifestPath: '/tmp/track.json',
      manifestExists: true,
      manifestDvc: null,
      conversionRoot: '/tmp/track-conv',
      ready: false,
      requiresClip: false,
      dvcRequired: true,
      dvcRuntime: null,
      note: 'Blocked by adapter',
    }],
    ['gsr.medium_v1', {
      suiteId: 'gsr.medium_v1',
      datasetRoot: '/tmp/SoccerNetGS',
      datasetExists: true,
      datasetDvc: null,
      manifestPath: '/tmp/gsr.json',
      manifestExists: true,
      manifestDvc: null,
      conversionRoot: '/tmp/gsr-conv',
      ready: false,
      requiresClip: false,
      dvcRequired: true,
      dvcRuntime: null,
      note: 'Blocked by adapter',
    }],
  ]);
  return { suites, recipes, detail, previewCells, datasetStateById };
}

test('benchmark chart metric options classify primary and runtime metrics', () => {
  const { suite } = createMatrixFixture();
  const options = metricOptionsForSuite(suite);

  assert.deepEqual(options.map((option) => [option.id, option.kind]), [
    ['ap_50_95', 'primary'],
    ['avg_image_latency_ms', 'runtime'],
  ]);
});

test('benchmark chart recipe focus toggles preserve at least one visible series', () => {
  const allIds = ['detector:soccana', 'detector:custom'];

  assert.deepEqual(toggleFocusedRecipeId([], 'detector:soccana', allIds), ['detector:custom']);
  assert.deepEqual(toggleFocusedRecipeId(['detector:soccana'], 'detector:soccana', allIds), allIds);
});

test('benchmark chart comparison data follows focused recipes and preserves selection metadata', () => {
  const { suite, recipes, detail, previewCells } = createMatrixFixture();
  const data = buildRecipeComparisonData(
    recipes,
    suite,
    detail,
    previewCells,
    'ap_50_95',
    ['detector:soccana', 'pipeline:soccermaster'],
    'detector:soccana',
    'detector:soccana',
  );

  assert.deepEqual(data.map((datum) => datum.recipeId), ['detector:soccana', 'pipeline:soccermaster']);
  assert.equal(data[0].selected, true);
  assert.equal(data[0].highlighted, true);
  assert.equal(data[1].status, 'not_supported');
  assert.equal(data[1].value, null);
});

test('suite evaluation charts derive comparable counts, blocked rates, and rank correlation', () => {
  const { suites, recipes, detail, previewCells, datasetStateById } = createSuiteEvaluationFixture();
  const data = buildSuiteEvaluationData(
    suites,
    recipes,
    detail,
    previewCells,
    datasetStateById,
    'track.sn_tracking_medium_v1',
  );

  assert.equal(data[0].suiteId, 'track.sn_tracking_medium_v1');
  assert.equal(data[0].comparableRecipes, 3);
  assert.equal(data[0].rankCorrelation, 1);
  assert.equal(data[1].suiteId, 'gsr.medium_v1');
  assert.equal(data[1].comparableRecipes, 2);
  assert.equal(data[1].blockedRate, 33.3);
  assert.equal(data[1].materializationCoverage, 66.7);
  assert.equal(data[1].rankCorrelation, -1);
});
