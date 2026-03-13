import assert from 'node:assert/strict';
import test from 'node:test';

import {
  describeLocation,
  formatCapabilitySummary,
  formatRecipeStack,
} from '../../frontend/src/benchmarkLab/presentation.js';

test('describeLocation compresses filesystem paths into a readable primary label and context', () => {
  const display = describeLocation('/Users/davidmontgomery/football_pose_workbench/backend/benchmarks/run_01/artifacts/predictions.json');

  assert.equal(display.primary, 'predictions.json');
  assert.equal(display.secondary, '.../benchmarks/run_01/artifacts');
  assert.equal(display.href, null);
});

test('describeLocation keeps URLs compact while preserving the full link', () => {
  const display = describeLocation('https://huggingface.co/datasets/martinjolif/football-player-detection');

  assert.equal(display.primary, 'football-player-detection');
  assert.equal(display.secondary, 'huggingface.co / .../datasets/martinjolif');
  assert.equal(display.href, 'https://huggingface.co/datasets/martinjolif/football-player-detection');
});

test('formatCapabilitySummary turns capability maps into readable text', () => {
  const summary = formatCapabilitySummary({
    detection: true,
    tracking: true,
    role_id: false,
    calibration: true,
  });

  assert.equal(summary, 'Detection, Tracking, Calibration');
});

test('formatRecipeStack prefers detector, tracker, and field binding labels over raw ids', () => {
  const summary = formatRecipeStack(
    {
      id: 'tracker:soccana+hybrid_reid+soccana_keypoint',
      label: 'soccana + Hybrid ReID',
      kind: 'tracking_recipe',
      assetId: 'recipe.tracker',
      sourceAssetIds: ['detector.soccana', 'tracker.hybrid'],
      pipeline: 'classic',
      detectorAssetId: 'detector.soccana',
      trackerAssetId: 'tracker.hybrid',
      requestedTrackerMode: 'hybrid_reid',
      keypointModel: 'soccana_keypoint',
      bundleMode: 'separable',
      runtimeBinding: 'replace_component',
      available: true,
      artifactPath: '/tmp/tracker',
      capabilities: {},
      classMapping: {},
      compatibleSuiteIds: [],
    },
    [
      {
        assetId: 'detector.soccana',
        kind: 'detector',
        provider: 'local',
        source: 'pretrained',
        label: 'soccana',
      },
      {
        assetId: 'tracker.hybrid',
        kind: 'tracker',
        provider: 'local',
        source: 'runtime',
        label: 'Hybrid ReID',
      },
    ],
  );

  assert.equal(summary, 'Detector soccana · Tracker Hybrid ReID · Field Soccana Keypoint');
});
