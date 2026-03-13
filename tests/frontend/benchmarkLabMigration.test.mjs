import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeBenchmarkHistory, normalizeBenchmarkRunDetail } from '../../frontend/src/benchmarkLab/types.ts';

test('benchmark migration keeps legacy operational records visible in normalized history', () => {
  const history = normalizeBenchmarkHistory([
    {
      benchmark_id: 'legacy_1',
      label: 'legacy op review',
      status: 'completed',
      created_at: '2026-03-12T00:00:00Z',
      primary_suite_id: 'ops.clip_review_v1',
      suite_ids: ['ops.clip_review_v1'],
      recipe_count: 1,
      legacy_record: true,
    },
  ]);

  assert.equal(history[0].legacyRecord, true);
  assert.equal(history[0].primarySuiteId, 'ops.clip_review_v1');
});

test('benchmark migration preserves suite-result payload shape for legacy operational rows', () => {
  const detail = normalizeBenchmarkRunDetail({
    benchmark_id: 'legacy_1',
    schema_version: 2,
    legacy_record: true,
    label: 'legacy op review',
    status: 'completed',
    created_at: '2026-03-12T00:00:00Z',
    primary_suite_id: 'ops.clip_review_v1',
    suite_ids: ['ops.clip_review_v1'],
    recipe_ids: ['detector:soccana'],
    assets: [],
    recipes: [],
    suite_results: {
      'ops.clip_review_v1': {
        'detector:soccana': {
          suite_id: 'ops.clip_review_v1',
          recipe_id: 'detector:soccana',
          status: 'completed',
          error: null,
          metrics: {
            fps: { value: 24.1, display_value: '24.10', sort_value: 24.1, is_na: false },
          },
          flattened_metrics: { fps: 24.1 },
          primary_metric: 'fps',
          artifacts: {},
          raw_result: {},
          legacy_record: true,
        },
      },
    },
    progress: 100,
    logs: [],
    error: null,
    dvc_runtime: null,
    legacy_clip_status: null,
  });

  assert.equal(detail.legacyRecord, true);
  assert.equal(detail.suiteResults['ops.clip_review_v1']['detector:soccana'].legacyRecord, true);
  assert.equal(detail.suiteResults['ops.clip_review_v1']['detector:soccana'].metrics.fps.displayValue, '24.10');
});
