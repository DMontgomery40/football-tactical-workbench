import assert from 'node:assert/strict';
import test from 'node:test';

import { sortRecipesForMatrix } from '../../frontend/src/benchmarkLab/utils.ts';
import { createMatrixFixture } from './benchmarkLabFixtures.mjs';

test('benchmark matrix sorts by primary metric descending and latency ascending', () => {
  const { suite, assets, recipes, detail, previewCells } = createMatrixFixture();

  const byAp = sortRecipesForMatrix(recipes, suite, detail, previewCells, { column: 'ap_50_95', direction: 'desc' }, assets);
  const byLatency = sortRecipesForMatrix(recipes, suite, detail, previewCells, { column: 'avg_image_latency_ms', direction: 'asc' }, assets);

  assert.deepEqual(byAp.map((recipe) => recipe.id), ['detector:soccana', 'detector:custom', 'pipeline:soccermaster']);
  assert.deepEqual(byLatency.map((recipe) => recipe.id), ['detector:custom', 'detector:soccana', 'pipeline:soccermaster']);
});
