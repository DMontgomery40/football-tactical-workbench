import assert from 'node:assert/strict';
import test from 'node:test';

import { collectMatrixFilterOptions, filterRecipesForMatrix } from '../../frontend/src/benchmarkLab/utils.ts';
import { createMatrixFixture } from './benchmarkLabFixtures.mjs';

test('benchmark filters expose active-suite provider, architecture, capability, and status options', () => {
  const { suite, assets, recipes, detail, previewCells } = createMatrixFixture();
  const options = collectMatrixFilterOptions(recipes, suite, detail, previewCells, assets);

  assert.deepEqual(options.providers, ['huggingface', 'local']);
  assert.deepEqual(options.architectures, ['rtdetr', 'ultralytics_yolo']);
  assert.ok(options.capabilities.includes('detection'));
  assert.ok(options.statuses.includes('completed'));
  assert.ok(options.statuses.includes('not_supported'));
});

test('benchmark filters keep unsupported rows visible by default and narrow rows when requested', () => {
  const { suite, assets, recipes, detail, previewCells } = createMatrixFixture();
  const defaultRows = filterRecipesForMatrix(
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

  assert.equal(defaultRows.length, 3);
  assert.deepEqual(supportedOnly.map((recipe) => recipe.id), ['detector:soccana', 'detector:custom']);
});
