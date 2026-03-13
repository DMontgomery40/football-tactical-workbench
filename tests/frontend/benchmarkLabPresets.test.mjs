import assert from 'node:assert/strict';
import test from 'node:test';

import { suiteFamilyForPreset } from '../../frontend/src/benchmarkLab/utils.ts';

test('benchmark presets map to stable suite families', () => {
  assert.equal(suiteFamilyForPreset('Detection'), 'detection');
  assert.equal(suiteFamilyForPreset('Spotting'), 'spotting');
  assert.equal(suiteFamilyForPreset('Localization'), 'localization');
  assert.equal(suiteFamilyForPreset('Calibration'), 'calibration');
  assert.equal(suiteFamilyForPreset('Tracking'), 'tracking');
  assert.equal(suiteFamilyForPreset('Game State'), 'game_state');
  assert.equal(suiteFamilyForPreset('Operational'), 'operational');
});
