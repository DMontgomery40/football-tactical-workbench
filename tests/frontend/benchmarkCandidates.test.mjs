import assert from 'node:assert/strict';
import test from 'node:test';

import { buildCoreBaselinePlan, splitBenchmarkCandidates } from '../../frontend/src/benchmarkLab/candidates.js';

test('splitBenchmarkCandidates keeps baseline trio ordered before detector checkpoints', () => {
  const input = [
    { id: 'custom_b', label: 'B custom', comparison_group: 'detector' },
    { id: 'sn_gamestate', label: 'sn-gamestate', comparison_group: 'baseline' },
    { id: 'soccana', label: 'classic', comparison_group: 'baseline' },
    { id: 'custom_a', label: 'A custom', comparison_group: 'detector' },
    { id: 'soccermaster', label: 'soccermaster', comparison_group: 'baseline' },
  ];

  const result = splitBenchmarkCandidates(input);

  assert.deepEqual(result.baselineCandidates.map((candidate) => candidate.id), ['soccana', 'soccermaster', 'sn_gamestate']);
  assert.deepEqual(result.detectorCandidates.map((candidate) => candidate.id), ['custom_a', 'custom_b']);
});

test('buildCoreBaselinePlan reports unavailable baseline trio entries', () => {
  const input = [
    { id: 'soccana', label: 'classic', available: true },
    { id: 'soccermaster', label: 'soccermaster', available: true },
    { id: 'sn_gamestate', label: 'sn-gamestate', available: false, availability_note: 'Clone repo first.' },
  ];

  const result = buildCoreBaselinePlan(input);

  assert.equal(result.ready, false);
  assert.deepEqual(result.missingIds, []);
  assert.deepEqual(result.candidateIds, ['soccana', 'soccermaster', 'sn_gamestate']);
  assert.equal(result.unavailableCandidates.length, 1);
  assert.equal(result.unavailableCandidates[0].id, 'sn_gamestate');
});
