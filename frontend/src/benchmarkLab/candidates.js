export const CORE_BASELINE_IDS = ['soccana', 'soccermaster', 'sn_gamestate'];

export function splitBenchmarkCandidates(candidates) {
  const baselineCandidates = [];
  const detectorCandidates = [];

  for (const candidate of Array.isArray(candidates) ? candidates : []) {
    if (candidate?.comparison_group === 'baseline') {
      baselineCandidates.push(candidate);
    } else {
      detectorCandidates.push(candidate);
    }
  }

  const baselineOrder = new Map(CORE_BASELINE_IDS.map((id, index) => [id, index]));
  baselineCandidates.sort((left, right) => {
    const leftOrder = baselineOrder.get(left?.id) ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = baselineOrder.get(right?.id) ?? Number.MAX_SAFE_INTEGER;
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    return String(left?.label || left?.id || '').localeCompare(String(right?.label || right?.id || ''));
  });
  detectorCandidates.sort((left, right) => String(left?.label || left?.id || '').localeCompare(String(right?.label || right?.id || '')));

  return { baselineCandidates, detectorCandidates };
}

export function buildCoreBaselinePlan(candidates) {
  const candidateMap = new Map((Array.isArray(candidates) ? candidates : []).map((candidate) => [candidate?.id, candidate]));
  const orderedCandidates = CORE_BASELINE_IDS.map((id) => candidateMap.get(id)).filter(Boolean);
  const missingIds = CORE_BASELINE_IDS.filter((id) => !candidateMap.has(id));
  const unavailableCandidates = orderedCandidates.filter((candidate) => candidate?.available === false);

  return {
    candidateIds: orderedCandidates.map((candidate) => candidate.id),
    candidates: orderedCandidates,
    missingIds,
    unavailableCandidates,
    ready: missingIds.length === 0 && unavailableCandidates.length === 0,
  };
}
