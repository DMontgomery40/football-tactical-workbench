export function formatTimestamp(value) {
  if (!value) return 'Unknown';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function formatMetric(value) {
  if (value === null || value === undefined || value === '') return 'n/a';
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return numeric.toFixed(3);
}

export function formatPathTail(value) {
  if (!value) return 'n/a';
  const parts = String(value).split(/[\\/]+/).filter(Boolean);
  if (parts.length <= 3) return String(value);
  return `.../${parts.slice(-3).join('/')}`;
}

export function formatClassIds(value) {
  if (!Array.isArray(value) || value.length === 0) return 'none';
  return value.join(', ');
}

export function formatCurveAxis(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '';
  return `E${numeric.toFixed(1)}`;
}

export function formatCurveTooltipLabel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'Training sample';
  return `Epoch progress ${numeric.toFixed(2)}`;
}

export function formatCurveValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '';
  const abs = Math.abs(numeric);
  if (abs >= 100) return numeric.toFixed(0);
  if (abs >= 10) return numeric.toFixed(1);
  if (abs >= 1) return numeric.toFixed(2);
  if (abs >= 0.1) return numeric.toFixed(3);
  if (abs === 0) return '0';
  return numeric.toExponential(1);
}

export function formatLearningRate(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '';
  if (numeric === 0) return '0';
  return numeric.toExponential(1);
}

export function expandChartDomain([min, max]) {
  const safeMin = Number.isFinite(min) ? min : 0;
  const safeMax = Number.isFinite(max) ? max : 0;
  if (safeMin === safeMax) {
    const pad = safeMax === 0 ? 1 : Math.abs(safeMax) * 0.18;
    return [Math.max(0, safeMin - pad), safeMax + pad];
  }
  const span = safeMax - safeMin;
  return [Math.max(0, safeMin - span * 0.12), safeMax + span * 0.12];
}
