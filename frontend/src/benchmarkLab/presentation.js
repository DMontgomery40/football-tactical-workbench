import { formatCapabilityLabel } from './types.ts';

function titleCaseToken(value) {
  return String(value || '')
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export function describeLocation(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return { primary: 'Not provided', secondary: null, full: null, href: null };
  }

  try {
    const url = new URL(raw);
    const parts = url.pathname.split('/').filter(Boolean);
    const primary = parts.at(-1) || url.hostname.replace(/^www\./, '');
    const secondary = parts.length > 1
      ? `${url.hostname.replace(/^www\./, '')} / .../${parts.slice(Math.max(0, parts.length - 3), -1).join('/')}`
      : url.hostname.replace(/^www\./, '');
    return {
      primary,
      secondary,
      full: raw,
      href: raw,
    };
  } catch {
    // Treat non-URL values as filesystem-ish locations.
  }

  const parts = raw.split(/[\\/]+/).filter(Boolean);
  const primary = parts.at(-1) || raw;
  const contextParts = parts.slice(Math.max(0, parts.length - 4), -1);

  return {
    primary,
    secondary: contextParts.length > 0 ? `.../${contextParts.join('/')}` : null,
    full: raw,
    href: null,
  };
}

export function formatCapabilitySummary(value, fallback = 'No special capability gate') {
  const rawValues = Array.isArray(value)
    ? value
    : Object.entries(value || {})
        .filter(([, enabled]) => Boolean(enabled))
        .map(([capability]) => capability);

  if (rawValues.length === 0) {
    return fallback;
  }

  return rawValues.map((capability) => formatCapabilityLabel(String(capability))).join(', ');
}

export function formatRecipeStack(recipe, sourceAssets) {
  const assetById = new Map((sourceAssets || []).map((asset) => [asset.assetId, asset]));
  const parts = [];

  const detectorLabel = recipe?.detectorAssetId ? assetById.get(recipe.detectorAssetId)?.label : null;
  if (detectorLabel) {
    parts.push(`Detector ${detectorLabel}`);
  }

  const trackerLabel = recipe?.trackerAssetId ? assetById.get(recipe.trackerAssetId)?.label : null;
  if (trackerLabel) {
    parts.push(`Tracker ${trackerLabel}`);
  } else if (recipe?.requestedTrackerMode) {
    parts.push(`Tracker ${titleCaseToken(recipe.requestedTrackerMode)}`);
  }

  if (recipe?.keypointModel) {
    parts.push(`Field ${titleCaseToken(recipe.keypointModel)}`);
  }

  if (parts.length === 0 && recipe?.pipeline && recipe.pipeline !== 'classic') {
    parts.push(`${titleCaseToken(recipe.pipeline)} pipeline`);
  }

  if (parts.length === 0) {
    parts.push(`${titleCaseToken(recipe?.bundleMode || 'classic')} recipe`);
  }

  return parts.join(' · ');
}

export function formatAssetSource(asset) {
  if (!asset) {
    return 'Source unknown';
  }
  return `${titleCaseToken(asset.provider)} · ${titleCaseToken(asset.source)}`;
}

export function formatRecipeFamilyLabel(kind) {
  switch (kind) {
    case 'detector_recipe':
      return 'Detector recipes';
    case 'tracking_recipe':
      return 'Tracking recipes';
    case 'pipeline_recipe':
      return 'Bundled pipelines';
    default:
      return titleCaseToken(kind || 'recipes');
  }
}
