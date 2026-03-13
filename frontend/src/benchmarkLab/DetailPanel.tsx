import { HelpPopover } from '../helpUi';
import { formatDvcRuntime, formatTimestamp } from '../trainingStudio/formatters.js';
import CompactLocation from './CompactLocation';
import { formatAssetSource, formatCapabilitySummary } from './presentation.js';
import type { BenchmarkAsset, BenchmarkDatasetState, BenchmarkRecipe, BenchmarkRunDetail, BenchmarkRunResult, BenchmarkSuite } from './types';
import {
  formatMetricDisplay,
  formatResultStatus,
  getPrimaryMetricDisplay,
} from './types';
import { cx, displayTitleClass, insetPanelClass, panelClass, sectionHeadingClass, statusTone } from './ui';

const METRIC_HELP_IDS: Record<string, string> = {
  fps: 'benchmark.metric.throughput',
  track_stability: 'benchmark.metric.track_stability',
  calibration: 'benchmark.metric.calibration_health',
  coverage: 'benchmark.metric.coverage',
};

const disclosureSummaryClass =
  'flex list-none cursor-pointer items-center justify-between gap-3 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/80 px-4 py-3 transition hover:border-[color:var(--accent)]/45 hover:bg-[color:var(--accent-soft)]/18';

function isLocationLike(value: unknown): value is string {
  return typeof value === 'string' && (value.includes('/') || value.includes('\\') || value.startsWith('http://') || value.startsWith('https://'));
}

interface DetailPanelProps {
  suite: BenchmarkSuite | null;
  recipe: BenchmarkRecipe | null;
  result: BenchmarkRunResult | null;
  benchmark: BenchmarkRunDetail | null;
  datasetState: BenchmarkDatasetState | null;
  assetById: Map<string, BenchmarkAsset>;
  helpIndex: Map<string, unknown>;
}

export default function DetailPanel({
  suite,
  recipe,
  result,
  benchmark,
  datasetState,
  assetById,
  helpIndex,
}: DetailPanelProps) {
  const sourceAssets = recipe?.sourceAssetIds.map((assetId) => assetById.get(assetId)).filter(Boolean) as BenchmarkAsset[] | undefined;
  const metricOrder = suite?.metricColumns || Object.keys(result?.metrics || {});
  const artifactEntries = Object.entries(result?.artifacts || {}).filter(([, value]) => value !== null && value !== undefined && value !== '');
  const hasSelection = Boolean(suite && recipe);
  const benchmarkDvcRuntimeJson = benchmark?.dvcRuntime ? JSON.stringify(benchmark.dvcRuntime, null, 2) : '';

  if (!hasSelection) {
    return (
      <section className={cx(panelClass, 'min-w-0 space-y-3')}>
        <div className={sectionHeadingClass}>Detail Panel</div>
        <div className={displayTitleClass}>Select a matrix cell</div>
        <p className="text-sm leading-6 text-[var(--text-muted)]">
          Pick any suite/recipe intersection to inspect its metrics, artifacts, and operational fit.
        </p>
      </section>
    );
  }

  return (
    <section className={cx(panelClass, 'min-w-0 space-y-5')}>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <div className={sectionHeadingClass}>Detail Panel</div>
          <div className={displayTitleClass}>{recipe?.label}</div>
          <div className="flex flex-wrap items-center gap-2 text-sm text-[var(--text-muted)]">
            <span>{suite?.label}</span>
            {result ? (
              <span className={cx('rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]', statusTone(result.status))}>
                {formatResultStatus(result.status)}
              </span>
            ) : null}
          </div>
        </div>
        <HelpPopover entry={helpIndex.get('benchmark.detail_panel')} />
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className={cx(insetPanelClass, 'min-w-0')}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Primary metric</div>
          <div className="mt-2 text-3xl font-semibold tracking-[-0.03em] text-[var(--text-strong)]">
            {result ? getPrimaryMetricDisplay(result, suite) : 'Preview'}
          </div>
          <div className="mt-1 text-sm text-[var(--text-muted)]">
            {suite?.primaryMetric || 'No primary metric declared'}
          </div>
        </div>
        <div className={cx(insetPanelClass, 'min-w-0')}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Benchmark run</div>
          <div className="mt-2 text-base font-semibold text-[var(--text-strong)]">
            {benchmark?.label || benchmark?.benchmarkId || 'Not run yet'}
          </div>
          <div className="mt-1 text-sm text-[var(--text-muted)]">
            {benchmark?.createdAt ? formatTimestamp(benchmark.createdAt) : 'Run this selection to persist a benchmark record.'}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
        <div className="min-w-0 space-y-4">
          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className="flex items-center gap-2">
              <div className={sectionHeadingClass}>Metrics</div>
              <HelpPopover entry={helpIndex.get('benchmark.results_matrix')} />
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {metricOrder.map((metricId) => {
                const metric = result?.metrics?.[metricId];
                return (
                  <div key={metricId} className="min-w-0 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/75 p-3">
                    <div className="flex items-center gap-2">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                        {metricId}
                      </div>
                      {METRIC_HELP_IDS[metricId] ? <HelpPopover entry={helpIndex.get(METRIC_HELP_IDS[metricId])} /> : null}
                    </div>
                    <div className="mt-2 text-xl font-semibold text-[var(--text-strong)]">
                      {metric ? formatMetricDisplay(metric) : 'Not available'}
                    </div>
                    <div className="mt-1 text-xs text-[var(--text-muted)]">{metric?.label || 'No metric explanation from backend.'}</div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className={sectionHeadingClass}>Artifacts</div>
            {artifactEntries.length > 0 ? (
              <div className="space-y-3">
                {artifactEntries.map(([artifactId, artifactValue]) => (
                  <div key={artifactId} className="min-w-0 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/75 p-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                      {artifactId}
                    </div>
                    <div className="mt-2">
                      {isLocationLike(artifactValue) ? (
                        <CompactLocation value={artifactValue} detailsLabel="Show full artifact location" />
                      ) : (
                        <div className="text-sm text-[var(--text)] [overflow-wrap:anywhere]">{String(artifactValue)}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-[var(--text-muted)]">This cell does not expose any artifact paths yet.</p>
            )}
          </div>
        </div>

        <div className="min-w-0 space-y-4">
          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className={sectionHeadingClass}>Recipe Composition</div>
            <div className="space-y-3">
              {sourceAssets?.map((asset) => (
                <div key={asset.assetId} className="min-w-0 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/75 p-3">
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-[var(--text-strong)]">{asset.label}</div>
                    <div className="text-xs uppercase tracking-[0.12em] text-[var(--text-muted)]">
                      {asset.kind} · {formatAssetSource(asset)}
                    </div>
                  </div>
                  <div className="mt-3">
                    <CompactLocation value={asset.artifactPath} detailsLabel="Show full asset location" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className={sectionHeadingClass}>Source + durability</div>
            <div className="grid gap-3">
              <div className="min-w-0 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/75 p-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Suite license</div>
                <div className="mt-2 text-sm text-[var(--text)]">{suite?.license || 'Not specified'}</div>
              </div>
              <div className="min-w-0 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/75 p-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Benchmark DVC runtime</div>
                <div className="mt-2 text-sm text-[var(--text)]">{formatDvcRuntime(benchmark?.dvcRuntime)}</div>
                {benchmarkDvcRuntimeJson ? (
                  <details className="group mt-3">
                    <summary className="cursor-pointer list-none text-xs font-medium text-[var(--accent-strong)] hover:text-[var(--accent)]">
                      Show raw DVC runtime payload
                    </summary>
                    <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/85 p-3 text-xs leading-5 text-[var(--text-muted)] [overflow-wrap:anywhere]">
                      {benchmarkDvcRuntimeJson}
                    </pre>
                  </details>
                ) : null}
              </div>
              <div className="min-w-0 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/75 p-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Source assets</div>
                <div className="mt-2 text-sm text-[var(--text)]">
                  {(sourceAssets || []).map((asset) => asset.label).join(' · ') || 'No source assets'}
                </div>
              </div>
            </div>
          </div>

          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className={sectionHeadingClass}>Benchmark Context</div>
            <div className="space-y-3 text-sm text-[var(--text)]">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Suite source</div>
                <div className="mt-2">
                  <CompactLocation value={suite?.sourceUrl} fallback="Not declared" detailsLabel="Show source URL" />
                </div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Suite license</div>
                <div className="mt-1">{suite?.license || 'Not declared'}</div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Benchmark DVC runtime</div>
                <div className="mt-1">{formatDvcRuntime(benchmark?.dvcRuntime)}</div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Runtime context</div>
                <div className="mt-1">
                  {result?.runtimeContext?.durationSeconds != null
                    ? `${result.runtimeContext.durationSeconds.toFixed(2)} s`
                    : 'No runtime context captured for this cell'}
                </div>
              </div>
            </div>
          </div>

          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className={sectionHeadingClass}>Dataset Readiness</div>
            <div className="space-y-3 text-sm text-[var(--text)]">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">State</div>
                <div className="mt-1">{datasetState?.readinessStatus || 'unknown'}</div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Dataset root</div>
                <div className="mt-2">
                  <CompactLocation
                    value={datasetState?.datasetRoot}
                    fallback="No dataset root declared"
                    detailsLabel="Show dataset root"
                  />
                </div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Manifest summary</div>
                <div className="mt-1">
                  {datasetState?.manifestSummary?.itemCount != null
                    ? `${datasetState.manifestSummary.itemCount} items`
                    : 'No fixed item count in manifest'}
                  {datasetState?.manifestSummary?.classCount != null ? ` · ${datasetState.manifestSummary.classCount} classes` : ''}
                </div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Readiness note</div>
                <div className="mt-1">{datasetState?.note || 'No readiness note'}</div>
              </div>
            </div>
          </div>

          <div className={cx(insetPanelClass, 'min-w-0 space-y-3')}>
            <div className={sectionHeadingClass}>Capabilities + requirements</div>
            <div className="space-y-3 text-sm text-[var(--text)]">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Suite requires</div>
                <div className="mt-1">{formatCapabilitySummary(suite?.requiredCapabilities || [])}</div>
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Recipe provides</div>
                <div className="mt-1">{formatCapabilitySummary(recipe?.capabilities || {}, 'No declared recipe capabilities')}</div>
              </div>
            </div>
          </div>

          {result?.error ? <div className="error-box">{result.error}</div> : null}

          {result?.blockers?.length ? (
            <div className={cx(insetPanelClass, 'min-w-0 space-y-2 border-amber-500/25 bg-amber-500/8')}>
              <div className={sectionHeadingClass}>Compatibility Notes</div>
              {result.blockers.map((blocker) => (
                <p key={blocker} className="text-sm leading-6 text-[var(--text)]">
                  {blocker}
                </p>
              ))}
            </div>
          ) : null}

          {benchmark?.logs?.length ? (
            <details className={cx(insetPanelClass, 'min-w-0 group')}>
              <summary className={disclosureSummaryClass}>
                <span className="text-sm font-semibold text-[var(--text-strong)]">
                  Benchmark logs ({benchmark.logs.length})
                </span>
                <span className="text-xs font-medium uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Click to expand
                </span>
              </summary>
              <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/85 p-3 text-xs leading-5 text-[var(--text-muted)] [overflow-wrap:anywhere]">
                {benchmark.logs.join('\n')}
              </pre>
            </details>
          ) : null}

          {(result?.rawResult && Object.keys(result.rawResult).length > 0) || (recipe && suite) ? (
            <details className={cx(insetPanelClass, 'min-w-0 group')}>
              <summary className={disclosureSummaryClass}>
                <span className="text-sm font-semibold text-[var(--text-strong)]">
                  Raw benchmark payload
                </span>
                <span className="text-xs font-medium uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Click to expand
                </span>
              </summary>
              <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/85 p-3 text-xs leading-5 text-[var(--text-muted)] [overflow-wrap:anywhere]">
                {JSON.stringify(result?.rawResult || { suite, recipe }, null, 2)}
              </pre>
            </details>
          ) : null}
        </div>
      </div>
    </section>
  );
}
