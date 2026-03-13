import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { HelpPopover } from '../helpUi';
import type { BenchmarkRecipe, BenchmarkRunDetail, BenchmarkSuite } from './types';
import {
  buildRecipeComparisonData,
  buildRecipeLegendEntries,
  buildRecipeMetricProfileData,
  metricOptionsForSuite,
  type BenchmarkChartMode,
  type ChartMetricOption,
} from './chartData';
import { cx, displayTitleClass, insetPanelClass, panelClass, sectionHeadingClass } from './ui';
import type { PreviewCell } from './utils';

interface BenchmarkChartsProps {
  suites: BenchmarkSuite[];
  activeSuite: BenchmarkSuite | null;
  recipes: BenchmarkRecipe[];
  currentRun: BenchmarkRunDetail | null;
  previewCells: Record<string, PreviewCell>;
  chartMode: BenchmarkChartMode;
  selectedMetricId: string;
  focusedRecipeIds: string[];
  highlightedRecipeId: string;
  selectedRecipeId: string;
  helpIndex: Map<string, unknown>;
  onActiveSuiteChange: (suiteId: string) => void;
  onChartModeChange: (mode: BenchmarkChartMode) => void;
  onMetricChange: (metricId: string) => void;
  onRecipeHighlight: (recipeId: string) => void;
  onRecipeHighlightClear: () => void;
  onRecipeSelect: (recipeId: string) => void;
  onRecipeToggle: (recipeId: string) => void;
  onResetRecipeFocus: () => void;
}

function RecipeLegend({
  entries,
  onToggle,
  onSelect,
  onHover,
  onLeave,
}: {
  entries: ReturnType<typeof buildRecipeLegendEntries>;
  onToggle: (recipeId: string) => void;
  onSelect: (recipeId: string) => void;
  onHover: (recipeId: string) => void;
  onLeave: () => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {entries.map((entry) => (
        <button
          key={entry.recipeId}
          className={cx(
            'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold transition',
            entry.visible
              ? 'border-transparent bg-[color:var(--accent-soft)]/55 text-[var(--text-strong)]'
              : 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text-muted)]',
            entry.selected && 'shadow-[0_0_0_1px_var(--accent)]',
            entry.highlighted && 'scale-[1.02]',
          )}
          type="button"
          onClick={() => onSelect(entry.recipeId)}
          onDoubleClick={() => onToggle(entry.recipeId)}
          onMouseEnter={() => onHover(entry.recipeId)}
          onMouseLeave={onLeave}
        >
          <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
          {entry.label}
          <span className="rounded-full border border-[color:var(--line)] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-[var(--text-muted)]">
            {entry.status}
          </span>
        </button>
      ))}
    </div>
  );
}

function ComparisonChart({
  title,
  subtitle,
  helpEntry,
  mode,
  data,
  onRecipeSelect,
}: {
  title: string;
  subtitle: string;
  helpEntry: unknown;
  mode: BenchmarkChartMode;
  data: ReturnType<typeof buildRecipeComparisonData>;
  onRecipeSelect: (recipeId: string) => void;
}) {
  const chartData = data.map((entry) => ({
    recipeId: entry.recipeId,
    label: entry.label,
    value: entry.value,
    note: entry.note,
    color: entry.color,
    status: entry.status,
  }));

  const handleChartClick = (state: any) => {
    const recipeId = state?.activePayload?.[0]?.payload?.recipeId;
    if (recipeId) {
      onRecipeSelect(recipeId);
    }
  };

  return (
    <div className={cx(insetPanelClass, 'space-y-4')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>{title}</div>
          <div className="text-sm leading-6 text-[var(--text-muted)]">{subtitle}</div>
        </div>
        <HelpPopover entry={helpEntry} />
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          {mode === 'line' ? (
            <LineChart data={chartData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }} onClick={handleChartClick}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              <Line dataKey="value" name={title} stroke="#1f5f92" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          ) : (
            <BarChart data={chartData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }} onClick={handleChartClick}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              <Bar dataKey="value" name={title} radius={[8, 8, 0, 0]}>
                {chartData.map((entry) => (
                  <Cell key={entry.recipeId} fill={entry.color} opacity={entry.value === null ? 0.2 : 0.92} />
                ))}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MetricProfileChart({
  title,
  helpEntry,
  mode,
  profileData,
  legendEntries,
  onRecipeSelect,
}: {
  title: string;
  helpEntry: unknown;
  mode: BenchmarkChartMode;
  profileData: ReturnType<typeof buildRecipeMetricProfileData>;
  legendEntries: ReturnType<typeof buildRecipeLegendEntries>;
  onRecipeSelect: (recipeId: string) => void;
}) {
  const visibleEntries = legendEntries.filter((entry) => entry.visible);

  return (
    <div className={cx(insetPanelClass, 'space-y-4')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>{title}</div>
          <div className="text-sm leading-6 text-[var(--text-muted)]">
            Normalized 0-100 profiles let you compare how each visible recipe behaves across the suite’s metric family.
          </div>
        </div>
        <HelpPopover entry={helpEntry} />
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          {mode === 'line' ? (
            <LineChart data={profileData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} domain={[0, 100]} />
              <Tooltip />
              {visibleEntries.map((entry) => (
                <Line
                  key={entry.recipeId}
                  dataKey={entry.recipeId}
                  name={entry.label}
                  stroke={entry.color}
                  strokeWidth={entry.highlighted || entry.selected ? 3.5 : 2.5}
                  dot={{ r: 3 }}
                  activeDot={{
                    r: 6,
                    onClick: () => onRecipeSelect(entry.recipeId),
                  }}
                  opacity={entry.highlighted || entry.selected ? 1 : 0.45}
                />
              ))}
            </LineChart>
          ) : (
            <BarChart data={profileData} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} domain={[0, 100]} />
              <Tooltip />
              {visibleEntries.map((entry) => (
                <Bar
                  key={entry.recipeId}
                  dataKey={entry.recipeId}
                  name={entry.label}
                  fill={entry.color}
                  opacity={entry.highlighted || entry.selected ? 0.95 : 0.45}
                  radius={[8, 8, 0, 0]}
                  onClick={() => onRecipeSelect(entry.recipeId)}
                />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function BenchmarkCharts({
  suites,
  activeSuite,
  recipes,
  currentRun,
  previewCells,
  chartMode,
  selectedMetricId,
  focusedRecipeIds,
  highlightedRecipeId,
  selectedRecipeId,
  helpIndex,
  onActiveSuiteChange,
  onChartModeChange,
  onMetricChange,
  onRecipeHighlight,
  onRecipeHighlightClear,
  onRecipeSelect,
  onRecipeToggle,
  onResetRecipeFocus,
}: BenchmarkChartsProps) {
  const metricOptions = metricOptionsForSuite(activeSuite);
  const selectedMetric = metricOptions.find((option) => option.id === selectedMetricId) || metricOptions[0] || null;
  const runtimeMetric = metricOptions.find((option) => option.kind === 'runtime') || null;
  const legendEntries = buildRecipeLegendEntries(
    recipes,
    activeSuite,
    currentRun,
    previewCells,
    focusedRecipeIds,
    highlightedRecipeId,
    selectedRecipeId,
  );
  const primaryData = buildRecipeComparisonData(
    recipes,
    activeSuite,
    currentRun,
    previewCells,
    activeSuite?.primaryMetric || '',
    focusedRecipeIds,
    highlightedRecipeId,
    selectedRecipeId,
  );
  const runtimeData = runtimeMetric
    ? buildRecipeComparisonData(
      recipes,
      activeSuite,
      currentRun,
      previewCells,
      runtimeMetric.id,
      focusedRecipeIds,
      highlightedRecipeId,
      selectedRecipeId,
    )
    : [];
  const metricProfileData = buildRecipeMetricProfileData(
    recipes,
    activeSuite,
    currentRun,
    previewCells,
    focusedRecipeIds,
  );

  if (!activeSuite) {
    return (
      <section className={cx(panelClass, 'space-y-3')}>
        <div className={sectionHeadingClass}>Benchmark Charts</div>
        <div className={displayTitleClass}>Choose an active suite first</div>
        <p className="text-sm leading-6 text-[var(--text-muted)]">
          Charts appear once the matrix has an honest suite in focus.
        </p>
      </section>
    );
  }

  return (
    <section className={cx(panelClass, 'space-y-6')}>
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Benchmark Charts</div>
          <div className={displayTitleClass}>Interactive recipe comparisons</div>
          <p className="max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
            Double-click a recipe legend pill to hide or restore it. Single-click any pill, bar, or point to drive the
            detail panel without leaving the matrix context.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <HelpPopover entry={helpIndex.get('benchmark.charts')} />
          <button
            className={cx(
              'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em]',
              chartMode === 'line'
                ? 'border-[color:var(--accent)] bg-[color:var(--accent-soft)] text-[var(--accent-strong)]'
                : 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text-muted)]',
            )}
            type="button"
            onClick={() => onChartModeChange('line')}
          >
            Line
          </button>
          <button
            className={cx(
              'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em]',
              chartMode === 'bar'
                ? 'border-[color:var(--accent)] bg-[color:var(--accent-soft)] text-[var(--accent-strong)]'
                : 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text-muted)]',
            )}
            type="button"
            onClick={() => onChartModeChange('bar')}
          >
            Bar
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {suites.map((suite) => (
          <button
            key={suite.id}
            className={cx(
              'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] transition',
              suite.id === activeSuite.id
                ? 'border-[color:var(--accent)] bg-[color:var(--accent-soft)] text-[var(--accent-strong)]'
                : 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text-muted)] hover:border-[color:var(--accent)]/40 hover:text-[var(--text-strong)]',
            )}
            type="button"
            onClick={() => onActiveSuiteChange(suite.id)}
          >
            {suite.label}
          </button>
        ))}
      </div>

      <div className={cx(insetPanelClass, 'space-y-4')}>
        <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1">
            <div className={sectionHeadingClass}>Recipe Selection</div>
            <div className="text-sm leading-6 text-[var(--text-muted)]">
              The legend is the recipe subset controller for both the charts and the matrix. Hover highlights, single-click selects, double-click toggles visibility.
            </div>
          </div>
          <div className="flex items-center gap-2">
            <HelpPopover entry={helpIndex.get('benchmark.recipe_subset')} />
            <button
              className="rounded-full border border-[color:var(--line)] bg-[var(--surface)] px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]"
              type="button"
              onClick={onResetRecipeFocus}
            >
              Show all recipes
            </button>
          </div>
        </div>

        <RecipeLegend
          entries={legendEntries}
          onToggle={onRecipeToggle}
          onSelect={onRecipeSelect}
          onHover={onRecipeHighlight}
          onLeave={onRecipeHighlightClear}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ComparisonChart
          title="Primary Metric"
          subtitle={activeSuite.primaryMetric || 'No declared primary metric'}
          helpEntry={helpIndex.get('benchmark.metric.primary_comparison')}
          mode={chartMode}
          data={primaryData}
          onRecipeSelect={onRecipeSelect}
        />

        <ComparisonChart
          title="Runtime Comparison"
          subtitle={runtimeMetric?.label || 'No runtime metric declared for this suite'}
          helpEntry={helpIndex.get('benchmark.metric.runtime_burden')}
          mode={chartMode}
          data={runtimeData}
          onRecipeSelect={onRecipeSelect}
        />
      </div>

      <div className={cx(insetPanelClass, 'space-y-4')}>
        <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1">
            <div className={sectionHeadingClass}>Metric Explorer</div>
            <div className="text-sm leading-6 text-[var(--text-muted)]">
              The dropdown controls the matrix-adjacent spotlight metric, while the normalized profile chart exposes how each visible recipe spreads across the suite’s metric family.
            </div>
          </div>
          <div className="flex items-center gap-2">
            <HelpPopover entry={helpIndex.get('benchmark.metric_explorer')} />
            <select
              className="rounded-full border border-[color:var(--line)] bg-[var(--surface)] px-4 py-2 text-sm text-[var(--text)]"
              value={selectedMetric?.id || ''}
              onChange={(event) => onMetricChange(event.target.value)}
            >
              {metricOptions.map((option: ChartMetricOption) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <ComparisonChart
          title={selectedMetric?.label || 'Metric comparison'}
          subtitle="Current metric spotlight across the visible recipe pool."
          helpEntry={selectedMetric?.id ? helpIndex.get(`benchmark.metric.${selectedMetric.id}`) : helpIndex.get('benchmark.metric_explorer')}
          mode={chartMode}
          data={buildRecipeComparisonData(
            recipes,
            activeSuite,
            currentRun,
            previewCells,
            selectedMetric?.id || '',
            focusedRecipeIds,
            highlightedRecipeId,
            selectedRecipeId,
          )}
          onRecipeSelect={onRecipeSelect}
        />

        <MetricProfileChart
          title="Per-metric Profile"
          helpEntry={helpIndex.get('benchmark.metric_profile')}
          mode={chartMode}
          profileData={metricProfileData}
          legendEntries={legendEntries}
          onRecipeSelect={onRecipeSelect}
        />
      </div>
    </section>
  );
}
