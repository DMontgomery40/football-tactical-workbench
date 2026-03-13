import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { HelpPopover } from '../helpUi';
import type { BenchmarkDatasetState, BenchmarkRecipe, BenchmarkRunDetail, BenchmarkSuite } from './types';
import {
  buildSuiteEvaluationRows,
  suiteEvaluationSeries,
  type PreviewCell,
  type SuiteEvaluationRow,
} from './utils';
import { cx, displayTitleClass, insetPanelClass, panelClass, sectionHeadingClass } from './ui';

type ChartMode = 'line' | 'bar';

const SUITE_SERIES_COLORS = ['#0f766e', '#1f5f92', '#d97706', '#b91c1c', '#6d28d9', '#047857'];

function suiteSeriesColor(index: number): string {
  return SUITE_SERIES_COLORS[index % SUITE_SERIES_COLORS.length];
}

function SuiteLegend({
  series,
  hidden,
  activeSeries,
  onToggle,
  onHover,
}: {
  series: Array<{ id: string; label: string }>;
  hidden: Set<string>;
  activeSeries: string | null;
  onToggle: (seriesId: string) => void;
  onHover: (seriesId: string | null) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {series.map((entry, index) => {
        const isHidden = hidden.has(entry.id);
        const highlighted = !activeSeries || activeSeries === entry.id;
        return (
          <button
            key={entry.id}
            className={cx(
              'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold transition',
              isHidden
                ? 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text-muted)]'
                : 'border-transparent bg-[color:var(--accent-soft)]/55 text-[var(--text-strong)]',
              !highlighted && 'opacity-45',
            )}
            type="button"
            onClick={() => onToggle(entry.id)}
            onMouseEnter={() => onHover(entry.id)}
            onMouseLeave={() => onHover(null)}
          >
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: suiteSeriesColor(index) }} />
            {entry.label}
          </button>
        );
      })}
    </div>
  );
}

function SuiteChartCard({
  title,
  subtitle,
  rows,
  series,
  chartMode,
  hidden,
  activeSeries,
  onToggleSeries,
  onHoverSeries,
  onSelectSuite,
  helpEntry,
}: {
  title: string;
  subtitle: string;
  rows: SuiteEvaluationRow[];
  series: Array<{ id: string; label: string }>;
  chartMode: ChartMode;
  hidden: Set<string>;
  activeSeries: string | null;
  onToggleSeries: (seriesId: string) => void;
  onHoverSeries: (seriesId: string | null) => void;
  onSelectSuite: (suiteId: string) => void;
  helpEntry: unknown;
}) {
  const visibleSeries = series.filter((entry) => !hidden.has(entry.id));
  const handleChartClick = (state: any) => {
    const suiteId = state?.activePayload?.[0]?.payload?.suiteId;
    if (suiteId) {
      onSelectSuite(suiteId);
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

      <SuiteLegend
        series={series}
        hidden={hidden}
        activeSeries={activeSeries}
        onToggle={onToggleSeries}
        onHover={onHoverSeries}
      />

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          {chartMode === 'line' ? (
            <LineChart data={rows} margin={{ top: 12, right: 18, left: 0, bottom: 0 }} onClick={handleChartClick}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
              <XAxis dataKey="suiteLabel" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              <Legend verticalAlign="top" height={0} />
              {visibleSeries.map((entry, index) => (
                <Line
                  key={entry.id}
                  type="monotone"
                  dataKey={entry.id}
                  name={entry.label}
                  stroke={suiteSeriesColor(index)}
                  strokeWidth={activeSeries && activeSeries !== entry.id ? 2 : 3}
                  dot={{ r: 3 }}
                  activeDot={{ r: 6 }}
                  opacity={activeSeries && activeSeries !== entry.id ? 0.25 : 1}
                />
              ))}
            </LineChart>
          ) : (
            <BarChart data={rows} margin={{ top: 12, right: 18, left: 0, bottom: 0 }} onClick={handleChartClick}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
              <XAxis dataKey="suiteLabel" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              <Legend verticalAlign="top" height={0} />
              {visibleSeries.map((entry, index) => (
                <Bar
                  key={entry.id}
                  dataKey={entry.id}
                  name={entry.label}
                  fill={suiteSeriesColor(index)}
                  opacity={activeSeries && activeSeries !== entry.id ? 0.25 : 0.92}
                  radius={[8, 8, 0, 0]}
                />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface SuiteEvaluationChartsProps {
  suites: BenchmarkSuite[];
  recipes: BenchmarkRecipe[];
  currentRun: BenchmarkRunDetail | null;
  datasetStateById: Map<string, BenchmarkDatasetState>;
  previewCells: Record<string, PreviewCell>;
  activeSuiteId: string | null;
  helpIndex: Map<string, unknown>;
  onActiveSuiteChange: (suiteId: string) => void;
}

export default function SuiteEvaluationCharts({
  suites,
  recipes,
  currentRun,
  datasetStateById,
  previewCells,
  activeSuiteId,
  helpIndex,
  onActiveSuiteChange,
}: SuiteEvaluationChartsProps) {
  const [chartMode, setChartMode] = useState<ChartMode>('bar');
  const [activeSeries, setActiveSeries] = useState<string | null>(null);
  const [hiddenSeries, setHiddenSeries] = useState<Record<string, string[]>>({
    performance: [],
    operations: [],
    agreement: [],
  });

  const rows = useMemo(
    () => buildSuiteEvaluationRows(suites, recipes, currentRun, datasetStateById, previewCells),
    [currentRun, datasetStateById, previewCells, recipes, suites],
  );
  const seriesGroups = suiteEvaluationSeries();

  if (suites.length === 0) {
    return (
      <section className={cx(panelClass, 'space-y-3')}>
        <div className={sectionHeadingClass}>Suite Evaluation Charts</div>
        <div className={displayTitleClass}>Select a benchmark run first</div>
        <p className="text-sm leading-6 text-[var(--text-muted)]">
          Suite comparison appears once the matrix has suites and recipes in view.
        </p>
      </section>
    );
  }

  return (
    <section className={cx(panelClass, 'space-y-6')}>
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Suite Evaluation Charts</div>
          <div className={displayTitleClass}>Judge the benchmark suites too</div>
          <p className="max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
            This section compares suite signal, burden, coverage, and agreement across the shared recipe pool so the
            operator can inspect the benchmark datasets themselves, not only the model rows. Active suite: {activeSuiteId || 'none'}.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <HelpPopover entry={helpIndex.get('benchmark.suite_evaluation')} />
          <button
            className={cx(
              'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em]',
              chartMode === 'line'
                ? 'border-[color:var(--accent)] bg-[color:var(--accent-soft)] text-[var(--accent-strong)]'
                : 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text-muted)]',
            )}
            type="button"
            onClick={() => setChartMode('line')}
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
            onClick={() => setChartMode('bar')}
          >
            Bar
          </button>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <SuiteChartCard
          title="Signal"
          subtitle="Primary spread, suite average, and dispersion across comparable recipes."
          rows={rows}
          series={seriesGroups.performance}
          chartMode={chartMode}
          hidden={new Set(hiddenSeries.performance)}
          activeSeries={activeSeries}
          onToggleSeries={(seriesId) => setHiddenSeries((current) => ({
            ...current,
            performance: current.performance.includes(seriesId)
              ? current.performance.filter((value) => value !== seriesId)
              : [...current.performance, seriesId],
          }))}
          onHoverSeries={setActiveSeries}
          onSelectSuite={onActiveSuiteChange}
          helpEntry={helpIndex.get('benchmark.metric.discriminative_power')}
        />

        <SuiteChartCard
          title="Cost + Coverage"
          subtitle="Runtime burden, blocked or unavailable rates, sample size, and materialization readiness."
          rows={rows}
          series={seriesGroups.operations}
          chartMode={chartMode}
          hidden={new Set(hiddenSeries.operations)}
          activeSeries={activeSeries}
          onToggleSeries={(seriesId) => setHiddenSeries((current) => ({
            ...current,
            operations: current.operations.includes(seriesId)
              ? current.operations.filter((value) => value !== seriesId)
              : [...current.operations, seriesId],
          }))}
          onHoverSeries={setActiveSeries}
          onSelectSuite={onActiveSuiteChange}
          helpEntry={helpIndex.get('benchmark.metric.runtime_burden')}
        />

        <SuiteChartCard
          title="Agreement"
          subtitle="Comparable-recipe coverage and average rank correlation between suites on shared results."
          rows={rows}
          series={seriesGroups.agreement}
          chartMode={chartMode}
          hidden={new Set(hiddenSeries.agreement)}
          activeSeries={activeSeries}
          onToggleSeries={(seriesId) => setHiddenSeries((current) => ({
            ...current,
            agreement: current.agreement.includes(seriesId)
              ? current.agreement.filter((value) => value !== seriesId)
              : [...current.agreement, seriesId],
          }))}
          onHoverSeries={setActiveSeries}
          onSelectSuite={onActiveSuiteChange}
          helpEntry={helpIndex.get('benchmark.metric.rank_correlation')}
        />
      </div>
    </section>
  );
}
