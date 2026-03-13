import { HelpPopover } from '../helpUi';
import type {
  BenchmarkFilterState,
  BenchmarkSortState,
  BenchmarkSuite,
  BenchmarkViewPreset,
} from './types';
import { isLatencyMetric, metricLabel, type MatrixFilterOptions, statusLabel } from './utils';
import {
  cx,
  displayTitleClass,
  panelClass,
  primaryButtonClass,
  secondaryButtonClass,
  sectionHeadingClass,
  textInputClass,
} from './ui';

interface MatrixToolbarProps {
  suites: BenchmarkSuite[];
  activeSuite: BenchmarkSuite | null;
  availableTiers: string[];
  visibleRecipeCount?: number;
  totalRecipeCount?: number;
  viewPreset: BenchmarkViewPreset;
  viewPresets: BenchmarkViewPreset[];
  filterState: BenchmarkFilterState;
  filterOptions: MatrixFilterOptions;
  sortState: BenchmarkSortState;
  helpIndex: Map<string, unknown>;
  onActiveSuiteChange: (suiteId: string) => void;
  onFilterChange: (next: BenchmarkFilterState) => void;
  onSortChange: (next: BenchmarkSortState) => void;
  onViewPresetChange: (preset: BenchmarkViewPreset) => void;
  onResetFilters: () => void;
  onResetSort: () => void;
}

export default function MatrixToolbar({
  suites,
  activeSuite,
  availableTiers,
  visibleRecipeCount,
  totalRecipeCount,
  viewPreset,
  viewPresets,
  filterState,
  filterOptions,
  sortState,
  helpIndex,
  onActiveSuiteChange,
  onFilterChange,
  onSortChange,
  onViewPresetChange,
  onResetFilters,
  onResetSort,
}: MatrixToolbarProps) {
  const sortOptions = activeSuite
    ? [
      { id: 'label', label: 'Recipe label' },
      { id: 'status', label: 'Status' },
      { id: 'provider', label: 'Provider' },
      { id: 'architecture', label: 'Architecture' },
      { id: 'bundleMode', label: 'Bundle mode' },
      ...activeSuite.metricColumns.map((metricId) => ({ id: metricId, label: metricLabel(metricId) })),
    ]
    : [{ id: 'label', label: 'Recipe label' }];

  return (
    <section className={cx(panelClass, 'space-y-5')}>
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Matrix Filters</div>
          <div className={displayTitleClass}>Keep one honest suite in focus</div>
          <p className="max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
            The filters, matrix, charts, and detail panel all consume the same active suite and visible recipe pool.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <HelpPopover entry={helpIndex.get('benchmark.matrix_filters')} />
          {typeof visibleRecipeCount === 'number' && typeof totalRecipeCount === 'number' ? (
            <div className="rounded-full border border-[color:var(--line)] bg-[var(--surface)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
              {visibleRecipeCount} / {totalRecipeCount} visible
            </div>
          ) : null}
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">View preset</span>
          <select
            className={textInputClass}
            value={viewPreset}
            onChange={(event) => onViewPresetChange(event.target.value as BenchmarkViewPreset)}
          >
            {viewPresets.map((preset) => (
              <option key={preset} value={preset}>
                {preset}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Active suite</span>
          <select
            className={textInputClass}
            value={activeSuite?.id || ''}
            onChange={(event) => onActiveSuiteChange(event.target.value)}
          >
            {suites.map((suite) => (
              <option key={suite.id} value={suite.id}>
                {suite.label}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Tier</span>
          <select
            className={textInputClass}
            value={filterState.tier}
            onChange={(event) => onFilterChange({ ...filterState, tier: event.target.value })}
          >
            <option value="all">All tiers</option>
            {availableTiers.map((tier) => (
              <option key={tier} value={tier}>
                {tier}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Search recipes</span>
          <input
            className={textInputClass}
            type="text"
            value={filterState.search}
            placeholder="Search label, provider, architecture"
            onChange={(event) => onFilterChange({ ...filterState, search: event.target.value })}
          />
        </label>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Provider</span>
          <select
            className={textInputClass}
            value={filterState.provider}
            onChange={(event) => onFilterChange({ ...filterState, provider: event.target.value })}
          >
            <option value="all">All providers</option>
            {filterOptions.providers.map((provider) => (
              <option key={provider} value={provider}>
                {provider}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Architecture</span>
          <select
            className={textInputClass}
            value={filterState.architecture}
            onChange={(event) => onFilterChange({ ...filterState, architecture: event.target.value })}
          >
            <option value="all">All architectures</option>
            {filterOptions.architectures.map((architecture) => (
              <option key={architecture} value={architecture}>
                {architecture}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Bundle mode</span>
          <select
            className={textInputClass}
            value={filterState.bundleMode}
            onChange={(event) => onFilterChange({ ...filterState, bundleMode: event.target.value })}
          >
            <option value="all">All bundle modes</option>
            {filterOptions.bundleModes.map((bundleMode) => (
              <option key={bundleMode} value={bundleMode}>
                {bundleMode}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Capability</span>
          <select
            className={textInputClass}
            value={filterState.capability}
            onChange={(event) => onFilterChange({ ...filterState, capability: event.target.value })}
          >
            <option value="all">All capabilities</option>
            {filterOptions.capabilities.map((capability) => (
              <option key={capability} value={capability}>
                {capability}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Status</span>
          <select
            className={textInputClass}
            value={filterState.status}
            onChange={(event) => onFilterChange({ ...filterState, status: event.target.value })}
          >
            <option value="all">All statuses</option>
            {filterOptions.statuses.map((status) => (
              <option key={status} value={status}>
                {statusLabel(status)}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Supports active suite</span>
          <select
            className={textInputClass}
            value={filterState.supportsActiveSuite}
            onChange={(event) => onFilterChange({
              ...filterState,
              supportsActiveSuite: event.target.value as BenchmarkFilterState['supportsActiveSuite'],
            })}
          >
            <option value="all">All</option>
            <option value="supported">Supported</option>
            <option value="unsupported">Unsupported</option>
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Has N/A metric</span>
          <select
            className={textInputClass}
            value={filterState.hasNa}
            onChange={(event) => onFilterChange({
              ...filterState,
              hasNa: event.target.value as BenchmarkFilterState['hasNa'],
            })}
          >
            <option value="all">All</option>
            <option value="has_na">Has N/A</option>
            <option value="no_na">No N/A</option>
          </select>
        </label>

        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Sort column</span>
          <select
            className={textInputClass}
            value={sortState.column || activeSuite?.primaryMetric || 'label'}
            onChange={(event) => {
              const nextColumn = event.target.value;
              onSortChange({
                column: nextColumn,
                direction: isLatencyMetric(nextColumn) ? 'asc' : 'desc',
              });
            }}
          >
            {sortOptions.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button className={secondaryButtonClass} type="button" onClick={onResetFilters}>
          Reset filters
        </button>
        <button className={secondaryButtonClass} type="button" onClick={onResetSort}>
          Reset sort
        </button>
        <div className="rounded-full border border-[color:var(--line)] bg-[var(--surface)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
          Sort: {sortOptions.find((option) => option.id === (sortState.column || activeSuite?.primaryMetric || 'label'))?.label || 'Recipe label'}
        </div>
        <div className="rounded-full border border-[color:var(--line)] bg-[var(--surface)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
          Direction: {sortState.direction || 'desc'}
        </div>
        <button className={primaryButtonClass} type="button" onClick={() => onActiveSuiteChange(activeSuite?.id || suites[0]?.id || '')}>
          Refresh suite focus
        </button>
      </div>
    </section>
  );
}
