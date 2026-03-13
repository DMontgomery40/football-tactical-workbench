import { HelpPopover } from '../helpUi';
import { formatTimestamp } from '../trainingStudio/formatters.js';
import type { BenchmarkHistoryItem, BenchmarkRunDetail } from './types';
import { cx } from './ui';
import {
  displayTitleClass,
  insetPanelClass,
  panelClass,
  primaryButtonClass,
  secondaryButtonClass,
  sectionHeadingClass,
  statusTone,
  textInputClass,
} from './ui';

interface RunControlsProps {
  benchmarkLabel: string;
  history: BenchmarkHistoryItem[];
  selectedBenchmarkId: string;
  currentRun: BenchmarkRunDetail | null;
  selectedRecipePipelines: string[];
  activePipeline: string;
  selectedSuiteCount: number;
  selectedRecipeCount: number;
  totalCells: number;
  runnableCells: number;
  blockedCells: number;
  unsupportedCells: number;
  missingDatasetCells: number;
  unavailableCells: number;
  isRunning: boolean;
  isRefreshing: boolean;
  error: string;
  operationMessage: string;
  helpIndex: Map<string, unknown>;
  onBenchmarkLabelChange: (value: string) => void;
  onSelectBenchmark: (benchmarkId: string) => void;
  onRun: () => Promise<void> | void;
  onRefresh: () => Promise<void> | void;
}

export default function RunControls({
  benchmarkLabel,
  history,
  selectedBenchmarkId,
  currentRun,
  selectedRecipePipelines,
  activePipeline,
  selectedSuiteCount,
  selectedRecipeCount,
  totalCells,
  runnableCells,
  blockedCells,
  unsupportedCells,
  missingDatasetCells,
  unavailableCells,
  isRunning,
  isRefreshing,
  error,
  operationMessage,
  helpIndex,
  onBenchmarkLabelChange,
  onSelectBenchmark,
  onRun,
  onRefresh,
}: RunControlsProps) {
  const pipelineMismatch = selectedRecipePipelines.length > 0 && selectedRecipePipelines.some((pipeline) => pipeline && pipeline !== activePipeline);
  const currentProgress = Math.max(0, Math.min(100, Math.round(currentRun?.progress || 0)));
  const runDisabled = isRunning || runnableCells === 0 || selectedSuiteCount === 0 || selectedRecipeCount === 0;

  return (
    <section className={cx(panelClass, 'space-y-5')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Run Controls</div>
          <div className={displayTitleClass}>Launch and inspect benchmark jobs</div>
          <p className="max-w-2xl text-sm leading-6 text-[var(--text-muted)]">
            The lab runs the full selected suite-by-recipe matrix. Unsupported or missing combinations stay visible so
            the review surface can explain exactly why a cell did not execute.
          </p>
        </div>
        <HelpPopover entry={helpIndex.get('benchmark.run_controls')} />
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <div className={insetPanelClass}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Selected suites</div>
          <div className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">{selectedSuiteCount}</div>
        </div>
        <div className={insetPanelClass}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Selected recipes</div>
          <div className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">{selectedRecipeCount}</div>
        </div>
        <div className={insetPanelClass}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Runnable cells now</div>
          <div className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">{runnableCells}</div>
          <div className="mt-1 text-xs text-[var(--text-muted)]">out of {totalCells} planned evaluations</div>
        </div>
        <div className={insetPanelClass}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Blocked cells now</div>
          <div className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">{blockedCells}</div>
          <div className="mt-1 text-xs text-[var(--text-muted)]">
            {unsupportedCells} unsupported · {missingDatasetCells} blocked or missing dataset · {unavailableCells} unavailable
          </div>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto_auto]">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
            Benchmark label
          </span>
          <input
            className={textInputClass}
            type="text"
            value={benchmarkLabel}
            placeholder="March detector bakeoff"
            onChange={(event) => onBenchmarkLabelChange(event.target.value)}
          />
        </label>
        <button className={primaryButtonClass} type="button" disabled={runDisabled} onClick={() => onRun()}>
          {isRunning ? 'Starting…' : 'Run selected matrix'}
        </button>
        <button className={secondaryButtonClass} type="button" disabled={isRefreshing} onClick={() => onRefresh()}>
          {isRefreshing ? 'Refreshing…' : 'Refresh snapshot'}
        </button>
      </div>

      {history.length > 0 ? (
        <div className={cx(insetPanelClass, 'space-y-3')}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className={sectionHeadingClass}>Run History</div>
              <div className="mt-1 text-sm text-[var(--text-muted)]">
                Select a benchmark to populate the matrix, detail panel, and operational review.
              </div>
            </div>
            {currentRun ? (
              <span className={cx('rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]', statusTone(currentRun.status))}>
                {currentRun.status}
              </span>
            ) : null}
          </div>
          <select
            className={textInputClass}
            value={selectedBenchmarkId}
            onChange={(event) => onSelectBenchmark(event.target.value)}
          >
            {history.map((item) => (
              <option key={item.benchmarkId} value={item.benchmarkId}>
                {item.label || item.benchmarkId} · {item.status} · {formatTimestamp(item.createdAt)}
              </option>
            ))}
          </select>
          {currentRun ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-[var(--text)]">{currentRun.label || currentRun.benchmarkId}</span>
                <span className="text-[var(--text-muted)]">{currentProgress}% complete</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-[color:var(--surface-soft)]">
                <div
                  className="h-full rounded-full bg-[var(--accent)] transition-[width]"
                  style={{ width: `${currentProgress}%` }}
                />
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {pipelineMismatch ? (
        <div className="rounded-2xl border border-[color:var(--accent)]/30 bg-[color:var(--accent-soft)]/40 px-4 py-3 text-sm text-[var(--accent-strong)]">
          Analysis Workspace is currently on <strong>{activePipeline}</strong>, while your selected benchmark recipes span{' '}
          <strong>{selectedRecipePipelines.join(', ')}</strong>. That is fine, but read the matrix as a benchmark-lab
          comparison, not a direct mirror of the live analysis workspace.
        </div>
      ) : null}

      {operationMessage ? (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/12 px-4 py-3 text-sm text-amber-700 dark:text-amber-300">
          {operationMessage}
        </div>
      ) : null}
      {error ? <div className="error-box">{error}</div> : null}
    </section>
  );
}
