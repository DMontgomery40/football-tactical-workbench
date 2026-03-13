import { HelpPopover } from '../helpUi';
import type { BenchmarkRecipe, BenchmarkRunDetail, BenchmarkSuite } from './types';
import {
  METRIC_HELP_IDS,
  type PreviewCell,
  SUITE_HELP_IDS,
  metricDisplay,
  metricLabel,
  resolveMatrixCellState,
  statusLabel,
} from './utils';
import { cx, displayTitleClass, panelClass, sectionHeadingClass, statusTone } from './ui';

interface SelectedCell {
  suiteId: string;
  recipeId: string;
}

interface ResultsMatrixProps {
  activeSuite: BenchmarkSuite | null;
  recipes: BenchmarkRecipe[];
  currentRun: BenchmarkRunDetail | null;
  selectedCell: SelectedCell | null;
  previewCells: Record<string, PreviewCell>;
  helpIndex: Map<string, unknown>;
  onSelectCell: (selection: SelectedCell) => void;
}

export default function ResultsMatrix({
  activeSuite,
  recipes,
  currentRun,
  selectedCell,
  previewCells,
  helpIndex,
  onSelectCell,
}: ResultsMatrixProps) {
  if (!activeSuite) {
    return (
      <section className={cx(panelClass, 'space-y-3')}>
        <div className={sectionHeadingClass}>Results Matrix</div>
        <div className={displayTitleClass}>Choose at least one suite</div>
        <p className="text-sm leading-6 text-[var(--text-muted)]">
          The matrix becomes sortable once there is an active suite in view.
        </p>
      </section>
    );
  }

  if (recipes.length === 0) {
    return (
      <section className={cx(panelClass, 'space-y-3')}>
        <div className={sectionHeadingClass}>Results Matrix</div>
        <div className={displayTitleClass}>No recipes match the current view</div>
        <p className="text-sm leading-6 text-[var(--text-muted)]">
          Relax the shared filters above or widen the chart legend spotlight to repopulate the matrix.
        </p>
      </section>
    );
  }

  const metricColumns = activeSuite.metricColumns.length > 0
    ? activeSuite.metricColumns
    : [activeSuite.primaryMetric].filter(Boolean);
  const suiteHelpEntry = helpIndex.get(SUITE_HELP_IDS[activeSuite.id] || 'benchmark.matrix');

  return (
    <section className={cx(panelClass, 'min-w-0 overflow-x-hidden space-y-5')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Results Matrix</div>
          <div className={displayTitleClass}>{activeSuite.label}</div>
          <p className="max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
            The matrix stays center-stage: one active suite, real metric columns, and every blocked or unavailable cell
            left visible instead of disappearing behind aggregate scoring.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <HelpPopover entry={suiteHelpEntry} />
          <HelpPopover entry={helpIndex.get('benchmark.results_matrix')} />
        </div>
      </div>

      <div className="w-full max-w-full overflow-x-auto rounded-[1.3rem] border border-[color:var(--line)] bg-[var(--surface)]/70 [contain:layout_paint]">
        <table className="min-w-full border-collapse text-left">
          <thead className="bg-[color:var(--surface-muted)]/70">
            <tr>
              <th className="sticky left-0 z-10 border-b border-[color:var(--line)] bg-[color:var(--surface-muted)]/90 px-4 py-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                Recipe
              </th>
              <th className="border-b border-[color:var(--line)] px-4 py-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                Status
              </th>
              {metricColumns.map((metricId) => (
                <th
                  key={metricId}
                  className="border-b border-[color:var(--line)] px-4 py-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]"
                >
                  <div className="flex items-center gap-2">
                    <span>{metricLabel(metricId)}</span>
                    <HelpPopover entry={helpIndex.get(METRIC_HELP_IDS[metricId] || 'benchmark.metric.na')} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recipes.map((recipe) => {
              const cellState = resolveMatrixCellState(currentRun, activeSuite, recipe, previewCells);
              const selected = selectedCell?.suiteId === activeSuite.id && selectedCell?.recipeId === recipe.id;
              return (
                <tr
                  key={recipe.id}
                  className={cx(
                    'cursor-pointer border-b border-[color:var(--line)] last:border-b-0 transition hover:bg-[color:var(--accent-soft)]/20',
                    selected && 'bg-[color:var(--accent-soft)]/35',
                  )}
                  onClick={() => onSelectCell({ suiteId: activeSuite.id, recipeId: recipe.id })}
                >
                  <th className="sticky left-0 z-10 w-[18rem] border-r border-[color:var(--line)] bg-[var(--surface)]/95 px-4 py-4 align-top">
                    <div className="space-y-1">
                      <div className="text-sm font-semibold text-[var(--text-strong)]">{recipe.label}</div>
                      <div className="text-xs uppercase tracking-[0.12em] text-[var(--text-muted)]">
                        {recipe.kind.replace(/_/g, ' ')} · {recipe.pipeline || 'classic'}
                      </div>
                      <div className="text-xs text-[var(--text-muted)]">{recipe.id}</div>
                    </div>
                  </th>
                  <td className="px-4 py-4 align-top">
                    <div className="space-y-2">
                      <span className={cx('inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]', statusTone(cellState.status))}>
                        {statusLabel(cellState.status)}
                      </span>
                      <div className="text-sm leading-5 text-[var(--text-muted)]">{cellState.note}</div>
                    </div>
                  </td>
                  {metricColumns.map((metricId) => {
                    const metric = cellState.result?.metrics?.[metricId] || null;
                    return (
                      <td key={metricId} className="px-4 py-4 align-top">
                        <div className="text-base font-semibold text-[var(--text-strong)]">
                          {metricDisplay(metric)}
                        </div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          {metric?.label || (activeSuite.primaryMetric === metricId ? 'Primary metric' : 'Metric pending')}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
