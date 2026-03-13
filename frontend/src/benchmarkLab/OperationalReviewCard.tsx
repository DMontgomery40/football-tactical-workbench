import { HelpPopover } from '../helpUi';
import { formatPathTail } from '../trainingStudio/formatters.js';
import type { BenchmarkRecipe, BenchmarkRunDetail, BenchmarkRunResult, BenchmarkSuite, ClipStatus } from './types';
import { cx, displayTitleClass, insetPanelClass, panelClass, sectionHeadingClass, statusTone } from './ui';

interface OperationalReviewCardProps {
  apiBase: string;
  suite: BenchmarkSuite | null;
  recipe: BenchmarkRecipe | null;
  result: BenchmarkRunResult | null;
  benchmark: BenchmarkRunDetail | null;
  clipStatus: ClipStatus | null;
  helpIndex: Map<string, unknown>;
}

function resolveVideoSource(apiBase: string, rawPath: unknown): string | null {
  const value = typeof rawPath === 'string' ? rawPath.trim() : '';
  if (!value) {
    return null;
  }
  if (value.startsWith('http://') || value.startsWith('https://')) {
    return value;
  }
  if (value.startsWith('/')) {
    return `${apiBase}${value}`;
  }
  return null;
}

export default function OperationalReviewCard({
  apiBase,
  suite,
  recipe,
  result,
  benchmark,
  clipStatus,
  helpIndex,
}: OperationalReviewCardProps) {
  const isOperational = suite?.protocol === 'operational';
  const overlaySource = resolveVideoSource(apiBase, result?.artifacts?.overlay_video);
  const runDirectory = typeof result?.artifacts?.run_dir === 'string' ? result.artifacts.run_dir : '';
  const summaryPath = typeof result?.artifacts?.summary_json === 'string' ? result.artifacts.summary_json : '';
  const provenancePath = typeof result?.artifacts?.provenance_json === 'string' ? result.artifacts.provenance_json : '';

  return (
    <section className={cx(panelClass, 'space-y-5')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Operational Review</div>
          <div className={displayTitleClass}>Overlay-first benchmark evidence</div>
          <p className="max-w-3xl text-sm leading-6 text-[var(--text-muted)]">
            The operational suite is where the benchmark becomes visually inspectable: same clip, same runtime profile,
            different recipe bindings, with the overlay video treated as the primary artifact.
          </p>
        </div>
        <HelpPopover entry={helpIndex.get('benchmark.operational_review')} />
      </div>

      {!isOperational ? (
        <div className={insetPanelClass}>
          <div className="text-sm text-[var(--text)]">
            Choose <strong>Operational Clip Review</strong> in the suite selector and then pick an operational cell in
            the matrix to populate this card.
          </div>
        </div>
      ) : null}

      {isOperational ? (
        <>
          <div className="grid gap-3 md:grid-cols-3">
            <div className={insetPanelClass}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Clip status</div>
              <div className="mt-2 text-base font-semibold text-[var(--text-strong)]">
                {clipStatus?.ready ? 'Cached and ready' : 'Needs canonical clip'}
              </div>
              <div className="mt-1 text-sm text-[var(--text-muted)]">
                {clipStatus?.path ? formatPathTail(clipStatus.path) : clipStatus?.note || 'No clip loaded yet.'}
              </div>
            </div>
            <div className={insetPanelClass}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Recipe</div>
              <div className="mt-2 text-base font-semibold text-[var(--text-strong)]">{recipe?.label || 'No recipe selected'}</div>
              <div className="mt-1 text-sm text-[var(--text-muted)]">{recipe?.pipeline || 'classic'} pipeline</div>
            </div>
            <div className={insetPanelClass}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Run state</div>
              <div className="mt-2">
                <span className={cx('rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]', statusTone(result?.status || benchmark?.status || 'pending'))}>
                  {result?.status || benchmark?.status || 'Pending'}
                </span>
              </div>
              <div className="mt-2 text-sm text-[var(--text-muted)]">
                {typeof result?.rawResult?.run_id === 'string' ? result.rawResult.run_id : benchmark?.benchmarkId || 'No operational run yet'}
              </div>
            </div>
          </div>

          {overlaySource ? (
            <div className="overflow-hidden rounded-[1.5rem] border border-[color:var(--line)] bg-[var(--surface)]">
              <video
                className="aspect-video w-full bg-black object-contain"
                controls
                preload="metadata"
                src={overlaySource}
              />
            </div>
          ) : (
            <div className={cx(insetPanelClass, 'space-y-2')}>
              <div className="text-sm font-semibold text-[var(--text-strong)]">Overlay not available yet</div>
              <div className="text-sm leading-6 text-[var(--text-muted)]">
                Run the operational suite with a compatible recipe to materialize the overlay video under the benchmarked
                analysis run.
              </div>
            </div>
          )}

          <div className="grid gap-3 xl:grid-cols-3">
            <div className={insetPanelClass}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Run directory</div>
              <div className="mt-2 break-all text-sm text-[var(--text)]">{runDirectory || 'Not created yet'}</div>
            </div>
            <div className={insetPanelClass}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Summary excerpt</div>
              <div className="mt-2 break-all text-sm text-[var(--text)]">{summaryPath || 'Not written yet'}</div>
            </div>
            <div className={insetPanelClass}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Provenance</div>
              <div className="mt-2 break-all text-sm text-[var(--text)]">{provenancePath || 'Not written yet'}</div>
            </div>
          </div>
        </>
      ) : null}
    </section>
  );
}
