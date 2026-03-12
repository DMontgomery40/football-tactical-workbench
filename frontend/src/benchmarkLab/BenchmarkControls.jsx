import { SectionTitleWithHelp } from '../helpUi';

const PANEL_STYLE = {
  border: '1px solid var(--line)',
  background: 'color-mix(in srgb, var(--surface) 92%, var(--surface-muted))',
};

function statusStyle(ready) {
  if (ready) {
    return {
      borderColor: 'color-mix(in srgb, var(--good-line) 80%, var(--line))',
      background: 'color-mix(in srgb, var(--good-bg) 84%, var(--surface))',
      color: 'var(--good-text)',
    };
  }
  return {
    borderColor: 'color-mix(in srgb, var(--warn-line) 80%, var(--line))',
    background: 'color-mix(in srgb, var(--warn-bg) 82%, var(--surface))',
    color: 'var(--warn-text)',
  };
}

function ReadinessChip({ label, ready }) {
  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em]"
      style={statusStyle(ready)}
    >
      <span className="h-2 w-2 rounded-full bg-current" />
      <span>{label}</span>
      <span className="opacity-80">{ready ? 'ready' : 'setup'}</span>
    </div>
  );
}

export default function BenchmarkControls({
  clipReady,
  candidates,
  coreBaselinePlan,
  selectedCandidateId,
  isRunning,
  benchmarkError,
  onRunBenchmark,
  helpIndex,
}) {
  const runnableCandidates = candidates.filter((candidate) => candidate?.available !== false);
  const runnableBaselines = (coreBaselinePlan?.candidates || []).filter((candidate) => candidate?.available !== false);
  const unavailableBaselines = coreBaselinePlan?.unavailableCandidates || [];
  const selectedCandidate = candidates.find((candidate) => candidate.id === selectedCandidateId) || null;
  const canRunAny = clipReady && runnableCandidates.length > 0 && !isRunning;
  const canRunBaselines = clipReady && Boolean(coreBaselinePlan?.ready) && !isRunning;
  const canRunSelected = Boolean(selectedCandidate && selectedCandidate.available !== false && canRunAny);

  let runAnyReason = '';
  if (!clipReady) runAnyReason = 'Prepare the benchmark clip first.';
  else if (runnableCandidates.length === 0) runAnyReason = 'No runnable benchmark candidates are available yet.';

  let runBaselinesReason = '';
  if (!clipReady) runBaselinesReason = 'Prepare the benchmark clip first.';
  else if ((coreBaselinePlan?.missingIds || []).length > 0) runBaselinesReason = 'The full baseline trio is still loading.';
  else if (unavailableBaselines.length > 0) runBaselinesReason = `Set up the missing baselines first: ${unavailableBaselines.map((candidate) => candidate.label || candidate.id).join(', ')}.`;

  return (
    <div className="card grid gap-5 rounded-[26px] p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionTitleWithHelp title="Run Desk" entry={helpIndex?.get('benchmark.runtime_profile')} />
        <span
          className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]"
          style={statusStyle(Boolean(coreBaselinePlan?.ready))}
        >
          {coreBaselinePlan?.ready ? 'Baseline trio ready' : 'Setup still needed'}
        </span>
      </div>

      <div className="grid gap-4 rounded-[22px] p-4" style={PANEL_STYLE}>
        <div className="grid gap-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Canonical run</div>
          <div className="flex flex-wrap gap-2">
            {(coreBaselinePlan?.candidates || []).map((candidate) => (
              <ReadinessChip key={candidate.id} label={candidate.label || candidate.id} ready={candidate.available !== false} />
            ))}
          </div>
        </div>
        <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">
          Run the baseline trio together when you want the cleanest apples-to-apples board. Use advanced runs only when you are intentionally bringing staged detector checkpoints into the same review loop.
        </p>
        <button
          type="button"
          className="workspace-primary-action"
          disabled={!canRunBaselines}
          onClick={() => onRunBenchmark(runnableBaselines.map((candidate) => candidate.id))}
          title={runBaselinesReason || undefined}
        >
          {isRunning ? 'Running baseline trio...' : 'Run baseline trio'}
        </button>

        {!isRunning && runBaselinesReason ? (
          <div className="rounded-[18px] p-4 text-sm leading-6" style={PANEL_STYLE}>
            {runBaselinesReason}
          </div>
        ) : null}
      </div>

      <details className="rounded-[22px] border p-4" style={PANEL_STYLE}>
        <summary className="cursor-pointer text-sm font-semibold text-[color:var(--text-strong)]">Advanced runs</summary>
        <div className="mt-4 grid gap-3">
          <div className="text-sm leading-6 text-[color:var(--text-muted)]">
            Use these only when you want to add extra staged checkpoints to the board or isolate one candidate for debugging.
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <button
              type="button"
              className="compact-button"
              disabled={!canRunAny}
              onClick={() => onRunBenchmark()}
              title={runAnyReason || undefined}
            >
              {isRunning ? 'Running...' : 'Run all available'}
            </button>

            <button
              type="button"
              className="compact-button"
              disabled={!canRunSelected}
              onClick={() => onRunBenchmark([selectedCandidateId])}
              title={selectedCandidate ? undefined : 'Select a candidate first.'}
            >
              {isRunning ? 'Running...' : selectedCandidate ? `Run ${selectedCandidate.label || selectedCandidate.id}` : 'Run selected only'}
            </button>
          </div>
          {!isRunning && !runBaselinesReason && runAnyReason ? (
            <div className="rounded-[18px] p-4 text-sm leading-6" style={PANEL_STYLE}>
              {runAnyReason}
            </div>
          ) : null}
          {unavailableBaselines.length > 0 ? (
            <div className="text-sm leading-6 text-[color:var(--text-muted)]">
              Setup still needed for {unavailableBaselines.map((candidate) => candidate.label || candidate.id).join(', ')}.
            </div>
          ) : null}
        </div>
      </details>

      {benchmarkError ? <div className="error-box">{benchmarkError}</div> : null}
    </div>
  );
}
