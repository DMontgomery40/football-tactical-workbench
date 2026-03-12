import { SectionTitleWithHelp } from '../helpUi';

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
    <div className="card benchmark-controls-card">
      <SectionTitleWithHelp title="Run benchmark" entry={helpIndex?.get('benchmark.runtime_profile')} />

      <p className="muted benchmark-controls-explainer">
        Run the three baseline pipelines on the same clip, or add extra classic detector checkpoints for side-by-side review.
        The ranking is still a proxy comparison over runtime evidence, not a label-backed accuracy score.
      </p>

      {(coreBaselinePlan?.candidates || []).length ? (
        <div className="benchmark-runtime-summary">
          <div className="micro-label">Standard trio</div>
          <div className="benchmark-runtime-grid">
            {(coreBaselinePlan.candidates || []).map((candidate) => (
              <span key={candidate.id} className="muted">
                {candidate.label || candidate.id}: <strong>{candidate.available === false ? 'setup needed' : 'ready'}</strong>
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="benchmark-run-actions">
        <button
          type="button"
          className="workspace-primary-action"
          disabled={!canRunBaselines}
          onClick={() => onRunBenchmark(runnableBaselines.map((candidate) => candidate.id))}
          title={runBaselinesReason || undefined}
        >
          {isRunning ? 'Running benchmark...' : 'Run baseline trio'}
        </button>

        <button
          type="button"
          className="compact-button"
          disabled={!canRunAny}
          onClick={() => onRunBenchmark()}
          title={runAnyReason || undefined}
        >
          {isRunning ? 'Running...' : 'Run all available'}
        </button>

        {selectedCandidateId ? (
          <button
            type="button"
            className="compact-button"
            disabled={!canRunSelected}
            onClick={() => onRunBenchmark([selectedCandidateId])}
          >
            {isRunning ? 'Running...' : 'Run selected only'}
          </button>
        ) : null}
      </div>

      {!isRunning && (runBaselinesReason || runAnyReason) ? (
        <div className="benchmark-controls-hint muted">
          {runBaselinesReason || runAnyReason}
        </div>
      ) : null}

      {benchmarkError ? <div className="error-box">{benchmarkError}</div> : null}
    </div>
  );
}
