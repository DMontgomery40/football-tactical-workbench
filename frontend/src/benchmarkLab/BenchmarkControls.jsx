import { SectionTitleWithHelp } from '../helpUi';

export default function BenchmarkControls({
  clipReady,
  candidates,
  selectedCandidateId,
  isRunning,
  benchmarkError,
  onRunBenchmark,
  helpIndex,
}) {
  const hasAnyCandidates = candidates.length > 0;
  const canRun = clipReady && hasAnyCandidates && !isRunning;

  let disabledReason = '';
  if (!clipReady) disabledReason = 'Prepare the benchmark clip first.';
  else if (!hasAnyCandidates) disabledReason = 'Import at least one candidate detector.';

  return (
    <div className="card benchmark-controls-card">
      <SectionTitleWithHelp title="Run benchmark" entry={helpIndex?.get('benchmark.runtime_profile')} />

      <p className="muted benchmark-controls-explainer">
        Each candidate detector runs the same clip under identical conditions. The current ranking is a
        proxy comparison over runtime evidence, not a label-backed accuracy score.
      </p>

      <div className="benchmark-run-actions">
        <button
          type="button"
          className="workspace-primary-action"
          disabled={!canRun}
          onClick={() => onRunBenchmark()}
          title={disabledReason || undefined}
        >
          {isRunning ? 'Running benchmark...' : 'Run all candidates'}
        </button>

        {selectedCandidateId && canRun ? (
          <button
            type="button"
            className="compact-button"
            disabled={isRunning}
            onClick={() => onRunBenchmark([selectedCandidateId])}
          >
            {isRunning ? 'Running...' : 'Run selected only'}
          </button>
        ) : null}
      </div>

      {disabledReason && !isRunning ? (
        <div className="benchmark-controls-hint muted">{disabledReason}</div>
      ) : null}

      {benchmarkError ? <div className="error-box">{benchmarkError}</div> : null}
    </div>
  );
}
