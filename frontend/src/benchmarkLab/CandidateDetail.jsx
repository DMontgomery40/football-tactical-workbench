import { MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';

const PANEL_STYLE = {
  border: '1px solid var(--line)',
  background: 'color-mix(in srgb, var(--surface) 92%, var(--surface-muted))',
};

function formatScore(value) {
  if (value == null) return '--';
  if (typeof value === 'number') return value.toFixed(1);
  return String(value);
}

function statusStyle(status, candidate) {
  if (status === 'failed') {
    return {
      borderColor: 'color-mix(in srgb, #ca5b4c 46%, var(--line))',
      background: 'color-mix(in srgb, #f3d7d3 72%, var(--surface))',
      color: '#7c3428',
    };
  }
  if (candidate?.available === false) {
    return {
      borderColor: 'color-mix(in srgb, var(--warn-line) 80%, var(--line))',
      background: 'color-mix(in srgb, var(--warn-bg) 84%, var(--surface))',
      color: 'var(--warn-text)',
    };
  }
  if (status === 'completed') {
    return {
      borderColor: 'color-mix(in srgb, var(--good-line) 80%, var(--line))',
      background: 'color-mix(in srgb, var(--good-bg) 84%, var(--surface))',
      color: 'var(--good-text)',
    };
  }
  return {
    borderColor: 'color-mix(in srgb, var(--accent) 24%, var(--line))',
    background: 'color-mix(in srgb, var(--accent-soft) 40%, var(--surface))',
    color: 'var(--accent-strong)',
  };
}

function MetricCard({ label, value, emphasis = false }) {
  return (
    <div className="grid gap-1 rounded-[18px] border p-4" style={PANEL_STYLE}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">{label}</div>
      <div className={`${emphasis ? 'text-3xl' : 'text-lg'} font-semibold text-[color:var(--text-strong)]`}>{formatScore(value)}</div>
    </div>
  );
}

function FactPill({ label, value }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm" style={PANEL_STYLE}>
      <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[color:var(--text-muted)]">{label}</span>
      <span className="font-semibold text-[color:var(--text-strong)]">{formatScore(value)}</span>
    </div>
  );
}

export default function CandidateDetail({
  apiBase,
  candidate,
  benchmarkResult,
  candidateRunResult,
  helpIndex,
}) {
  if (!candidate) {
    return (
      <div className="card grid gap-4 rounded-[26px] p-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Review desk</div>
        <div className="rounded-[22px] p-5" style={PANEL_STYLE}>
          <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">
            Select a baseline or detector candidate to turn this panel into the active review canvas.
          </p>
        </div>
      </div>
    );
  }

  const result = benchmarkResult;
  const summary = candidateRunResult?.summary_excerpt || null;
  const diagnostics = summary?.diagnostics?.length ? summary.diagnostics : summary?.heuristic_diagnostics || [];
  const overlayVideo = summary?.overlay_video ? `${apiBase}${summary.overlay_video}` : '';
  const candidateLogs = Array.isArray(candidateRunResult?.logs) ? candidateRunResult.logs : [];
  const hasResultMetrics = Boolean(
    result
    && (result.composite != null
      || result.track_stability != null
      || result.calibration != null
      || result.coverage != null
      || result.throughput != null
      || result.score_note),
  );
  const artifactLabel = candidate.pipeline_override === 'sn_gamestate'
    ? 'Repository'
    : candidate.pipeline_override === 'soccermaster'
      ? 'Model bundle'
      : 'Weights';
  const factRows = [
    { label: 'Pipeline', value: result?.pipeline || candidate.pipeline_override || 'classic' },
    { label: 'Frames processed', value: summary?.frames_processed },
    { label: 'Runtime FPS', value: summary?.fps },
    { label: 'Unique tracks', value: summary?.unique_player_track_ids },
    { label: 'Field registered ratio', value: summary?.field_registered_ratio },
  ].filter((item) => item.value != null && item.value !== '');

  return (
    <div className="card grid gap-5 rounded-[28px] p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="grid gap-2">
          <SectionTitleWithHelp title={candidate.label || candidate.id} entry={helpIndex?.get('benchmark.composite_score')} />
          <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            <span>{candidate.source || 'unknown'}</span>
            {candidate.pipeline_override ? <span>{candidate.pipeline_override}</span> : null}
          </div>
        </div>
        <span
          className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]"
          style={statusStyle(result?.status, candidate)}
        >
          {result?.status === 'failed'
            ? 'Failed'
            : result?.status === 'completed'
              ? 'Review ready'
              : candidate.available === false
                ? 'Setup required'
                : 'Awaiting run'}
        </span>
      </div>

      <div className="grid gap-4 rounded-[24px] p-4" style={PANEL_STYLE}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Overlay review</div>
          {result?.score_note ? <div className="text-sm text-[color:var(--text-muted)]">{result.score_note}</div> : null}
        </div>
        {overlayVideo ? (
          <video controls preload="metadata" src={overlayVideo} className="h-auto min-h-[22rem] w-full rounded-[18px] border border-[color:var(--line)] bg-black object-contain" />
        ) : (
          <div className="grid min-h-[22rem] place-items-center rounded-[18px] border border-dashed border-[color:var(--line)] px-6 text-center text-sm leading-6 text-[color:var(--text-muted)]">
            {candidate.available === false
              ? candidate.availability_note || 'This candidate is not configured on this machine yet.'
              : 'No overlay yet. Run this candidate to turn the review desk into a video-first evidence panel.'}
          </div>
        )}

        {factRows.length ? (
          <div className="flex flex-wrap gap-2">
            {factRows.map((fact) => (
              <FactPill key={fact.label} label={fact.label} value={fact.value} />
            ))}
          </div>
        ) : null}
      </div>

      {result?.status === 'failed' ? (
        <div className="error-box">Benchmark failed: {result.error || 'Unknown error'}</div>
      ) : null}

      {hasResultMetrics ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <MetricCard label="Proxy score" value={result.composite} emphasis />
          <MetricCard label="Track stability" value={result.track_stability} />
          <MetricCard label="Calibration" value={result.calibration} />
          <MetricCard label="Coverage" value={result.coverage} />
          <MetricCard label="Throughput" value={result.throughput} />
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(16rem,18rem)]">
        <div className="grid gap-4">
          {summary?.diagnostics_summary_line ? (
            <div className="grid gap-2 rounded-[22px] p-4" style={PANEL_STYLE}>
              <MicroLabelWithHelp label="Diagnostics summary" entry={helpIndex?.get('benchmark.leaderboard')} />
              <div className="text-sm leading-6 text-[color:var(--text-muted)]">{summary.diagnostics_summary_line}</div>
            </div>
          ) : null}

          {diagnostics.length ? (
            <div className="grid gap-3">
              {diagnostics.slice(0, 3).map((item, index) => (
                <div key={`${item.title || 'diag'}-${index}`} className="grid gap-3 rounded-[20px] border p-4" style={PANEL_STYLE}>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="text-sm font-semibold text-[color:var(--text-strong)]">{item.title || 'Diagnostic'}</div>
                    <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">{item.level || 'info'}</span>
                  </div>
                  {item.message ? <div className="text-sm leading-6 text-[color:var(--text-muted)]">{item.message}</div> : null}
                  {item.next_step ? <div className="text-sm leading-6 text-[color:var(--text-strong)]">Next: {item.next_step}</div> : null}
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="grid gap-4">
          <div className="grid gap-2 rounded-[22px] p-4" style={PANEL_STYLE}>
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">{artifactLabel}</div>
            <div className="break-all text-sm leading-6 text-[color:var(--text-muted)]">{candidate.path || 'N/A'}</div>
          </div>

          {candidateLogs.length ? (
            <details className="rounded-[22px] border p-4" style={PANEL_STYLE}>
              <summary className="cursor-pointer text-sm font-semibold text-[color:var(--text-strong)]">Candidate logs</summary>
              <pre className="m-0 mt-4 max-h-[260px] overflow-auto whitespace-pre-wrap rounded-[18px] border p-4 text-xs leading-6" style={PANEL_STYLE}>
                {candidateLogs.slice(-24).join('\n')}
              </pre>
            </details>
          ) : null}
        </div>
      </div>
    </div>
  );
}
