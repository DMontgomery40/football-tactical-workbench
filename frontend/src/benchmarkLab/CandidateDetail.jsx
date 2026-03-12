import { MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';

function formatScore(value) {
  if (value == null) return '--';
  if (typeof value === 'number') return value.toFixed(1);
  return String(value);
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
      <div className="card benchmark-detail-empty">
        <p className="muted">Select a candidate to view details.</p>
      </div>
    );
  }

  // benchmarkResult is a leaderboard row with scores at top level
  const result = benchmarkResult;
  const hasScores = result && result.composite != null;
  const summary = candidateRunResult?.summary_excerpt || null;
  const diagnostics = summary?.diagnostics?.length ? summary.diagnostics : summary?.heuristic_diagnostics || [];
  const overlayVideo = summary?.overlay_video ? `${apiBase}${summary.overlay_video}` : '';
  const candidateLogs = Array.isArray(candidateRunResult?.logs) ? candidateRunResult.logs : [];

  return (
    <div className="card benchmark-detail">
      <div className="benchmark-detail-header">
        <SectionTitleWithHelp title={candidate.label || candidate.id} entry={helpIndex?.get('benchmark.composite_score')} />
        <span className="benchmark-candidate-source muted">{candidate.source || 'unknown'}</span>
        {candidate.pipeline_override ? (
          <span className="benchmark-candidate-pipeline muted">{candidate.pipeline_override}</span>
        ) : null}
      </div>

      {hasScores ? (
        <div className="benchmark-detail-metrics">
          <div className="benchmark-detail-metric-grid">
            <div className="benchmark-metric-item">
              <span className="micro-label">Proxy score</span>
              <span className="benchmark-metric-value benchmark-metric-primary">{formatScore(result.composite)}</span>
            </div>
            <div className="benchmark-metric-item">
              <span className="micro-label">Track Stability (30%)</span>
              <span className="benchmark-metric-value">{formatScore(result.track_stability)}</span>
            </div>
            <div className="benchmark-metric-item">
              <span className="micro-label">Calibration (25%)</span>
              <span className="benchmark-metric-value">{formatScore(result.calibration)}</span>
            </div>
            <div className="benchmark-metric-item">
              <span className="micro-label">Coverage (25%)</span>
              <span className="benchmark-metric-value">{formatScore(result.coverage)}</span>
            </div>
            <div className="benchmark-metric-item">
              <span className="micro-label">Throughput (20%)</span>
              <span className="benchmark-metric-value">{formatScore(result.throughput)}</span>
            </div>
          </div>
          {result.pipeline ? (
            <div className="benchmark-detail-pipeline-note muted">
              Pipeline: <strong>{result.pipeline}</strong>
            </div>
          ) : null}
        </div>
      ) : result?.status === 'failed' ? (
        <div className="error-box">
          Benchmark failed: {result.error || 'Unknown error'}
        </div>
      ) : (
        <p className="muted">No benchmark results for this candidate yet.</p>
      )}

      {overlayVideo ? (
        <div className="benchmark-detail-video">
          <video controls preload="metadata" src={overlayVideo} className="benchmark-detail-video-el" />
        </div>
      ) : null}

      {summary?.diagnostics_summary_line ? (
        <div className="benchmark-detail-diagnostics">
          <MicroLabelWithHelp label="Diagnostics summary" entry={helpIndex?.get('benchmark.leaderboard')} />
          <div className="benchmark-detail-diagnostics-line">{summary.diagnostics_summary_line}</div>
        </div>
      ) : null}

      {diagnostics.length ? (
        <div className="benchmark-detail-diagnostics-grid">
          {diagnostics.slice(0, 4).map((item, index) => (
            <div key={`${item.title || 'diag'}-${index}`} className={`benchmark-diagnostic-card ${item.level || 'neutral'}`}>
              <div className="row-between">
                <div className="diagnostic-title">{item.title || 'Diagnostic'}</div>
                <div className="benchmark-status-badge muted">{item.level || 'info'}</div>
              </div>
              {item.message ? <div className="stat-hint">{item.message}</div> : null}
              {item.next_step ? <div className="diagnostic-next">Next: {item.next_step}</div> : null}
            </div>
          ))}
        </div>
      ) : null}

      {candidateLogs.length ? (
        <div className="benchmark-detail-logs">
          <MicroLabelWithHelp label="Candidate logs" entry={helpIndex?.get('benchmark.candidate')} />
          <pre className="benchmark-detail-log-pre">{candidateLogs.slice(-18).join('\n')}</pre>
        </div>
      ) : null}

      <div className="benchmark-detail-meta">
        <span className="micro-label">Weights</span>
        <span className="muted">{candidate.path || 'N/A'}</span>
      </div>
    </div>
  );
}
