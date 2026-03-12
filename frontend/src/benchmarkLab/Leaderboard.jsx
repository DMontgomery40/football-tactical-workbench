import { useMemo, useState } from 'react';

import { SectionTitleWithHelp } from '../helpUi';

const SORT_FIELDS = [
  { id: 'rank', label: 'Rank' },
  { id: 'composite', label: 'Composite' },
  { id: 'track_stability', label: 'Track Stability' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'coverage', label: 'Coverage' },
  { id: 'throughput', label: 'Throughput' },
];

function formatScore(value) {
  if (value == null) return '--';
  if (typeof value === 'number') return value.toFixed(1);
  return String(value);
}

function statusBadge(status) {
  if (status === 'completed') return null;
  if (status === 'failed') return <span className="benchmark-status-badge failed">failed</span>;
  if (status === 'running') return <span className="benchmark-status-badge running">running</span>;
  return <span className="benchmark-status-badge muted">{status}</span>;
}

export default function Leaderboard({
  benchmarks,
  selectedBenchmarkId,
  onSelectBenchmark,
  onSelectCandidate,
  helpIndex,
}) {
  const [sortField, setSortField] = useState('rank');
  const [sortAsc, setSortAsc] = useState(true);

  const activeBenchmark = useMemo(() => {
    if (!selectedBenchmarkId) return null;
    return benchmarks.find((b) => b.benchmark_id === selectedBenchmarkId) || null;
  }, [benchmarks, selectedBenchmarkId]);

  const leaderboard = useMemo(() => {
    const rows = Array.isArray(activeBenchmark?.leaderboard) ? [...activeBenchmark.leaderboard] : [];
    if (sortField === 'rank') {
      rows.sort((a, b) => (sortAsc ? a.rank - b.rank : b.rank - a.rank));
    } else {
      rows.sort((a, b) => {
        const va = a[sortField] ?? -Infinity;
        const vb = b[sortField] ?? -Infinity;
        return sortAsc ? va - vb : vb - va;
      });
    }
    return rows;
  }, [activeBenchmark, sortField, sortAsc]);

  const isRunning = activeBenchmark?.status === 'running' || activeBenchmark?.status === 'queued';
  const candidateCount = activeBenchmark?.candidate_count || 0;
  const completedCount = leaderboard.filter((r) => r.status === 'completed' || r.status === 'failed').length;

  function handleSort(field) {
    if (field === sortField) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === 'rank');
    }
  }

  return (
    <div className="card benchmark-leaderboard">
      <div className="benchmark-leaderboard-header">
        <SectionTitleWithHelp title="Leaderboard" entry={helpIndex?.get('benchmark.leaderboard')} />
        <span className="benchmark-honest-label muted">
          Heuristic operational benchmarking -- not ground-truth accuracy
        </span>
      </div>

      {benchmarks.length > 1 ? (
        <div className="benchmark-history-selector">
          <span className="micro-label">Benchmark run</span>
          <select
            value={selectedBenchmarkId}
            onChange={(e) => onSelectBenchmark(e.target.value)}
          >
            {benchmarks.map((b) => (
              <option key={b.benchmark_id} value={b.benchmark_id}>
                {b.benchmark_id} ({b.status})
              </option>
            ))}
          </select>
        </div>
      ) : null}

      {isRunning ? (
        <div className="benchmark-running-status">
          <div className="benchmark-running-header">
            <span className="benchmark-running-dot" />
            <span className="benchmark-running-text">
              Benchmark in progress — {completedCount} of {candidateCount} candidates evaluated
            </span>
          </div>
          <div className="benchmark-progress-bar-container">
            <div
              className="benchmark-progress-bar-fill"
              style={{ width: `${candidateCount > 0 ? Math.round((completedCount / candidateCount) * 100) : 0}%` }}
            />
          </div>
          {leaderboard.length > 0 ? (
            <div className="benchmark-partial-results muted">
              Partial results below (updates every 3s):
            </div>
          ) : (
            <p className="muted benchmark-running-hint">
              Loading first candidate... SoccerMaster&apos;s unified backbone takes ~60s to initialize.
            </p>
          )}
        </div>
      ) : null}

      {!activeBenchmark ? (
        <p className="muted">No benchmark results yet. Run a benchmark from the Setup tab.</p>
      ) : leaderboard.length === 0 && !isRunning ? (
        <p className="muted">No results available for this benchmark run.</p>
      ) : leaderboard.length > 0 ? (
        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                {SORT_FIELDS.map((field) => (
                  <th
                    key={field.id}
                    className={`benchmark-th${sortField === field.id ? ' sorted' : ''}`}
                    onClick={() => handleSort(field.id)}
                  >
                    {field.label}
                    {sortField === field.id ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}
                  </th>
                ))}
                <th className="benchmark-th">Detector</th>
                <th className="benchmark-th">Pipeline</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((row) => (
                <tr
                  key={row.candidate_id}
                  className={`benchmark-row${row.status === 'failed' ? ' benchmark-row-failed' : ''}`}
                  onClick={() => onSelectCandidate?.(row.candidate_id)}
                >
                  <td className="benchmark-rank-cell">
                    <span className={`benchmark-rank-badge rank-${Math.min(row.rank, 4)}`}>
                      {row.rank}
                    </span>
                  </td>
                  <td className="benchmark-score-cell">
                    {formatScore(row.composite)}
                    {statusBadge(row.status)}
                  </td>
                  <td>{formatScore(row.track_stability)}</td>
                  <td>{formatScore(row.calibration)}</td>
                  <td>{formatScore(row.coverage)}</td>
                  <td>{formatScore(row.throughput)}</td>
                  <td className="benchmark-detector-cell">{row.label || row.candidate_id}</td>
                  <td className="benchmark-pipeline-cell muted">{row.pipeline || 'classic'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {isRunning ? null : (
            <div className="benchmark-score-legend muted">
              Scores 0-100. Weights: track stability 30%, calibration 25%, coverage 25%, throughput 20%.
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
