import { useMemo, useState } from 'react';

import { SectionTitleWithHelp } from '../helpUi';

const SORT_FIELDS = [
  { id: 'rank', label: 'Rank' },
  { id: 'composite', label: 'Proxy score' },
  { id: 'track_stability', label: 'Track stability' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'coverage', label: 'Coverage' },
  { id: 'throughput', label: 'Throughput' },
];

const PANEL_STYLE = {
  border: '1px solid var(--line)',
  background: 'color-mix(in srgb, var(--surface) 92%, var(--surface-muted))',
};

function formatScore(value) {
  if (value == null) return '--';
  if (typeof value === 'number') return value.toFixed(1);
  return String(value);
}

function statusStyle(status, row) {
  if (status === 'failed') {
    return {
      borderColor: 'color-mix(in srgb, #ca5b4c 46%, var(--line))',
      background: 'color-mix(in srgb, #f3d7d3 72%, var(--surface))',
      color: '#7c3428',
    };
  }
  if (status === 'completed' && row?.composite == null && row?.score_kind === 'partial_proxy') {
    return {
      borderColor: 'color-mix(in srgb, var(--warn-line) 82%, var(--line))',
      background: 'color-mix(in srgb, var(--warn-bg) 84%, var(--surface))',
      color: 'var(--warn-text)',
    };
  }
  if (status === 'completed') {
    return {
      borderColor: 'color-mix(in srgb, var(--good-line) 82%, var(--line))',
      background: 'color-mix(in srgb, var(--good-bg) 84%, var(--surface))',
      color: 'var(--good-text)',
    };
  }
  return {
    borderColor: 'color-mix(in srgb, var(--accent) 28%, var(--line))',
    background: 'color-mix(in srgb, var(--accent-soft) 44%, var(--surface))',
    color: 'var(--accent-strong)',
  };
}

function SortButton({ active, label, onClick }) {
  return (
    <button
      type="button"
      className={`rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] ${active ? 'text-[color:var(--accent-strong)]' : 'text-[color:var(--text-muted)]'}`}
      style={active ? {
        border: '1px solid color-mix(in srgb, var(--accent) 34%, var(--line))',
        background: 'color-mix(in srgb, var(--accent-soft) 44%, var(--surface))',
      } : PANEL_STYLE}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function InlineMetric({ label, value }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm" style={PANEL_STYLE}>
      <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[color:var(--text-muted)]">{label}</span>
      <span className="font-semibold text-[color:var(--text-strong)]">{formatScore(value)}</span>
    </div>
  );
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
    return benchmarks.find((benchmark) => benchmark.benchmark_id === selectedBenchmarkId) || null;
  }, [benchmarks, selectedBenchmarkId]);

  const leaderboard = useMemo(() => {
    const rows = Array.isArray(activeBenchmark?.leaderboard) ? [...activeBenchmark.leaderboard] : [];
    if (sortField === 'rank') {
      rows.sort((left, right) => {
        const leftRank = left.rank ?? Number.MAX_SAFE_INTEGER;
        const rightRank = right.rank ?? Number.MAX_SAFE_INTEGER;
        return sortAsc ? leftRank - rightRank : rightRank - leftRank;
      });
      return rows;
    }

    rows.sort((left, right) => {
      const leftMissing = left[sortField] == null;
      const rightMissing = right[sortField] == null;
      if (leftMissing && rightMissing) return 0;
      if (leftMissing) return 1;
      if (rightMissing) return -1;
      return sortAsc ? left[sortField] - right[sortField] : right[sortField] - left[sortField];
    });
    return rows;
  }, [activeBenchmark, sortField, sortAsc]);

  const isRunning = activeBenchmark?.status === 'running' || activeBenchmark?.status === 'queued';
  const candidateCount = activeBenchmark?.candidate_count || 0;
  const completedCount = leaderboard.filter((row) => row.status === 'completed' || row.status === 'failed').length;
  const hasPartialProxyRows = leaderboard.some((row) => row.score_kind === 'partial_proxy');

  function handleSort(field) {
    if (field === sortField) {
      setSortAsc(!sortAsc);
      return;
    }
    setSortField(field);
    setSortAsc(field === 'rank');
  }

  return (
    <div className="card grid gap-5 rounded-[26px] p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="grid gap-2">
          <SectionTitleWithHelp title="Leaderboard" entry={helpIndex?.get('benchmark.leaderboard')} />
          <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">
            Read this board as triage, not as ground truth. Close scores or unscored rows should send you straight back to the overlay and diagnostics.
          </p>
        </div>
        <span
          className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]"
          style={PANEL_STYLE}
        >
          Proxy comparison only
        </span>
      </div>

      {benchmarks.length > 1 ? (
        <label className="grid gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Benchmark run</span>
          <select value={selectedBenchmarkId} onChange={(event) => onSelectBenchmark(event.target.value)}>
            {benchmarks.map((benchmark) => (
              <option key={benchmark.benchmark_id} value={benchmark.benchmark_id}>
                {benchmark.benchmark_id} ({benchmark.status})
              </option>
            ))}
          </select>
        </label>
      ) : null}

      {isRunning ? (
        <div className="grid gap-3 rounded-[22px] p-4" style={PANEL_STYLE}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm font-semibold text-[color:var(--text-strong)]">
              Benchmark in progress
            </div>
            <div className="text-sm text-[color:var(--text-muted)]">
              {completedCount} of {candidateCount} candidates evaluated
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[color:var(--surface-soft)]">
            <div
              className="h-full rounded-full"
              style={{
                width: `${candidateCount > 0 ? Math.round((completedCount / candidateCount) * 100) : 0}%`,
                background: 'linear-gradient(90deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 72%, #87c1ef) 100%)',
              }}
            />
          </div>
          <div className="text-sm leading-6 text-[color:var(--text-muted)]">
            {leaderboard.length > 0
              ? 'Partial rows refresh automatically while the benchmark is still running.'
              : 'The first row can take a while to land, especially when SoccerMaster is still warming up.'}
          </div>
        </div>
      ) : null}

      {!activeBenchmark ? (
        <div className="rounded-[22px] p-5 text-sm leading-6 text-[color:var(--text-muted)]" style={PANEL_STYLE}>
          No benchmark results yet. Start a run from the Setup tab.
        </div>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            {SORT_FIELDS.map((field) => (
              <SortButton
                key={field.id}
                active={field.id === sortField}
                label={`${field.label}${field.id === sortField ? (sortAsc ? ' ▲' : ' ▼') : ''}`}
                onClick={() => handleSort(field.id)}
              />
            ))}
          </div>

          {leaderboard.length === 0 && !isRunning ? (
            <div className="rounded-[22px] p-5 text-sm leading-6 text-[color:var(--text-muted)]" style={PANEL_STYLE}>
              No results are available for this benchmark run yet.
            </div>
          ) : (
            <div className="grid gap-3">
              {leaderboard.map((row) => (
                <button
                  key={row.candidate_id}
                  type="button"
                  className="grid gap-4 rounded-[22px] p-5 text-left transition"
                  style={{
                    ...PANEL_STYLE,
                    ...(row.status === 'failed'
                      ? { background: 'color-mix(in srgb, var(--warn-bg) 56%, var(--surface))' }
                      : row.rank === 1
                        ? { boxShadow: '0 20px 38px color-mix(in srgb, var(--accent-soft) 20%, transparent)' }
                        : null),
                  }}
                  onClick={() => onSelectCandidate?.(row.candidate_id)}
                >
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="flex items-start gap-4">
                      <div className="grid h-11 w-11 place-items-center rounded-full border text-sm font-semibold text-[color:var(--text-strong)]" style={PANEL_STYLE}>
                        {row.rank ?? '--'}
                      </div>
                      <div className="grid gap-1">
                        <div className="text-base font-semibold text-[color:var(--text-strong)]">{row.label || row.candidate_id}</div>
                        <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
                          <span>{row.source || 'unknown'}</span>
                          <span>{row.pipeline || 'classic'}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-3">
                      <div className="grid gap-1 text-right">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Proxy score</div>
                        <div className="text-2xl font-semibold text-[color:var(--text-strong)]">{formatScore(row.composite)}</div>
                      </div>
                      <span
                        className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]"
                        style={statusStyle(row.status, row)}
                      >
                        {row.status === 'completed' && row.composite == null && row.score_kind === 'partial_proxy'
                          ? 'Unscored review'
                          : row.status}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <InlineMetric label="Track" value={row.track_stability} />
                    <InlineMetric label="Calib" value={row.calibration} />
                    <InlineMetric label="Coverage" value={row.coverage} />
                    <InlineMetric label="Throughput" value={row.throughput} />
                  </div>

                  {row.score_note ? (
                    <div className="text-sm leading-6 text-[color:var(--text-muted)]">{row.score_note}</div>
                  ) : null}
                </button>
              ))}
            </div>
          )}

          <div className="grid gap-2 rounded-[22px] p-4 text-sm leading-6 text-[color:var(--text-muted)]" style={PANEL_STYLE}>
            <div>Proxy score 0-100. Weights: track stability 30%, calibration 25%, coverage 25%, throughput 20%.</div>
            {hasPartialProxyRows ? (
              <div>External baselines can still finish on this board without a ranked proxy score if they do not emit the native runtime metrics the workbench uses for ranking.</div>
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
