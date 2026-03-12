import { useState } from 'react';

import { SectionTitleWithHelp } from '../helpUi';
import { splitBenchmarkCandidates } from './candidates';

const PANEL_STYLE = {
  border: '1px solid var(--line)',
  background: 'color-mix(in srgb, var(--surface) 92%, var(--surface-muted))',
};

function statusStyle(available) {
  if (available) {
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

function candidateButtonStyle(selected, available, variant = 'baseline') {
  if (selected) {
    return {
      border: '1px solid color-mix(in srgb, var(--accent) 34%, var(--line))',
      background: 'linear-gradient(180deg, color-mix(in srgb, var(--accent-soft) 52%, var(--surface)) 0%, color-mix(in srgb, var(--accent-soft) 22%, var(--surface)) 100%)',
      color: 'var(--accent-strong)',
      boxShadow: '0 18px 38px color-mix(in srgb, var(--accent-soft) 22%, transparent)',
    };
  }
  if (!available) {
    return {
      border: '1px solid color-mix(in srgb, var(--warn-line) 54%, var(--line))',
      background: 'color-mix(in srgb, var(--warn-bg) 58%, var(--surface))',
      color: 'var(--text)',
    };
  }
  if (variant === 'detector') {
    return {
      border: '1px solid var(--line)',
      background: 'color-mix(in srgb, var(--surface) 96%, var(--surface-muted))',
      color: 'var(--text)',
    };
  }
  return {
    border: '1px solid var(--line)',
    background: 'color-mix(in srgb, var(--surface) 94%, var(--surface-muted))',
    color: 'var(--text)',
  };
}

function CandidateButton({ candidate, selected, onSelect, variant = 'baseline' }) {
  const available = candidate.available !== false;

  return (
    <button
      type="button"
      className={`grid gap-3 text-left transition ${variant === 'baseline' ? 'rounded-[22px] p-5' : 'rounded-[18px] p-4'}`}
      style={candidateButtonStyle(selected, available, variant)}
      onClick={() => onSelect(candidate.id)}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1">
          <div className="text-base font-semibold">{candidate.label || candidate.id}</div>
          <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            <span>{candidate.source || 'unknown'}</span>
            {candidate.pipeline_override ? <span>{candidate.pipeline_override}</span> : null}
          </div>
        </div>
        <span
          className="inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]"
          style={statusStyle(available)}
        >
          {available ? 'Ready' : 'Setup needed'}
        </span>
      </div>
      {candidate.availability_note ? (
        <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">{candidate.availability_note}</p>
      ) : null}
    </button>
  );
}

export default function CandidateLibrary({
  candidates,
  selectedCandidateId,
  onSelectCandidate,
  onImportLocal,
  onImportHf,
  isImporting,
  candidatesError,
  helpIndex,
}) {
  const [importMode, setImportMode] = useState(null);
  const [localPath, setLocalPath] = useState('');
  const [hfRepoId, setHfRepoId] = useState('');
  const [hfFilename, setHfFilename] = useState('');
  const { baselineCandidates, detectorCandidates } = splitBenchmarkCandidates(candidates);

  function handleImportLocal(event) {
    event.preventDefault();
    const trimmed = localPath.trim();
    if (!trimmed) return;
    onImportLocal(trimmed);
    setLocalPath('');
    setImportMode(null);
  }

  function handleImportHf(event) {
    event.preventDefault();
    const repo = hfRepoId.trim();
    if (!repo) return;
    onImportHf(repo, hfFilename.trim() || undefined);
    setHfRepoId('');
    setHfFilename('');
    setImportMode(null);
  }

  return (
    <div className="card grid gap-5 rounded-[26px] p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionTitleWithHelp title="Candidate Board" entry={helpIndex?.get('benchmark.candidate')} />
        <span className="text-sm text-[color:var(--text-muted)]">
          {baselineCandidates.length} baselines · {detectorCandidates.length} extra checkpoints
        </span>
      </div>

      {candidatesError ? <div className="error-box">{candidatesError}</div> : null}

      {baselineCandidates.length > 0 ? (
        <div className="grid gap-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Core baselines</div>
          <div className="grid gap-3 xl:grid-cols-3">
            {baselineCandidates.map((candidate) => (
              <CandidateButton
                key={candidate.id}
                candidate={candidate}
                selected={candidate.id === selectedCandidateId}
                onSelect={onSelectCandidate}
                variant="baseline"
              />
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Additional detector checkpoints</div>
          <span className="text-sm text-[color:var(--text-muted)]">{detectorCandidates.length} staged</span>
        </div>

        {detectorCandidates.length > 0 ? (
          <div className="grid max-h-[18rem] gap-3 overflow-auto pr-1">
            {detectorCandidates.map((candidate) => (
              <CandidateButton
                key={candidate.id}
                candidate={candidate}
                selected={candidate.id === selectedCandidateId}
                onSelect={onSelectCandidate}
                variant="detector"
              />
            ))}
          </div>
        ) : (
          <div className="rounded-[22px] p-4 text-sm leading-6 text-[color:var(--text-muted)]" style={PANEL_STYLE}>
            No extra detector checkpoints are staged yet. Use Training Studio outputs or import a local / HuggingFace checkpoint here.
          </div>
        )}
      </div>

      <details className="rounded-[22px] border p-4" style={PANEL_STYLE}>
        <summary className="cursor-pointer text-sm font-semibold text-[color:var(--text-strong)]">Stage extra detector checkpoint</summary>
        <div className="mt-4 grid gap-3">
          <div className="flex flex-wrap gap-2">
            <button type="button" className="compact-button" onClick={() => setImportMode('local')}>
              Import local weights
            </button>
            <button type="button" className="compact-button" onClick={() => setImportMode('hf')}>
              Import from HuggingFace
            </button>
          </div>

          {importMode === 'local' ? (
            <form className="grid gap-3 rounded-[18px] border p-4" style={PANEL_STYLE} onSubmit={handleImportLocal}>
              <label className="grid gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Local weights path</span>
                <input
                  type="text"
                  value={localPath}
                  onChange={(event) => setLocalPath(event.target.value)}
                  placeholder="/path/to/weights/best.pt"
                  autoFocus
                />
              </label>
              <div className="flex flex-wrap gap-3">
                <button type="submit" className="compact-button" disabled={isImporting || !localPath.trim()}>
                  {isImporting ? 'Importing...' : 'Stage checkpoint'}
                </button>
                <button type="button" className="compact-button" onClick={() => setImportMode(null)}>
                  Cancel
                </button>
              </div>
            </form>
          ) : null}

          {importMode === 'hf' ? (
            <form className="grid gap-3 rounded-[18px] border p-4" style={PANEL_STYLE} onSubmit={handleImportHf}>
              <label className="grid gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">HuggingFace repo ID</span>
                <input
                  type="text"
                  value={hfRepoId}
                  onChange={(event) => setHfRepoId(event.target.value)}
                  placeholder="org/model-name"
                  autoFocus
                />
              </label>
              <label className="grid gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Filename (optional)</span>
                <input
                  type="text"
                  value={hfFilename}
                  onChange={(event) => setHfFilename(event.target.value)}
                  placeholder="best.pt"
                />
              </label>
              <div className="flex flex-wrap gap-3">
                <button type="submit" className="compact-button" disabled={isImporting || !hfRepoId.trim()}>
                  {isImporting ? 'Importing...' : 'Stage checkpoint'}
                </button>
                <button type="button" className="compact-button" onClick={() => setImportMode(null)}>
                  Cancel
                </button>
              </div>
            </form>
          ) : null}
        </div>
      </details>
    </div>
  );
}
