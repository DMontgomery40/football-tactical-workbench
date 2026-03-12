import { useState } from 'react';

import { SectionTitleWithHelp } from '../helpUi';
import { splitBenchmarkCandidates } from './candidates';

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
  const [importMode, setImportMode] = useState(null); // null | 'local' | 'hf'
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

  function renderCandidateRow(candidate) {
    const available = candidate.available !== false;
    return (
      <li key={candidate.id} className="benchmark-candidate-row">
        <button
          type="button"
          className={`benchmark-candidate-btn${candidate.id === selectedCandidateId ? ' selected' : ''}${available ? '' : ' opacity-70'}`}
          onClick={() => onSelectCandidate(candidate.id)}
        >
          <span className="benchmark-candidate-name">{candidate.label || candidate.id}</span>
          <span className="benchmark-candidate-source muted">{candidate.source || 'unknown'}</span>
          {candidate.pipeline_override ? (
            <span className="benchmark-candidate-pipeline muted">{candidate.pipeline_override}</span>
          ) : null}
          <span className={`benchmark-status-badge ${available ? 'muted' : 'failed'}`}>
            {available ? 'ready' : 'setup required'}
          </span>
          {candidate.availability_note ? (
            <span className="muted text-left text-sm">{candidate.availability_note}</span>
          ) : null}
        </button>
      </li>
    );
  }

  return (
    <div className="card benchmark-candidates">
      <SectionTitleWithHelp title="Baselines and detector candidates" entry={helpIndex?.get('benchmark.candidate')} />

      {candidates.length === 0 && !candidatesError ? (
        <p className="muted">No benchmark candidates found yet.</p>
      ) : null}

      {candidatesError ? <div className="error-box">{candidatesError}</div> : null}

      {baselineCandidates.length > 0 ? (
        <>
          <div className="micro-label">Core baselines</div>
          <ul className="benchmark-candidate-list">
            {baselineCandidates.map(renderCandidateRow)}
          </ul>
        </>
      ) : null}

      {detectorCandidates.length > 0 ? (
        <>
          <div className="micro-label">Additional detector checkpoints</div>
          <ul className="benchmark-candidate-list">
            {detectorCandidates.map(renderCandidateRow)}
          </ul>
        </>
      ) : null}

      <div className="benchmark-import-controls">
        {importMode === null ? (
          <div className="benchmark-import-buttons">
            <button type="button" className="compact-button" onClick={() => setImportMode('local')}>
              Import local weights
            </button>
            <button type="button" className="compact-button" onClick={() => setImportMode('hf')}>
              Import from HuggingFace
            </button>
          </div>
        ) : null}

        {importMode === 'local' ? (
          <form className="benchmark-import-form" onSubmit={handleImportLocal}>
            <label>
              <span className="micro-label">Local weights path</span>
              <input
                type="text"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
                placeholder="/path/to/weights/best.pt"
                autoFocus
              />
            </label>
            <div className="benchmark-import-form-actions">
              <button type="submit" className="compact-button" disabled={isImporting || !localPath.trim()}>
                {isImporting ? 'Importing...' : 'Import'}
              </button>
              <button type="button" className="compact-button muted-button" onClick={() => setImportMode(null)}>
                Cancel
              </button>
            </div>
          </form>
        ) : null}

        {importMode === 'hf' ? (
          <form className="benchmark-import-form" onSubmit={handleImportHf}>
            <label>
              <span className="micro-label">HuggingFace repo ID</span>
              <input
                type="text"
                value={hfRepoId}
                onChange={(e) => setHfRepoId(e.target.value)}
                placeholder="org/model-name"
                autoFocus
              />
            </label>
            <label>
              <span className="micro-label">Filename (optional)</span>
              <input
                type="text"
                value={hfFilename}
                onChange={(e) => setHfFilename(e.target.value)}
                placeholder="best.pt"
              />
            </label>
            <div className="benchmark-import-form-actions">
              <button type="submit" className="compact-button" disabled={isImporting || !hfRepoId.trim()}>
                {isImporting ? 'Importing...' : 'Import'}
              </button>
              <button type="button" className="compact-button muted-button" onClick={() => setImportMode(null)}>
                Cancel
              </button>
            </div>
          </form>
        ) : null}
      </div>
    </div>
  );
}
