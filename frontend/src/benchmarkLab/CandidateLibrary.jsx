import { useState } from 'react';

import { SectionTitleWithHelp } from '../helpUi';

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
    <div className="card benchmark-candidates">
      <SectionTitleWithHelp title="Candidate detectors" entry={helpIndex?.get('benchmark.candidate')} />

      {candidates.length === 0 && !candidatesError ? (
        <p className="muted">No candidates found. Import a detector or train one in Training Studio.</p>
      ) : null}

      {candidatesError ? <div className="error-box">{candidatesError}</div> : null}

      {candidates.length > 0 ? (
        <ul className="benchmark-candidate-list">
          {candidates.map((c) => (
            <li key={c.id} className="benchmark-candidate-row">
              <button
                type="button"
                className={`benchmark-candidate-btn${c.id === selectedCandidateId ? ' selected' : ''}`}
                onClick={() => onSelectCandidate(c.id)}
              >
                <span className="benchmark-candidate-name">{c.label || c.id}</span>
                <span className="benchmark-candidate-source muted">{c.source || 'unknown'}</span>
                {c.pipeline_override ? (
                  <span className="benchmark-candidate-pipeline muted">{c.pipeline_override}</span>
                ) : null}
              </button>
            </li>
          ))}
        </ul>
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
