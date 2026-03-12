import { useRef, useState } from 'react';

import { SectionTitleWithHelp } from '../helpUi';

const PANEL_STYLE = {
  border: '1px solid var(--line)',
  background: 'color-mix(in srgb, var(--surface) 92%, var(--surface-muted))',
};

function badgeStyle(ready) {
  if (ready) {
    return {
      borderColor: 'color-mix(in srgb, var(--good-line) 82%, var(--line))',
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

function FactChip({ children }) {
  return (
    <span
      className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium text-[color:var(--text-muted)]"
      style={PANEL_STYLE}
    >
      {children}
    </span>
  );
}

export default function ClipCard({
  clipStatus,
  clipSourcePath,
  onClipSourcePathChange,
  isEnsuring,
  clipError,
  onEnsureClip,
  onUploadClip,
  helpIndex,
}) {
  const ready = Boolean(clipStatus?.ready);
  const cachedPath = clipStatus?.path || '';
  const sizeMb = clipStatus?.size_mb;
  const duration = clipStatus?.duration;
  const hasTextPath = clipSourcePath.trim().length > 0;

  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);
  const hasAnyInput = hasTextPath || selectedFile != null;

  function handleSubmit(event) {
    event.preventDefault();
    if (isEnsuring || !hasAnyInput) return;

    if (selectedFile) {
      onUploadClip(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    onEnsureClip(clipSourcePath.trim());
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
  }

  function handleClearFile() {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  return (
    <div className="card grid gap-5 rounded-[26px] p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionTitleWithHelp title="Reference Clip" entry={helpIndex?.get('benchmark.clip')} />
        <span
          className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]"
          style={badgeStyle(ready)}
        >
          {ready ? 'Reference locked' : 'Clip needed'}
        </span>
      </div>

      <div className="grid gap-3 rounded-[22px] p-4" style={PANEL_STYLE}>
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Current benchmark clip</div>
        {ready ? (
          <>
            <div className="text-sm font-medium text-[color:var(--text-strong)] break-all">{cachedPath}</div>
            <div className="flex flex-wrap gap-2">
              {sizeMb != null ? <FactChip>{sizeMb.toFixed(1)} MB</FactChip> : null}
              {duration != null ? <FactChip>{duration.toFixed(1)}s</FactChip> : null}
              {clipStatus?.fps != null ? <FactChip>{Number(clipStatus.fps).toFixed(1)} fps</FactChip> : null}
              {clipStatus?.width && clipStatus?.height ? <FactChip>{clipStatus.width}×{clipStatus.height}</FactChip> : null}
            </div>
            <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">
              This clip is the canonical reference for every row on the board. Replace it only when you want to restart the comparison from a new camera context.
            </p>
          </>
        ) : (
          <>
            <div className="text-sm font-medium text-[color:var(--text-strong)]">No clip is locked yet.</div>
            <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">
              Load one short reference clip before benchmarking. Keep it tight enough to iterate quickly, but long enough to expose tracking churn, field registration, and visible ball phases.
            </p>
          </>
        )}
      </div>

      <form className="grid gap-4 rounded-[22px] p-4" style={PANEL_STYLE} onSubmit={handleSubmit}>
        <div className="grid gap-1">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Replace clip</div>
          <p className="m-0 text-sm leading-6 text-[color:var(--text-muted)]">
            Use a local path or upload a new file. Keep one clip locked while you compare the trio so every row stays on the same camera context.
          </p>
        </div>
        <label className="grid gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Local clip path</span>
          <input
            type="text"
            value={clipSourcePath}
            onChange={(event) => onClipSourcePathChange(event.target.value)}
            placeholder="/Users/you/path/to/clip.mp4"
            disabled={selectedFile != null}
          />
        </label>

        <div className="grid gap-3">
          <label className="grid gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-muted)]">Upload reference clip</span>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
            />
          </label>
          {selectedFile ? (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-[18px] border p-3" style={PANEL_STYLE}>
              <div className="text-sm text-[color:var(--text-muted)]">
                <strong className="text-[color:var(--text-strong)]">{selectedFile.name}</strong>
                {' '}({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)
              </div>
              <button type="button" className="compact-button" onClick={handleClearFile}>Clear file</button>
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            type="submit"
            className="workspace-primary-action"
            disabled={isEnsuring || !hasAnyInput}
          >
            {isEnsuring
              ? 'Preparing clip...'
              : selectedFile
                ? `Upload ${selectedFile.name}`
                : ready
                  ? 'Replace benchmark clip'
                  : 'Prepare benchmark clip'}
          </button>
        </div>
      </form>

      {clipError ? <div className="error-box">{clipError}</div> : null}
    </div>
  );
}
