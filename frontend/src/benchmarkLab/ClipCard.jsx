import { useRef, useState } from 'react';

import { MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';

export default function ClipCard({
  clipStatus,
  runtimeProfile,
  clipSourcePath,
  onClipSourcePathChange,
  isEnsuring,
  clipError,
  onEnsureClip,
  onUploadClip,
  helpIndex,
}) {
  const ready = clipStatus?.ready;
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

    // File takes priority over text path
    if (selectedFile) {
      onUploadClip(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } else {
      onEnsureClip(clipSourcePath.trim());
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    // Do NOT auto-upload. Wait for the user to click Prepare.
  }

  function handleClearFile() {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  return (
    <div className="card benchmark-clip-card">
      <div className="benchmark-clip-header">
        <SectionTitleWithHelp title="Benchmark clip" entry={helpIndex?.get('benchmark.clip')} />
        <span className={`benchmark-clip-badge ${ready ? 'clip-ready' : 'clip-pending'}`}>
          {ready ? 'Ready' : 'Not ready'}
        </span>
      </div>

      {ready ? (
        <div className="benchmark-clip-meta">
          <div className="benchmark-clip-path muted">{cachedPath}</div>
          <div className="benchmark-clip-info">
            {sizeMb != null ? <span className="muted">{sizeMb.toFixed(1)} MB</span> : null}
            {duration != null ? <span className="muted">{duration.toFixed(1)}s</span> : null}
          </div>
          <p className="muted benchmark-clip-ready-hint">
            This clip is cached and locked for all benchmark runs. To replace it, enter a new path or pick a file below and click Prepare.
          </p>
        </div>
      ) : (
        <p className="muted benchmark-clip-empty-hint">
          A short reference clip is needed to benchmark candidates under identical conditions.
          Provide a local video path or upload a file, then click Prepare.
        </p>
      )}

      <form className="benchmark-clip-form" onSubmit={handleSubmit}>
        <label>
          <span className="micro-label">Local video path</span>
          <input
            type="text"
            value={clipSourcePath}
            onChange={(e) => onClipSourcePathChange(e.target.value)}
            placeholder="/Users/you/path/to/clip.mp4"
            disabled={selectedFile != null}
          />
        </label>

        <div className="benchmark-clip-or muted">or</div>

        <label className="benchmark-clip-upload-label">
          <span className="micro-label">Upload video file</span>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
          />
        </label>

        {selectedFile ? (
          <div className="benchmark-clip-selected-file">
            <span className="muted">Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)</span>
            <button type="button" className="inline-link-button" onClick={handleClearFile}>Clear</button>
          </div>
        ) : null}

        <button
          type="submit"
          className="compact-button"
          disabled={isEnsuring || !hasAnyInput}
        >
          {isEnsuring
            ? 'Preparing clip...'
            : selectedFile
              ? `Upload and prepare ${selectedFile.name}`
              : 'Prepare benchmark clip'}
        </button>
      </form>

      {clipError ? <div className="error-box">{clipError}</div> : null}

      {runtimeProfile ? (
        <div className="benchmark-runtime-summary">
          <MicroLabelWithHelp label="Shared runtime defaults" entry={helpIndex?.get('benchmark.runtime_profile')} />
          <div className="benchmark-runtime-grid">
            <span className="muted">Native pipeline: <strong>{runtimeProfile.native_pipeline || runtimeProfile.pipeline || 'classic'}</strong></span>
            <span className="muted">Native keypoint: <strong>{runtimeProfile.native_keypoint_model || runtimeProfile.keypoint_model || 'soccana_keypoint'}</strong></span>
            <span className="muted">Tracker: <strong>{runtimeProfile.tracker_mode || 'hybrid_reid'}</strong></span>
            <span className="muted">Ball: <strong>{runtimeProfile.include_ball ? 'yes' : 'no'}</strong></span>
            <span className="muted">Player conf: <strong>{runtimeProfile.player_conf ?? '0.25'}</strong></span>
            <span className="muted">Ball conf: <strong>{runtimeProfile.ball_conf ?? '0.20'}</strong></span>
            <span className="muted">IoU: <strong>{runtimeProfile.iou ?? '0.50'}</strong></span>
          </div>
          {runtimeProfile.note ? (
            <p className="muted benchmark-clip-ready-hint">{runtimeProfile.note}</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
