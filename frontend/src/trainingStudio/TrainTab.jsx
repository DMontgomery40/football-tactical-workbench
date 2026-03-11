import { FieldLabel, HelpPopover, MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';
import { formatDvcRuntime, formatPathTail } from './formatters';

function resolveWorkspaceStatus(datasetPath, datasetScan, scanMatchesCurrentPath) {
  if (!datasetPath) {
    return { className: 'idle', label: 'needs dataset' };
  }
  if (!datasetScan || !scanMatchesCurrentPath) {
    return { className: 'stopping', label: 'scan pending' };
  }
  if (datasetScan.can_start) {
    return { className: 'completed', label: 'scan ready' };
  }
  if (datasetScan.tier === 'invalid') {
    return { className: 'failed', label: 'fix scan issues' };
  }
  return { className: 'stopping', label: 'review warnings' };
}

export default function TrainTab({
  helpIndex,
  datasetPath,
  datasetScan,
  scanMatchesCurrentPath,
  form,
  enabledBaseWeights,
  deviceOptions,
  trainingConfig,
  isStarting,
  trainDisabledReason,
  trainError,
  onFormChange,
  onStartTraining,
  onOpenDatasets,
}) {
  const trimmedDatasetPath = String(datasetPath || '').trim();
  const datasetContextPath = scanMatchesCurrentPath ? datasetScan?.path : trimmedDatasetPath;
  const datasetContextLabel = datasetContextPath ? formatPathTail(datasetContextPath) : 'No dataset selected';
  const workspaceStatus = resolveWorkspaceStatus(datasetContextPath, datasetScan, scanMatchesCurrentPath);
  const hasReadyScan = Boolean(scanMatchesCurrentPath && datasetScan?.can_start);

  return (
    <section className="studio-panel-grid studio-train-grid">
      <section className="card studio-panel">
        <SectionTitleWithHelp title="Training family" entry={helpIndex.get('training.training_family')} />
        <div className="studio-family-grid">
          <div className="studio-family-card active-family-card">
            <div className="micro-label">Enabled now</div>
            <div className="studio-family-title">Detector fine-tuning</div>
            <div className="muted">Starts from `soccana`, keeps training local, and writes a detector registry entry when the run finishes.</div>
          </div>
          <div className="studio-family-card">
            <div className="micro-label">Reserved next</div>
            <div className="studio-family-title">Field calibration</div>
            <div className="muted">Future keypoint or calibration-family training slots in here without polluting the detector flow.</div>
          </div>
        </div>

        <div className={`studio-readiness-card ${hasReadyScan ? 'ready' : ''}`}>
          <MicroLabelWithHelp label="Dataset readiness" entry={helpIndex.get('training.scan_result')} />
          <div className="studio-readiness-title">
            {hasReadyScan ? 'Ready to fine-tune' : 'Dataset scan still required'}
          </div>
          <div className="muted">
            {!datasetContextPath
              ? 'Choose a dataset path from the Datasets tab and scan it before training.'
              : hasReadyScan
                ? 'The worker will generate a run-local manifest and leave your source dataset untouched.'
                : 'Fix the blocking dataset issues or rescan the current dataset path before starting a run.'}
          </div>
        </div>

        <div className="studio-form-context">
          <div className="row-between studio-form-context-head">
            <div>
              <MicroLabelWithHelp label="Dataset in scope" entry={helpIndex.get('training.dataset_path')} />
              <div className="studio-form-context-title">{datasetContextLabel}</div>
            </div>
            <div className={`status-pill ${workspaceStatus.className}`}>{workspaceStatus.label}</div>
          </div>
          <div className="studio-path-text">
            {datasetContextPath || 'Choose a dataset path from the Datasets tab, run a scan, and this workspace will stay stable while the details fill in.'}
          </div>
          <div className="studio-form-context-grid">
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="Scan status" entry={helpIndex.get('training.scan_result')} />
              <div className="studio-meta-value">{scanMatchesCurrentPath ? (datasetScan?.tier || 'waiting') : 'stale'}</div>
            </div>
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="Validation" entry={helpIndex.get('training.validation_strategy')} />
              <div className="studio-meta-value">{scanMatchesCurrentPath ? (datasetScan?.suggested_validation_strategy || 'n/a') : 'rescan required'}</div>
            </div>
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="Classes" entry={helpIndex.get('training.class_mapping')} />
              <div className="studio-meta-value">{scanMatchesCurrentPath && Array.isArray(datasetScan?.classes) ? datasetScan.classes.length : 0}</div>
            </div>
          </div>
        </div>
      </section>

      <form className="card studio-panel" onSubmit={onStartTraining}>
        <div className="studio-form-header">
          <div className="studio-form-header-copy">
            <SectionTitleWithHelp title="Detector fine-tuning" entry={helpIndex.get('training.detector_finetuning')} />
          </div>
          <div className={`status-pill ${workspaceStatus.className}`}>{workspaceStatus.label}</div>
        </div>

        <div className="studio-form-context training-form-context-inline">
          <div>
            <MicroLabelWithHelp label="Dataset source" entry={helpIndex.get('training.dataset_path')} />
            <div className="studio-form-context-title">{datasetContextLabel}</div>
            <div className="studio-path-text">{datasetContextPath || 'Choose a dataset first from the Datasets tab.'}</div>
          </div>
          <div className="studio-form-context-grid compact-form-context-grid">
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="Class source" entry={helpIndex.get('training.class_source')} />
              <div className="studio-meta-value">{scanMatchesCurrentPath ? (datasetScan?.classes_source || 'n/a') : 'n/a'}</div>
            </div>
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="Device default" entry={helpIndex.get('training.device')} />
              <div className="studio-meta-value">{form.device || 'auto'}</div>
            </div>
          </div>
        </div>

        <div className="training-form-grid">
          <label>
            <FieldLabel label="Base weights" entry={helpIndex.get('training.base_weights')} />
            <select value={form.baseWeights} onChange={(event) => onFormChange({ baseWeights: event.target.value })}>
              {enabledBaseWeights.map((item) => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
          </label>
          <label>
            <FieldLabel label="Run name" entry={helpIndex.get('training.run_name')} />
            <input
              type="text"
              value={form.runName}
              onChange={(event) => onFormChange({ runName: event.target.value })}
              placeholder="broadcast-side-cam-tune"
            />
          </label>
          <label>
            <FieldLabel label="Epochs" entry={helpIndex.get('training.epochs')} />
            <input type="number" min="1" value={form.epochs} onChange={(event) => onFormChange({ epochs: event.target.value })} />
          </label>
          <label>
            <FieldLabel label="Image size" entry={helpIndex.get('training.imgsz')} />
            <input type="number" min="32" step="32" value={form.imgsz} onChange={(event) => onFormChange({ imgsz: event.target.value })} />
          </label>
          <label>
            <FieldLabel label="Batch" entry={helpIndex.get('training.batch')} />
            <input type="number" min="1" value={form.batch} onChange={(event) => onFormChange({ batch: event.target.value })} />
          </label>
          <label>
            <FieldLabel label="Device" entry={helpIndex.get('training.device')} />
            <select value={form.device} onChange={(event) => onFormChange({ device: event.target.value })}>
              {deviceOptions.map((item) => (
                <option key={item.id || item} value={item.id || item}>{item.label || item}</option>
              ))}
            </select>
          </label>
          <label>
            <FieldLabel label="Workers" entry={helpIndex.get('training.workers')} />
            <input type="number" min="0" value={form.workers} onChange={(event) => onFormChange({ workers: event.target.value })} />
          </label>
          <label>
            <FieldLabel label="Patience" entry={helpIndex.get('training.patience')} />
            <input type="number" min="0" value={form.patience} onChange={(event) => onFormChange({ patience: event.target.value })} />
          </label>
          <label>
            <FieldLabel label="Freeze" entry={helpIndex.get('training.freeze')} />
            <input type="number" min="0" value={form.freeze} onChange={(event) => onFormChange({ freeze: event.target.value })} placeholder="leave blank for none" />
          </label>
          <label className="studio-checkbox-field">
            <span className="checkbox-label-row">
              <span>Cache images</span>
              <HelpPopover entry={helpIndex.get('training.cache')} />
            </span>
            <input type="checkbox" checked={form.cache} onChange={(event) => onFormChange({ cache: event.target.checked })} />
          </label>
        </div>

        <div className="studio-runtime-note">
          <div className="label-with-help">
            <div className="micro-label">Runtime behavior</div>
            <HelpPopover entry={helpIndex.get('training.runtime_behavior')} />
          </div>
          <div>
            The worker gets a run-local <code>dataset_runtime.yaml</code>, writes <code>summary.json</code>, <code>train.log</code>,
            plot artifacts, and the best checkpoint under <code>backend/training_runs/&lt;run_id&gt;/</code>. When you activate a run,
            the checkpoint is also copied into <code>backend/models/promoted/custom_&lt;run_id&gt;/</code> with a matching
            <code>training_provenance.json</code>.
          </div>
          <div className="studio-meta-grid compact-form-context-grid">
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="DVC runtime" entry={helpIndex.get('training.dvc_status')} />
              <div className="studio-meta-value">{formatDvcRuntime(trainingConfig?.dvc)}</div>
            </div>
            <div className="studio-context-stat">
              <MicroLabelWithHelp label="Repo config" entry={helpIndex.get('training.durable_artifacts')} />
              <div className="studio-meta-value">{trainingConfig?.dvc?.repo_enabled ? 'initialized' : 'not initialized'}</div>
            </div>
          </div>
          <div className="muted">
            DVC is optional for local runs, but when it is available the promoted detector folder and dataset pointers can become durable,
            reviewable artifacts instead of one-off local paths.
          </div>
          {trainingConfig?.device_guidance ? <div className="muted">{trainingConfig.device_guidance}</div> : null}
        </div>

        <div className="source-toolbar">
          <button className="secondary-button compact-button" type="button" onClick={onOpenDatasets}>
            Back to dataset scan
          </button>
          <button className="primary-button compact-button" type="submit" disabled={isStarting || Boolean(trainDisabledReason)}>
            {isStarting ? 'Starting fine-tuning...' : 'Start fine-tuning'}
          </button>
        </div>
        {trainDisabledReason ? <div className="field-note">{trainDisabledReason}</div> : null}
        {trainError ? <div className="error-box">{trainError}</div> : null}
      </form>
    </section>
  );
}
