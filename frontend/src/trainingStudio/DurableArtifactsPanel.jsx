import { HelpPopover, MicroLabelWithHelp } from '../helpUi';
import { formatDvcRuntime, formatDvcTrackedState, formatPathTail, formatTimestamp } from './formatters';

export default function DurableArtifactsPanel({
  helpIndex,
  dvc,
  trainingProvenance,
  trainingProvenancePath,
  title = 'Durable artifacts',
}) {
  const runtime = trainingProvenance?.dvc_runtime || dvc || null;
  const datasetTracking = trainingProvenance?.dataset?.source_dvc || null;
  const checkpointTracking = trainingProvenance?.output?.checkpoint_dvc || null;
  const activation = trainingProvenance?.activation || null;

  return (
    <section className="studio-runtime-note">
      <div className="label-with-help">
        <div className="micro-label">{title}</div>
        <HelpPopover entry={helpIndex.get('training.durable_artifacts')} />
      </div>

      <div className="studio-meta-grid">
        <div>
          <MicroLabelWithHelp label="DVC runtime" entry={helpIndex.get('training.dvc_status')} />
          <div className="studio-meta-value">{formatDvcRuntime(runtime)}</div>
        </div>
        <div>
          <MicroLabelWithHelp label="Dataset tracking" entry={helpIndex.get('training.training_provenance')} />
          <div className="studio-meta-value">{formatDvcTrackedState(datasetTracking)}</div>
        </div>
        <div>
          <MicroLabelWithHelp label="Checkpoint tracking" entry={helpIndex.get('training.training_provenance')} />
          <div className="studio-meta-value">{formatDvcTrackedState(checkpointTracking)}</div>
        </div>
        <div>
          <MicroLabelWithHelp label="Provenance file" entry={helpIndex.get('training.training_provenance')} />
          <div className="studio-meta-value">{trainingProvenancePath || 'Not written yet'}</div>
        </div>
      </div>

      {runtime?.config_path ? <div className="muted">Repo config: {formatPathTail(runtime.config_path)}</div> : null}
      {trainingProvenance?.dataset?.source_path ? <div className="muted">Dataset source: {trainingProvenance.dataset.source_path}</div> : null}
      {trainingProvenance?.output?.best_checkpoint ? <div className="muted">Checkpoint: {trainingProvenance.output.best_checkpoint}</div> : null}
      {activation?.detector_id ? (
        <div className="muted">
          Activated as {activation.detector_id} on {formatTimestamp(activation.activated_at)}
        </div>
      ) : null}
      {runtime?.probe_error ? <div className="error-box">{runtime.probe_error}</div> : null}
      {trainingProvenance?.load_error ? <div className="error-box">{trainingProvenance.load_error}</div> : null}
    </section>
  );
}
