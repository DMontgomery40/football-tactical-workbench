import { HelpPopover } from '../helpUi';

function resolveArtifactHelpEntry(key, helpIndex, fallbackEntry) {
  const normalized = String(key || '').toLowerCase();
  if (normalized.includes('summary')) {
    return helpIndex?.get('training.run_summary_artifact') || fallbackEntry;
  }
  if (normalized.includes('checkpoint') || normalized.includes('weights') || normalized.includes('best')) {
    return helpIndex?.get('training.checkpoint_path') || fallbackEntry;
  }
  if (normalized.includes('dataset') && (normalized.includes('yaml') || normalized.includes('manifest'))) {
    return helpIndex?.get('training.dataset_yaml') || fallbackEntry;
  }
  if (normalized.includes('provenance') || normalized.includes('promoted')) {
    return helpIndex?.get('training.training_provenance') || fallbackEntry;
  }
  if (normalized.includes('analysis')) {
    return helpIndex?.get('training.ai_review') || fallbackEntry;
  }
  if (normalized.includes('metric') || normalized.includes('result')) {
    return helpIndex?.get('training.job_metrics') || fallbackEntry;
  }
  return fallbackEntry;
}

export default function ArtifactList({ artifacts, helpIndex, helpEntry }) {
  const entries = Object.entries(artifacts || {}).filter(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    return Boolean(value);
  });

  if (entries.length === 0) {
    return <div className="muted">No artifacts written yet.</div>;
  }

  return (
    <div className="studio-artifact-list">
      {entries.map(([key, value]) => {
        const entry = resolveArtifactHelpEntry(key, helpIndex, helpEntry);
        return (
          <div key={key} className="studio-artifact-row">
            <div className="label-with-help">
              <div className="micro-label">{key.replace(/_/g, ' ')}</div>
              <HelpPopover entry={entry} />
            </div>
            {Array.isArray(value) ? (
              <div className="studio-artifact-values">
                {value.map((item) => (
                  <div key={item} className="studio-meta-value">{item}</div>
                ))}
              </div>
            ) : (
              <div className="studio-meta-value">{value}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
