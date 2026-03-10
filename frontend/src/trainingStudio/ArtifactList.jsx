import { HelpPopover } from '../helpUi';

export default function ArtifactList({ artifacts, helpEntry }) {
  const entries = Object.entries(artifacts || {}).filter(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    return Boolean(value);
  });

  if (entries.length === 0) {
    return <div className="muted">No artifacts written yet.</div>;
  }

  return (
    <div className="studio-artifact-list">
      {entries.map(([key, value]) => (
        <div key={key} className="studio-artifact-row">
          <div className="label-with-help">
            <div className="micro-label">{key.replace(/_/g, ' ')}</div>
            <HelpPopover entry={helpEntry} />
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
      ))}
    </div>
  );
}
