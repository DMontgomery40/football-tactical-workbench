import { SectionTitleWithHelp } from '../helpUi';
import { formatClassIds, formatMetric, formatTimestamp } from './formatters';

function RegistryListItem({ entry, isSelected, isActive, onSelect }) {
  return (
    <button
      type="button"
      className={`studio-selection-item ${isSelected ? 'active-studio-selection-item' : ''}`}
      onClick={() => onSelect(entry.id)}
    >
      <div className="row-between">
        <div className="studio-selection-title">{entry.label || entry.id}</div>
        <div className={`active-badge ${isActive ? '' : 'inactive-badge'}`}>
          {isActive ? 'Active' : entry.is_pretrained ? 'Pretrained' : 'Available'}
        </div>
      </div>
      <div className="studio-selection-subtitle">{entry.id}</div>
      <div className="studio-selection-meta">
        <span>{entry.base_weights || 'soccana'}</span>
        <span>mAP50 {formatMetric(entry.metrics?.mAP50)}</span>
      </div>
    </button>
  );
}

function RegistryDetail({ entry, activeRegistryId, pendingActivateDetectorId, onActivateEntry }) {
  if (!entry) {
    return <div className="empty-card studio-panel">Select a detector to inspect activation state, metadata, and checkpoint details.</div>;
  }

  return (
    <section className={`card studio-panel studio-detail-panel ${entry.id === activeRegistryId ? 'active-registry-row' : ''}`}>
      <div className="row-between">
        <div>
          <div className="section-title">{entry.label || entry.id}</div>
          <div className="muted">{entry.id}</div>
        </div>
        <div className={`active-badge ${entry.id === activeRegistryId ? '' : 'inactive-badge'}`}>
          {entry.id === activeRegistryId ? 'Active' : entry.is_pretrained ? 'Pretrained' : 'Available'}
        </div>
      </div>

      <div className="studio-meta-grid">
        <div>
          <div className="micro-label">Created</div>
          <div className="studio-meta-value">{formatTimestamp(entry.created_at)}</div>
        </div>
        <div>
          <div className="micro-label">Base weights</div>
          <div className="studio-meta-value">{entry.base_weights || 'soccana'}</div>
        </div>
        <div>
          <div className="micro-label">Resolved device</div>
          <div className="studio-meta-value">{entry.resolved_device || 'n/a'}</div>
        </div>
        <div>
          <div className="micro-label">Backend</div>
          <div className="studio-meta-value">{[entry.backend, entry.backend_version ? `v${entry.backend_version}` : null].filter(Boolean).join(' ') || 'n/a'}</div>
        </div>
        <div>
          <div className="micro-label">mAP50</div>
          <div className="studio-meta-value">{formatMetric(entry.metrics?.mAP50)}</div>
        </div>
        <div>
          <div className="micro-label">Checkpoint</div>
          <div className="studio-meta-value">{entry.path}</div>
        </div>
      </div>

      {!entry.is_pretrained ? (
        <div className="studio-class-map compact-class-map">
          <div className="studio-class-map-card">
            <div className="micro-label">Player ids</div>
            <div className="studio-meta-value">{formatClassIds(entry.class_ids?.player_class_ids)}</div>
          </div>
          <div className="studio-class-map-card">
            <div className="micro-label">Ball ids</div>
            <div className="studio-meta-value">{formatClassIds(entry.class_ids?.ball_class_ids)}</div>
          </div>
          <div className="studio-class-map-card">
            <div className="micro-label">Ref ids</div>
            <div className="studio-meta-value">{formatClassIds(entry.class_ids?.referee_class_ids)}</div>
          </div>
        </div>
      ) : null}

      {entry.summary_path ? (
        <div className="studio-runtime-note compact-note">
          <div className="micro-label">Run summary</div>
          <div>{entry.summary_path}</div>
        </div>
      ) : null}

      {entry.id !== activeRegistryId ? (
        <div className="source-toolbar">
          <button
            className="primary-button compact-button"
            type="button"
            onClick={() => onActivateEntry(entry)}
            disabled={pendingActivateDetectorId === entry.id}
          >
            {pendingActivateDetectorId === entry.id ? 'Activating...' : entry.is_pretrained ? 'Use pretrained' : 'Activate'}
          </button>
        </div>
      ) : null}
    </section>
  );
}

export default function RegistryTab({
  helpIndex,
  registry,
  registryError,
  selectedRegistryEntryId,
  activeRegistryId,
  pendingActivateDetectorId,
  onSelectEntry,
  onActivateEntry,
}) {
  const detectors = registry?.detectors || [];
  const selectedEntry = detectors.find((entry) => entry.id === selectedRegistryEntryId) || detectors[0] || null;

  return (
    <section className="studio-stack">
      {registryError ? <div className="error-box">{registryError}</div> : null}
      <section className="card studio-panel">
        <div className="row-between">
          <div>
            <SectionTitleWithHelp title="Detector registry" entry={helpIndex.get('training.registry')} />
            <div className="field-note">Analysis and live preview use the active detector whenever the analysis selector stays on `soccana`.</div>
          </div>
          <div className="active-badge">{activeRegistryId === 'soccana' ? 'using pretrained detector' : 'using custom detector'}</div>
        </div>
      </section>

      {!detectors.length ? (
        <div className="card empty-card studio-panel">No detectors are registered yet.</div>
      ) : (
        <section className="studio-master-detail">
          <section className="card studio-panel studio-list-panel">
            <div className="micro-label">Detector entries</div>
            <div className="studio-selection-list">
              {detectors.map((entry) => (
                <RegistryListItem
                  key={entry.id}
                  entry={entry}
                  isSelected={entry.id === selectedRegistryEntryId}
                  isActive={entry.id === activeRegistryId}
                  onSelect={onSelectEntry}
                />
              ))}
            </div>
          </section>
          <RegistryDetail
            entry={selectedEntry}
            activeRegistryId={activeRegistryId}
            pendingActivateDetectorId={pendingActivateDetectorId}
            onActivateEntry={onActivateEntry}
          />
        </section>
      )}
    </section>
  );
}
