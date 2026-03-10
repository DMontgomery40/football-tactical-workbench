import { FieldLabel, MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';
import { formatClassIds } from './formatters';

export default function DatasetsTab({
  helpIndex,
  datasetPath,
  onDatasetPathChange,
  onScanDataset,
  onOpenTrain,
  isScanning,
  scanError,
  datasetScan,
  isScanStale,
}) {
  const scanTierClass = datasetScan?.tier === 'valid' ? 'completed' : datasetScan?.tier === 'invalid' ? 'failed' : 'stopping';

  return (
    <section className="studio-panel-grid">
      <section className="card studio-panel">
        <SectionTitleWithHelp title="Dataset intake" entry={helpIndex.get('training.dataset_intake')} />
        <div className="field-note">
          Point this at a YOLO detector dataset root. The scanner checks split structure, label integrity, class mapping, and whether the trained checkpoint can safely come back into analysis.
        </div>
        <label>
          <FieldLabel label="Dataset path" entry={helpIndex.get('training.dataset_path')} />
          <input
            type="text"
            value={datasetPath}
            onChange={(event) => onDatasetPathChange(event.target.value)}
            placeholder="/Users/you/datasets/football-yolo"
          />
        </label>
        <div className="source-toolbar">
          <button className="secondary-button compact-button" type="button" onClick={onScanDataset}>
            {isScanning ? 'Scanning...' : 'Scan dataset'}
          </button>
          <button className="secondary-button compact-button" type="button" onClick={onOpenTrain}>
            Open training form
          </button>
        </div>
        {scanError ? <div className="error-box">{scanError}</div> : null}
      </section>

      <section className={`card studio-panel scan-result-card ${datasetScan?.tier || 'neutral'}`}>
        <div className="row-between">
          <div>
            <SectionTitleWithHelp title="Scan result" entry={helpIndex.get('training.scan_result')} />
            <div className="muted">
              {datasetScan ? (datasetScan.can_start ? 'Ready for detector fine-tuning.' : 'Needs fixes before a run can start.') : 'Run a scan to inspect readiness.'}
            </div>
          </div>
          <div className={`status-pill ${scanTierClass}`}>{datasetScan?.tier || 'waiting'}</div>
        </div>

        {!datasetScan ? (
          <div className="empty-card">Run a dataset scan to inspect splits, classes, and blocking issues before training.</div>
        ) : (
          <>
            {isScanStale ? (
              <div className="studio-list-block warn-list">
                <div className="micro-label">Current path changed</div>
                <div>The last successful scan belongs to a different dataset path. Training is blocked until you rescan the current path.</div>
              </div>
            ) : null}

            <div className="studio-meta-grid">
              <div>
                <MicroLabelWithHelp label="Dataset root" entry={helpIndex.get('training.dataset_path')} />
                <div className="studio-meta-value">{datasetScan.path}</div>
              </div>
              <div>
                <MicroLabelWithHelp label="Dataset YAML" entry={helpIndex.get('training.dataset_intake')} />
                <div className="studio-meta-value">{datasetScan.has_yaml ? datasetScan.yaml_path : 'Missing'}</div>
              </div>
              <div>
                <MicroLabelWithHelp label="Validation strategy" entry={helpIndex.get('training.validation_strategy')} />
                <div className="studio-meta-value">{datasetScan.suggested_validation_strategy || 'existing_split'}</div>
              </div>
              <div>
                <MicroLabelWithHelp label="Class source" entry={helpIndex.get('training.class_mapping')} />
                <div className="studio-meta-value">{datasetScan.classes_source || 'missing'}</div>
              </div>
            </div>

            <div className="micro-label">Classes</div>
            <div className="class-chip-row">
              {(datasetScan.classes || []).length ? (
                datasetScan.classes.map((item) => (
                  <span key={item} className="class-chip">{item}</span>
                ))
              ) : (
                <span className="muted">No class names were parsed from the dataset metadata.</span>
              )}
            </div>

            <div className="studio-class-map">
              <div className="studio-class-map-card">
                <MicroLabelWithHelp label="Player / keeper ids" entry={helpIndex.get('training.class_mapping')} />
                <div className="studio-meta-value">{formatClassIds(datasetScan.class_mapping?.player_class_ids)}</div>
              </div>
              <div className="studio-class-map-card">
                <MicroLabelWithHelp label="Ball ids" entry={helpIndex.get('training.class_mapping')} />
                <div className="studio-meta-value">{formatClassIds(datasetScan.class_mapping?.ball_class_ids)}</div>
              </div>
              <div className="studio-class-map-card">
                <MicroLabelWithHelp label="Referee ids" entry={helpIndex.get('training.class_mapping')} />
                <div className="studio-meta-value">{formatClassIds(datasetScan.class_mapping?.referee_class_ids)}</div>
              </div>
            </div>

            <div className="studio-split-grid">
              {Object.entries(datasetScan.splits || {}).map(([splitName, split]) => (
                <div key={splitName} className="studio-split-card detailed-split-card">
                  <MicroLabelWithHelp label={splitName} entry={helpIndex.get('training.dataset_intake')} />
                  <div className="studio-split-count">{split.images || 0} images</div>
                  <div className="muted">{split.label_files || 0} label files</div>
                  <div className="muted">{split.labeled_images || 0} labeled images</div>
                  <div className="muted">{split.instances || 0} instances</div>
                  <div className="muted">path: {split.path || 'missing'}</div>
                </div>
              ))}
            </div>

            {(datasetScan.warnings || []).length ? (
              <div className="studio-list-block warn-list">
                <div className="micro-label">Warnings</div>
                {(datasetScan.warnings || []).map((item) => <div key={item}>{item}</div>)}
              </div>
            ) : null}

            {(datasetScan.errors || []).length ? (
              <div className="studio-list-block error-list">
                <div className="micro-label">Blocking issues</div>
                {(datasetScan.errors || []).map((item) => <div key={item}>{item}</div>)}
              </div>
            ) : null}
          </>
        )}
      </section>
    </section>
  );
}
