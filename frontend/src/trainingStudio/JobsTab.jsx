import { HelpPopover, MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';
import ArtifactList from './ArtifactList';
import DurableArtifactsPanel from './DurableArtifactsPanel';
import TrainingCurvesPanel from './TrainingCurvesPanel';
import { ACTIVE_JOB_STATUSES } from './state';
import { formatClassIds, formatMetric, formatTimestamp } from './formatters';

function JobListItem({ job, isSelected, onSelect }) {
  return (
    <button
      type="button"
      className={`studio-selection-item ${isSelected ? 'active-studio-selection-item' : ''}`}
      onClick={() => onSelect(job.job_id)}
    >
      <div className="row-between">
        <div className="studio-selection-title">{job.config?.run_name || job.run_id}</div>
        <div className={`status-pill ${job.status || 'idle'}`}>{job.status || 'idle'}</div>
      </div>
      <div className="studio-selection-subtitle">{job.run_id}</div>
      <div className="studio-selection-meta">
        <span>{job.progress || 0}%</span>
        <span>{job.current_epoch || 0}/{job.total_epochs || 0} epochs</span>
        <span>mAP50 {formatMetric(job.metrics?.mAP50)}</span>
      </div>
    </button>
  );
}

function JobDetail({
  job,
  helpIndex,
  pendingStopJobId,
  pendingActivateRunId,
  onOpenRegistry,
  onStopJob,
  onActivateRun,
}) {
  if (!job) {
    return <div className="empty-card studio-panel">Select a training job to inspect runtime state, logs, curves, and artifacts.</div>;
  }

  return (
    <section className="card studio-panel studio-detail-panel">
      <div className="row-between">
        <div>
          <div className="micro-label">Run</div>
          <div className="section-title">{job.config?.run_name || job.run_id}</div>
          <div className="muted">{job.run_id}</div>
        </div>
        <div className={`status-pill ${job.status || 'idle'}`}>{job.status || 'idle'}</div>
      </div>

      <div className="studio-meta-grid">
        <div>
          <MicroLabelWithHelp label="Base weights" entry={helpIndex.get('training.base_weights')} />
          <div className="studio-meta-value">{job.config?.base_weights || 'soccana'}</div>
        </div>
        <div>
          <MicroLabelWithHelp label="Backend" entry={helpIndex.get('training.backend_runtime')} />
          <div className="studio-meta-value">{[job.backend, job.backend_version ? `v${job.backend_version}` : null].filter(Boolean).join(' ')}</div>
        </div>
        <div>
          <MicroLabelWithHelp label="Resolved device" entry={helpIndex.get('training.resolved_device')} />
          <div className="studio-meta-value">{job.resolved_device || job.config?.device || 'pending'}</div>
        </div>
        <div>
          <MicroLabelWithHelp label="Epoch progress" entry={helpIndex.get('training.epoch_progress')} />
          <div className="studio-meta-value">{job.current_epoch || 0} / {job.total_epochs || 0}</div>
        </div>
        <div>
          <div className="micro-label">Started</div>
          <div className="studio-meta-value">{formatTimestamp(job.started_at || job.created_at)}</div>
        </div>
        <div>
          <div className="micro-label">Finished</div>
          <div className="studio-meta-value">{job.finished_at ? formatTimestamp(job.finished_at) : 'still running'}</div>
        </div>
      </div>

      <div className="progress-shell">
        <div className="progress-bar" style={{ width: `${job.progress || 0}%` }} />
      </div>

      <div className="studio-detail-toolbar">
        <div className="muted">{job.progress || 0}% complete</div>
        <div className="studio-metric-cluster">
          <MicroLabelWithHelp label="Validation metrics" entry={helpIndex.get('training.job_metrics')} />
          <div className="studio-metric-inline">
            <span>mAP50 {formatMetric(job.metrics?.mAP50)}</span>
            <span>mAP50-95 {formatMetric(job.metrics?.mAP50_95)}</span>
            <span>Precision {formatMetric(job.metrics?.precision)}</span>
            <span>Recall {formatMetric(job.metrics?.recall)}</span>
          </div>
        </div>
      </div>

      {job.error ? <div className="error-box">{job.error}</div> : null}

      <DurableArtifactsPanel
        helpIndex={helpIndex}
        trainingProvenancePath={job.training_provenance_path}
        trainingProvenance={job.training_provenance}
      />

      <TrainingCurvesPanel trainingCurves={job.training_curves} />

      <div className="source-toolbar">
        <button className="secondary-button compact-button" type="button" onClick={onOpenRegistry}>
          Open registry
        </button>
        {ACTIVE_JOB_STATUSES.has(String(job.status || '')) ? (
          <button
            className="secondary-button compact-button"
            type="button"
            onClick={() => onStopJob(job.job_id)}
            disabled={pendingStopJobId === job.job_id}
          >
            {pendingStopJobId === job.job_id ? 'Stopping...' : 'Stop'}
          </button>
        ) : null}
        {job.status === 'completed' && job.best_checkpoint ? (
          <button
            className="primary-button compact-button"
            type="button"
            onClick={() => onActivateRun(job.run_id)}
            disabled={pendingActivateRunId === job.run_id}
          >
            {pendingActivateRunId === job.run_id ? 'Activating...' : 'Activate detector'}
          </button>
        ) : null}
      </div>

      <details className="studio-log-details">
        <summary>Run artifacts and dataset contract</summary>
        <div className="studio-detail-stack">
          <div className="studio-runtime-note">
            <div className="label-with-help">
              <div className="micro-label">Generated dataset manifest</div>
              <HelpPopover entry={helpIndex.get('training.dataset_yaml')} />
            </div>
            <div>{job.generated_dataset_yaml || 'Not written yet'}</div>
          </div>
          <ArtifactList artifacts={job.artifacts} helpIndex={helpIndex} helpEntry={helpIndex.get('training.job_artifacts')} />
          {job.dataset_scan ? (
            <div className="studio-dataset-review">
              <MicroLabelWithHelp label="Dataset scan snapshot" entry={helpIndex.get('training.scan_result')} />
              <div className="studio-meta-grid">
                <div>
                  <MicroLabelWithHelp label="Tier" entry={helpIndex.get('training.scan_result')} />
                  <div className="studio-meta-value">{job.dataset_scan.tier}</div>
                </div>
                <div>
                  <MicroLabelWithHelp label="Validation strategy" entry={helpIndex.get('training.validation_strategy')} />
                  <div className="studio-meta-value">{job.validation_strategy || job.dataset_scan.suggested_validation_strategy}</div>
                </div>
                <div>
                  <MicroLabelWithHelp label="Player ids" entry={helpIndex.get('training.class_mapping')} />
                  <div className="studio-meta-value">{formatClassIds(job.dataset_scan.class_mapping?.player_class_ids)}</div>
                </div>
                <div>
                  <MicroLabelWithHelp label="Ball ids" entry={helpIndex.get('training.class_mapping')} />
                  <div className="studio-meta-value">{formatClassIds(job.dataset_scan.class_mapping?.ball_class_ids)}</div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </details>

      <details className="studio-log-details">
        <summary>Logs</summary>
        <div className="log-panel">
          {(job.logs || ['No logs yet.']).slice().reverse().map((line, index) => (
            <div key={`${job.job_id}-${index}-${line}`}>{line}</div>
          ))}
        </div>
      </details>
    </section>
  );
}

export default function JobsTab({
  helpIndex,
  jobs,
  jobsError,
  selectedJobId,
  onSelectJob,
  onOpenRegistry,
  onStopJob,
  onActivateRun,
  pendingStopJobId,
  pendingActivateRunId,
}) {
  const selectedJob = jobs.find((job) => job.job_id === selectedJobId) || jobs[0] || null;

  return (
    <section className="studio-stack">
      <section className="card studio-panel">
        <SectionTitleWithHelp title="Training jobs" entry={helpIndex.get('training.jobs')} />
      </section>
      {jobsError ? <div className="error-box">{jobsError}</div> : null}
      {!jobs.length ? (
        <div className="card empty-card studio-panel">No training jobs yet. Start a fine-tuning run from the Train tab.</div>
      ) : (
        <section className="studio-master-detail">
          <section className="card studio-panel studio-list-panel">
            <div className="micro-label">Recent runs</div>
            <div className="studio-selection-list">
              {jobs.map((job) => (
                <JobListItem
                  key={job.job_id}
                  job={job}
                  isSelected={job.job_id === selectedJobId}
                  onSelect={onSelectJob}
                />
              ))}
            </div>
          </section>
          <JobDetail
            job={selectedJob}
            helpIndex={helpIndex}
            pendingStopJobId={pendingStopJobId}
            pendingActivateRunId={pendingActivateRunId}
            onOpenRegistry={onOpenRegistry}
            onStopJob={onStopJob}
            onActivateRun={onActivateRun}
          />
        </section>
      )}
    </section>
  );
}
