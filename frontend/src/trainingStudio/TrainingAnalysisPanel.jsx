import { useEffect, useMemo, useState } from 'react';

import { HelpPopover, MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';
import { formatPathTail } from './formatters';

function friendlyProviderName(provider) {
  const normalized = String(provider || '').trim().toLowerCase();
  if (normalized === 'openai') return 'OpenAI';
  if (normalized === 'openrouter') return 'OpenRouter';
  if (normalized === 'anthropic') return 'Anthropic';
  if (normalized === 'local') return 'Local OpenAI-compatible';
  return provider || 'AI provider';
}

function sectionHelpEntry(sectionId, helpIndex) {
  return helpIndex.get(`training.ai_${sectionId}`) || helpIndex.get('training.ai_review');
}

function overallStatusLabel(value) {
  if (value === 'good') return 'Strong signal';
  if (value === 'blocked') return 'Blocked';
  return 'Mixed signal';
}

function activationRecommendationLabel(value) {
  if (value === 'activate') return 'Activation candidate';
  if (value === 'reject') return 'Do not activate';
  return 'Hold for review';
}

function resolveArtifactReferences(job, refs) {
  const artifacts = job?.artifacts || {};
  return (refs || [])
    .map((key) => {
      const value = artifacts[key];
      if (Array.isArray(value)) {
        return value.length ? { key, value: value.join('\n') } : null;
      }
      if (value) {
        return { key, value };
      }
      if (key === 'best_checkpoint' && job?.best_checkpoint) {
        return { key, value: job.best_checkpoint };
      }
      if (key === 'generated_dataset_yaml' && job?.generated_dataset_yaml) {
        return { key, value: job.generated_dataset_yaml };
      }
      if (key === 'training_provenance' && job?.training_provenance_path) {
        return { key, value: job.training_provenance_path };
      }
      return null;
    })
    .filter(Boolean);
}

function TrainingAnalysisSectionCard({ section, helpIndex, job }) {
  const artifactReferences = useMemo(
    () => resolveArtifactReferences(job, section?.artifact_refs),
    [job, section?.artifact_refs],
  );
  const helpEntry = sectionHelpEntry(section?.id, helpIndex);

  return (
    <details className={`card training-analysis-section ${section?.status || 'neutral'}`} open={section?.status !== 'good'}>
      <summary className="training-analysis-summary">
        <div className="training-analysis-summary-copy">
          <div className="label-with-help">
            <div className="training-analysis-section-title">{section?.title || 'Section'}</div>
            <HelpPopover entry={helpEntry} />
          </div>
          <div className="training-analysis-section-summary">{section?.summary || ''}</div>
        </div>
        <div className={`training-analysis-pill ${section?.status || 'neutral'}`}>{section?.status || 'neutral'}</div>
      </summary>

      <div className="training-analysis-section-body">
        <p className="training-analysis-details">{section?.details || ''}</p>

        {(section?.evidence || []).length ? (
          <div className="training-analysis-list-block">
            <MicroLabelWithHelp label="Evidence" entry={helpEntry} />
            <ul className="training-analysis-list">
              {(section.evidence || []).map((item) => (
                <li key={`${section.id}-evidence-${item}`}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {(section?.actions || []).length ? (
          <div className="training-analysis-list-block">
            <MicroLabelWithHelp label="What to do next" entry={helpEntry} />
            <ul className="training-analysis-list">
              {(section.actions || []).map((item) => (
                <li key={`${section.id}-action-${item}`}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {section?.implementation_diagnosis || section?.suggested_fix ? (
          <div className="training-analysis-drilldown-grid">
            {section?.implementation_diagnosis ? (
              <div className="training-analysis-drilldown-card">
                <MicroLabelWithHelp label="Likely code path" entry={helpEntry} />
                <div className="training-analysis-drilldown-text">{section.implementation_diagnosis}</div>
              </div>
            ) : null}
            {section?.suggested_fix ? (
              <div className="training-analysis-drilldown-card">
                <MicroLabelWithHelp label="Suggested change" entry={helpEntry} />
                <div className="training-analysis-drilldown-text">{section.suggested_fix}</div>
              </div>
            ) : null}
          </div>
        ) : null}

        {(section?.code_refs || []).length ? (
          <div className="training-analysis-chip-block">
            <MicroLabelWithHelp label="Code refs" entry={helpEntry} />
            <div className="training-analysis-chip-list">
              {(section.code_refs || []).map((ref) => (
                <span key={`${section.id}-code-${ref}`} className="training-analysis-chip">
                  {ref}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        {(section?.evidence_keys || []).length ? (
          <div className="training-analysis-chip-block">
            <MicroLabelWithHelp label="Evidence keys" entry={helpIndex.get('training.ai_review')} />
            <div className="training-analysis-chip-list">
              {(section.evidence_keys || []).map((key) => (
                <span key={`${section.id}-metric-${key}`} className="training-analysis-chip subtle">
                  {key}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        {artifactReferences.length ? (
          <div className="training-analysis-artifact-grid">
            {artifactReferences.map((artifact) => (
              <div key={`${section.id}-artifact-${artifact.key}`} className="training-analysis-artifact-card">
                <div className="micro-label">{artifact.key.replace(/_/g, ' ')}</div>
                <div className="training-analysis-artifact-value">{artifact.value}</div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </details>
  );
}

function TrainingAnalysisPromptContext({ apiBase, job, helpIndex }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [artifactState, setArtifactState] = useState({ loading: false, error: '', promptContext: null });
  const promptContext = artifactState.promptContext;

  useEffect(() => {
    setIsExpanded(false);
    setArtifactState({ loading: false, error: '', promptContext: null });
  }, [job?.run_id]);

  useEffect(() => {
    if (!isExpanded || promptContext || artifactState.loading || !job?.run_id || !job?.training_analysis_json) {
      return;
    }

    let cancelled = false;
    setArtifactState((current) => ({ ...current, loading: true, error: '' }));
    fetch(`${apiBase}/api/train/runs/${job.run_id}/analysis`)
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data?.detail || 'Could not load training analysis artifact');
        }
        if (!cancelled) {
          setArtifactState({ loading: false, error: '', promptContext: data?.prompt_context || null });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setArtifactState({ loading: false, error: error.message || 'Could not load training analysis artifact', promptContext: null });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [apiBase, artifactState.loading, isExpanded, job?.run_id, job?.training_analysis_json, promptContext]);

  if (!job?.training_analysis_json) {
    return null;
  }

  return (
    <details className="card prompt-context-card" onToggle={(event) => setIsExpanded(event.currentTarget.open)}>
      <summary className="prompt-context-summary">
        <SectionTitleWithHelp title="Training analysis prompt context" entry={helpIndex.get('training.ai_prompt_context')} />
        <div className="prompt-context-meta">
          This is the curated runtime context sent to the training-analysis model. It stays collapsed by default so the main review remains compact.
        </div>
      </summary>

      {artifactState.loading ? <div className="muted">Loading training analysis context...</div> : null}
      {artifactState.error ? <div className="error-box">{artifactState.error}</div> : null}
      {promptContext ? (
        <>
          <div className="prompt-context-grid">
            <div><span className="micro-label">Context chars</span><div>{promptContext.budget?.context_json_chars ?? 'n/a'}</div></div>
            <div><span className="micro-label">Code slices</span><div>{promptContext.budget?.code_slice_count ?? 0}</div></div>
            <div><span className="micro-label">Recent logs</span><div>{promptContext.budget?.recent_log_count ?? 0}</div></div>
            <div><span className="micro-label">Max output tokens</span><div>{promptContext.budget?.max_output_tokens ?? 'n/a'}</div></div>
          </div>
          {(promptContext.failure_context?.candidate_local_code_refs || []).length ? (
            <details className="prompt-context-subdetail">
              <summary>Failure-local code refs</summary>
              <pre className="prompt-context-code">{(promptContext.failure_context.candidate_local_code_refs || []).join('\n')}</pre>
            </details>
          ) : null}
          {(promptContext.log_highlights || []).length ? (
            <details className="prompt-context-subdetail">
              <summary>Important log chunks</summary>
              <pre className="prompt-context-code">{promptContext.log_highlights.join('\n')}</pre>
            </details>
          ) : null}
          {(promptContext.recent_logs || []).length ? (
            <details className="prompt-context-subdetail">
              <summary>Recent logs</summary>
              <pre className="prompt-context-code">{promptContext.recent_logs.join('\n')}</pre>
            </details>
          ) : null}
          {(promptContext.code_context || []).map((slice) => (
            <details key={slice.label} className="prompt-context-subdetail">
              <summary>{slice.label}</summary>
              <div className="muted">{slice.reason}</div>
              <pre className="prompt-context-code">{slice.excerpt}</pre>
            </details>
          ))}
        </>
      ) : null}
    </details>
  );
}

export default function TrainingAnalysisPanel({
  apiBase,
  job,
  helpIndex,
  onRefreshAnalysis,
  isRefreshingAnalysis,
}) {
  const sections = Array.isArray(job?.training_analysis_sections) ? job.training_analysis_sections : [];
  const hasAnalysis = sections.length > 0 || job?.training_analysis_summary_line;

  if (!job) {
    return null;
  }

  return (
    <section className="training-analysis-stack">
      <section className="card studio-panel training-analysis-overview">
        <div className="training-analysis-header">
          <div className="training-analysis-copy">
            <SectionTitleWithHelp title="AI training review" entry={helpIndex.get('training.ai_review')} />
            <div className="field-note">
              This review turns the run config, dataset contract, runtime logs, metrics, and artifact state into one activation-focused debugging narrative.
            </div>
          </div>
          <button
            className="secondary-button compact-button"
            type="button"
            onClick={() => onRefreshAnalysis(job.run_id)}
            disabled={!job?.run_id || isRefreshingAnalysis || ['queued', 'running', 'stopping', 'finalizing'].includes(String(job.status || '').toLowerCase())}
          >
            {isRefreshingAnalysis ? (job?.training_analysis_stale ? 'Regenerating...' : 'Refreshing...') : (job?.training_analysis_stale ? 'Regenerate' : 'Refresh')}
          </button>
        </div>

        {hasAnalysis ? (
          <>
            {job?.training_analysis_summary_line ? (
              <div className={`workspace-summary-line training-analysis-summary-line ${job?.training_analysis_overall_status || 'mixed'}`}>
                {job.training_analysis_summary_line}
              </div>
            ) : null}
            <div className="diagnostics-meta">
              {job.training_analysis_source === 'ai'
                ? `AI-curated for this training run via ${friendlyProviderName(job.training_analysis_provider)}${job.training_analysis_model ? ` · ${job.training_analysis_model}` : ''}.`
                : 'Heuristic training analysis is showing for this run.'}
              {job.training_analysis_error ? ` Last generation error: ${job.training_analysis_error}` : ''}
            </div>
            <div className="training-analysis-pill-row">
              <div className={`training-analysis-pill ${job?.training_analysis_overall_status || 'mixed'}`}>
                {overallStatusLabel(job?.training_analysis_overall_status || 'mixed')}
              </div>
              <div className={`training-analysis-pill recommendation ${job?.training_analysis_activation_recommendation || 'hold'}`}>
                {activationRecommendationLabel(job?.training_analysis_activation_recommendation || 'hold')}
              </div>
              {job?.training_analysis_json ? (
                <div className="training-analysis-path-chip" title={job.training_analysis_json}>
                  {formatPathTail(job.training_analysis_json)}
                </div>
              ) : null}
            </div>
            {job?.training_analysis_stale ? (
              <div className="error-box">
                {job.training_analysis_stale_reason || 'Stored AI training analysis is outdated.'} Use `Regenerate` to rebuild it for the current Training Studio prompt.
              </div>
            ) : null}
          </>
        ) : (
          <div className="empty-card">No training analysis is available for this run yet.</div>
        )}
      </section>

      {sections.length ? (
        <section className="training-analysis-sections">
          {sections.map((section) => (
            <TrainingAnalysisSectionCard
              key={`${job.run_id}-${section.id}`}
              section={section}
              helpIndex={helpIndex}
              job={job}
            />
          ))}
        </section>
      ) : null}

      {hasAnalysis ? (
        <TrainingAnalysisPromptContext apiBase={apiBase} job={job} helpIndex={helpIndex} />
      ) : null}
    </section>
  );
}
