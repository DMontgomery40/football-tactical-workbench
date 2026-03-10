import { MicroLabelWithHelp, SectionTitleWithHelp } from '../helpUi';
import ActiveDetectorSummary from './ActiveDetectorSummary';

export default function StudioHeader({ trainingConfig, registry, activeDetector, helpIndex }) {
  return (
    <section className="card studio-header training-studio-hero">
      <div className="studio-header-copy">
        <div className="eyebrow">training studio</div>
        <SectionTitleWithHelp
          title="Fine-tune the football detector in its own workspace."
          entry={helpIndex.get('training.studio_overview')}
        />
        <p className="studio-intro">
          This V1 stays focused: start from the football-pretrained `soccana` checkpoint, adapt it to your camera domain locally,
          and then promote the best checkpoint straight back into analysis.
        </p>
        <MicroLabelWithHelp label="Runtime profile" entry={helpIndex.get('training.runtime_profile')} />
        <div className="studio-status-ribbon">
          <span>{trainingConfig?.backend_label || 'Training backend'}</span>
          {trainingConfig?.backend_version ? <span>v{trainingConfig.backend_version}</span> : null}
          <span>mac-first via MPS</span>
          <span>CUDA-ready path preserved</span>
        </div>
      </div>
      <div className="studio-header-side">
        <ActiveDetectorSummary registry={registry} activeDetector={activeDetector} helpIndex={helpIndex} />
        {trainingConfig?.license_note ? (
          <div className="studio-license-note">
            <div className="micro-label">Licensing caveat</div>
            <div>{trainingConfig.license_note}</div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
