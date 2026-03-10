import { MicroLabelWithHelp } from '../helpUi';
import { formatPathTail } from './formatters';

export default function ActiveDetectorSummary({ registry, activeDetector, helpIndex }) {
  const activeId = registry?.active_detector || activeDetector || 'soccana';
  const activeEntry = (registry?.detectors || []).find((item) => item.id === activeId);

  return (
    <div className="studio-active-detector">
      <div>
        <MicroLabelWithHelp label="Active detector" entry={helpIndex.get('training.active_detector')} />
        <div className="studio-active-detector-name">{activeEntry?.label || activeId}</div>
        <div className="muted">{formatPathTail(activeEntry?.path || '')}</div>
      </div>
      <div className={`active-badge ${activeId === 'soccana' ? 'default-active-badge' : ''}`}>
        {activeId === 'soccana' ? 'pretrained active' : 'custom checkpoint active'}
      </div>
    </div>
  );
}
