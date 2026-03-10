import { STUDIO_TABS } from './state';

export default function StudioTabsNav({ studioTab, onSelectTab }) {
  return (
    <section className="card studio-nav">
      {STUDIO_TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className={`studio-tab ${studioTab === tab.id ? 'active-studio-tab' : ''}`}
          onClick={() => onSelectTab(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </section>
  );
}
