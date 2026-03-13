import { useMemo, useRef, useState, type ChangeEvent, type FormEvent } from 'react';

import { HelpPopover } from '../helpUi';
import { formatDvcRuntime, formatDvcTrackedState } from '../trainingStudio/formatters.js';
import CompactLocation from './CompactLocation';
import { formatCapabilitySummary } from './presentation';
import type { BenchmarkDatasetState, BenchmarkSuite, ClipStatus } from './types';
import {
  cx,
  displayTitleClass,
  insetPanelClass,
  panelClass,
  primaryButtonClass,
  secondaryButtonClass,
  sectionHeadingClass,
  selectionTone,
  statusTone,
  textInputClass,
} from './ui';

const TIER_ORDER = ['operational', 'quick', 'medium', 'long'];

interface SuiteSelectorProps {
  suites: BenchmarkSuite[];
  datasetStates: BenchmarkDatasetState[];
  selectedSuiteIds: string[];
  clipStatus: ClipStatus | null;
  clipSourcePath: string;
  isPreparingClip: boolean;
  error: string;
  helpIndex: Map<string, unknown>;
  onClipSourcePathChange: (value: string) => void;
  onToggleSuite: (suiteId: string) => void;
  onPrepareClip: (sourcePath: string) => Promise<void> | void;
  onUploadClip: (file: File) => Promise<void> | void;
}

function formatTierLabel(value: string): string {
  return value ? `${value.charAt(0).toUpperCase()}${value.slice(1)}` : 'Suite';
}

export default function SuiteSelector({
  suites,
  datasetStates,
  selectedSuiteIds,
  clipStatus,
  clipSourcePath,
  isPreparingClip,
  error,
  helpIndex,
  onClipSourcePathChange,
  onToggleSuite,
  onPrepareClip,
  onUploadClip,
}: SuiteSelectorProps) {
  const datasetStateById = useMemo(
    () => new Map(datasetStates.map((state) => [state.suiteId, state])),
    [datasetStates],
  );
  const groupedSuites = useMemo(() => {
    const groups = new Map<string, BenchmarkSuite[]>();
    for (const suite of suites) {
      const tier = suite.tier || 'other';
      if (!groups.has(tier)) {
        groups.set(tier, []);
      }
      groups.get(tier)?.push(suite);
    }
    return groups;
  }, [suites]);
  const tiers = useMemo(() => {
    const remaining = [...groupedSuites.keys()].filter((tier) => !TIER_ORDER.includes(tier));
    return [...TIER_ORDER.filter((tier) => groupedSuites.has(tier)), ...remaining];
  }, [groupedSuites]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!clipSourcePath.trim() || isPreparingClip || selectedFile) {
      return;
    }
    onPrepareClip(clipSourcePath.trim());
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] || null;
    setSelectedFile(nextFile);
  }

  async function handleUploadSelected() {
    if (!selectedFile || isPreparingClip) {
      return;
    }
    await onUploadClip(selectedFile);
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  const clipReady = Boolean(clipStatus?.ready);

  return (
    <section className={cx(panelClass, 'space-y-5')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Benchmark Suites</div>
          <div className={displayTitleClass}>Pick the evidence you want</div>
          <p className="max-w-2xl text-sm leading-6 text-[var(--text-muted)]">
            Each suite fixes its own protocol, metric column set, and dataset expectations. Multi-select when you
            want the same recipes to survive more than one benchmark shape.
          </p>
        </div>
        <HelpPopover entry={helpIndex.get('benchmark.suites')} />
      </div>

      <div className={cx(insetPanelClass, 'space-y-4')}>
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className={sectionHeadingClass}>Operational Clip</span>
              <HelpPopover entry={helpIndex.get('benchmark.clip')} />
            </div>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              The operational suite uses a cached canonical clip. Detection, tracking, and overlay evidence only mean
              anything when every recipe runs the same footage.
            </p>
          </div>
          <span className={cx('rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]', statusTone(clipReady ? 'ready' : 'pending'))}>
            {clipReady ? 'Clip Ready' : 'Clip Missing'}
          </span>
        </div>

        <form className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]" onSubmit={handleSubmit}>
          <label className="space-y-2">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Local clip path
            </span>
            <input
              className={textInputClass}
              type="text"
              value={clipSourcePath}
              placeholder="/Users/you/path/to/benchmark_clip.mp4"
              disabled={Boolean(selectedFile)}
              onChange={(event) => onClipSourcePathChange(event.target.value)}
            />
          </label>
          <button
            className={primaryButtonClass}
            type="submit"
            disabled={isPreparingClip || Boolean(selectedFile) || !clipSourcePath.trim()}
          >
            {isPreparingClip ? 'Preparing…' : 'Prepare Clip'}
          </button>
        </form>

        <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
          <label className="space-y-2">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Upload video file
            </span>
            <input
              ref={fileInputRef}
              className={textInputClass}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
            />
          </label>
          <button
            className={secondaryButtonClass}
            type="button"
            disabled={isPreparingClip || !selectedFile}
            onClick={handleUploadSelected}
          >
            {isPreparingClip ? 'Uploading…' : selectedFile ? `Upload ${selectedFile.name}` : 'Upload Clip'}
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-4">
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Cached clip</div>
              <div className="mt-2">
                <CompactLocation
                  value={clipStatus?.path}
                  fallback={clipStatus?.note || 'No operational clip cached yet.'}
                  detailsLabel="Show full clip path"
                />
              </div>
              <div className="mt-2 text-xs text-[var(--text-muted)]">
                {clipStatus?.sizeMb ? `${clipStatus.sizeMb.toFixed(1)} MB` : 'Size unavailable'}
              </div>
            </div>
            <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-4">
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Durability</div>
              <div className="mt-2 text-sm text-[var(--text)]">
                DVC tracking: {formatDvcTrackedState(clipStatus?.dvc)}
              </div>
              <div className="mt-2">
                <CompactLocation
                  value={clipStatus?.cacheDir}
                  fallback="Cache directory unavailable"
                  detailsLabel="Show cache directory"
                />
              </div>
            </div>
          </div>

        {selectedFile ? (
          <div className="rounded-2xl border border-dashed border-[color:var(--accent)]/45 bg-[color:var(--accent-soft)]/40 px-4 py-3 text-sm text-[var(--accent-strong)]">
            Selected file: <strong>{selectedFile.name}</strong>
          </div>
        ) : null}
        {error ? <div className="error-box">{error}</div> : null}
      </div>

      <div className="space-y-5">
        {tiers.map((tier) => {
          const tierSuites = groupedSuites.get(tier) || [];
          if (tierSuites.length === 0) {
            return null;
          }

          return (
            <div key={tier} className="space-y-3">
              <div className="flex items-center gap-3">
                <div className={sectionHeadingClass}>{formatTierLabel(tier)}</div>
                <div className="h-px flex-1 bg-[color:var(--line)]" />
              </div>
              <div className="grid gap-3">
                {tierSuites.map((suite) => {
                  const datasetState = datasetStateById.get(suite.id) || null;
                  const selected = selectedSuiteIds.includes(suite.id);
                  const readinessStatus = datasetState?.ready ? 'ready' : (datasetState?.readinessStatus || (suite.requiresClip && clipReady ? 'ready' : 'blocked'));
                  const readinessLabel = datasetState?.ready
                    ? 'Ready'
                    : suite.requiresClip
                      ? 'Needs Clip'
                      : readinessStatus === 'blocked'
                        ? 'Blocked'
                        : 'Dataset Missing';
                  return (
                    <article
                      key={suite.id}
                      className={cx(
                        'rounded-[1.3rem] border p-4 transition',
                        selectionTone(selected),
                      )}
                    >
                      <button className="w-full text-left" type="button" onClick={() => onToggleSuite(suite.id)}>
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div className="min-w-0 space-y-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-base font-semibold text-[var(--text-strong)]">{suite.label}</span>
                              <span className={cx('rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]', statusTone(readinessStatus))}>
                                {readinessLabel}
                              </span>
                            </div>
                          <div className="text-sm text-[var(--text-muted)]">
                            {suite.family} · {suite.protocol} · primary metric {suite.primaryMetric}
                          </div>
                        </div>
                          <div className="text-right text-xs text-[var(--text-muted)]">
                            <div>{suite.datasetSplit ? `Split ${suite.datasetSplit}` : 'Split not specified'}</div>
                            <div>{selected ? 'Selected' : 'Click to select'}</div>
                          </div>
                        </div>
                      </button>

                      <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
                        <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 px-4 py-3">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                            Requirements
                          </div>
                          <div className="mt-2 text-sm text-[var(--text)]">
                            {formatCapabilitySummary(suite.requiredCapabilities)}
                          </div>
                        </div>
                        <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 px-4 py-3 text-sm text-[var(--text-muted)]">
                          {suite.id}
                        </div>
                      </div>

                      {selected ? (
                        <div className="mt-4 space-y-3">
                          <p className="text-sm leading-6 text-[var(--text)]">{suite.notes}</p>
                          <div className="grid gap-3 md:grid-cols-3">
                          <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                            <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                              Dataset
                            </div>
                            <div className="mt-2">
                              <CompactLocation
                                value={datasetState?.datasetRoot}
                                fallback={suite.requiresClip ? 'Uses the cached operational clip' : 'No dataset root declared'}
                                detailsLabel="Show dataset root"
                              />
                            </div>
                            <div className="mt-2 text-xs text-[var(--text-muted)]">
                              {datasetState?.datasetExists ? 'Materialized' : suite.requiresClip ? 'Clip-backed' : 'Not materialized'}
                            </div>
                          </div>
                          <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                            <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                              Manifest + DVC
                            </div>
                            <div className="mt-2">
                              <CompactLocation
                                value={datasetState?.manifestPath}
                                fallback="No manifest path"
                                detailsLabel="Show manifest path"
                              />
                            </div>
                            <div className="mt-2 text-xs text-[var(--text-muted)]">
                              {formatDvcTrackedState(datasetState?.manifestDvc)}
                            </div>
                          </div>
                          <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                            <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                              Runtime note
                            </div>
                            <div className="mt-2 text-sm text-[var(--text)]">{formatDvcRuntime(datasetState?.dvcRuntime)}</div>
                            <div className="mt-2 text-xs leading-5 text-[var(--text-muted)]">
                              {datasetState?.blockers?.[0] || datasetState?.note || 'No additional suite note.'}
                            </div>
                          </div>
                          </div>
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
