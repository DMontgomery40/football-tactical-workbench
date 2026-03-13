import { useDeferredValue, useMemo, useState, type FormEvent } from 'react';

import { HelpPopover } from '../helpUi';
import { formatClassIds } from '../trainingStudio/formatters.js';
import CompactLocation from './CompactLocation';
import {
  formatAssetSource,
  formatCapabilitySummary,
  formatRecipeFamilyLabel,
  formatRecipeStack,
} from './presentation.js';
import type { BenchmarkAsset, BenchmarkRecipe } from './types';
import {
  recipeSupportsAllSuites,
} from './types';
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

interface AssetBrowserProps {
  assets: BenchmarkAsset[];
  recipes: BenchmarkRecipe[];
  selectedSuiteIds: string[];
  selectedRecipeIds: string[];
  isImportingLocal: boolean;
  isImportingHf: boolean;
  error: string;
  helpIndex: Map<string, unknown>;
  onToggleRecipe: (recipeId: string) => void;
  onImportLocal: (payload: { checkpointPath: string; label?: string }) => Promise<void> | void;
  onImportHf: (payload: { repoId: string; filename?: string; label?: string }) => Promise<void> | void;
}

type ImportMode = 'local' | 'hf' | null;
type RecipeKindFilter = 'all' | 'detector_recipe' | 'tracking_recipe' | 'pipeline_recipe';

export default function AssetBrowser({
  assets,
  recipes,
  selectedSuiteIds,
  selectedRecipeIds,
  isImportingLocal,
  isImportingHf,
  error,
  helpIndex,
  onToggleRecipe,
  onImportLocal,
  onImportHf,
}: AssetBrowserProps) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<ImportMode>(null);
  const [kindFilter, setKindFilter] = useState<RecipeKindFilter>('all');
  const [showSelectedOnly, setShowSelectedOnly] = useState(false);
  const [localPath, setLocalPath] = useState('');
  const [localLabel, setLocalLabel] = useState('');
  const [hfRepoId, setHfRepoId] = useState('');
  const [hfFilename, setHfFilename] = useState('best.pt');
  const [hfLabel, setHfLabel] = useState('');
  const deferredQuery = useDeferredValue(query.trim().toLowerCase());

  const assetById = useMemo(
    () => new Map(assets.map((asset) => [asset.assetId, asset])),
    [assets],
  );

  const filteredRecipes = useMemo(() => {
    const kindRank: Record<string, number> = {
      detector_recipe: 0,
      tracking_recipe: 1,
      pipeline_recipe: 2,
    };
    return recipes
      .filter((recipe) => {
        if (kindFilter !== 'all' && recipe.kind !== kindFilter) {
          return false;
        }
        if (showSelectedOnly && !selectedRecipeIds.includes(recipe.id)) {
          return false;
        }
      if (!deferredQuery) {
        return true;
      }
      const assetLabels = recipe.sourceAssetIds
        .map((assetId) => assetById.get(assetId)?.label || assetId)
        .join(' ')
        .toLowerCase();
      return [
        recipe.id,
        recipe.label,
        recipe.kind,
        recipe.pipeline,
        assetLabels,
      ]
        .join(' ')
        .toLowerCase()
        .includes(deferredQuery);
      })
      .sort((left, right) => {
        const selectedDelta = Number(selectedRecipeIds.includes(right.id)) - Number(selectedRecipeIds.includes(left.id));
        if (selectedDelta !== 0) {
          return selectedDelta;
        }
        const availabilityDelta = Number(right.available) - Number(left.available);
        if (availabilityDelta !== 0) {
          return availabilityDelta;
        }
        const kindDelta = (kindRank[left.kind] ?? 99) - (kindRank[right.kind] ?? 99);
        if (kindDelta !== 0) {
          return kindDelta;
        }
        return left.label.localeCompare(right.label);
      });
  }, [assetById, deferredQuery, kindFilter, recipes, selectedRecipeIds, showSelectedOnly]);

  async function handleLocalSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const checkpointPath = localPath.trim();
    if (!checkpointPath || isImportingLocal) {
      return;
    }
    await onImportLocal({ checkpointPath, label: localLabel.trim() || undefined });
    setLocalPath('');
    setLocalLabel('');
    setMode(null);
  }

  async function handleHfSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const repoId = hfRepoId.trim();
    if (!repoId || isImportingHf) {
      return;
    }
    await onImportHf({
      repoId,
      filename: hfFilename.trim() || undefined,
      label: hfLabel.trim() || undefined,
    });
    setHfRepoId('');
    setHfFilename('best.pt');
    setHfLabel('');
    setMode(null);
  }

  return (
    <section className={cx(panelClass, 'space-y-5')}>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className={sectionHeadingClass}>Asset Browser</div>
          <div className={displayTitleClass}>Select runnable recipes</div>
          <p className="max-w-2xl text-sm leading-6 text-[var(--text-muted)]">
            The backend runs recipes, not loose assets. A recipe already encodes which detector, tracker, and keypoint
            stack will be bound into the suite protocol.
          </p>
        </div>
        <HelpPopover entry={helpIndex.get('benchmark.recipe_browser')} />
      </div>

      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_14rem_auto]">
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
            Filter recipes
          </span>
          <input
            className={textInputClass}
            type="text"
            value={query}
            placeholder="Search by recipe, detector, or pipeline"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <label className="space-y-2">
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
            Recipe family
          </span>
          <select
            className={textInputClass}
            value={kindFilter}
            onChange={(event) => setKindFilter(event.target.value as RecipeKindFilter)}
          >
            <option value="all">All families</option>
            <option value="detector_recipe">Detector recipes</option>
            <option value="tracking_recipe">Tracking recipes</option>
            <option value="pipeline_recipe">Bundled pipelines</option>
          </select>
        </label>
        <div className="flex flex-wrap items-end gap-2">
          <button className={secondaryButtonClass} type="button" onClick={() => setMode(mode === 'local' ? null : 'local')}>
            Import local
          </button>
          <button className={secondaryButtonClass} type="button" onClick={() => setMode(mode === 'hf' ? null : 'hf')}>
            Import Hugging Face
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 px-4 py-3 text-sm text-[var(--text-muted)]">
        <div>
          Showing {filteredRecipes.length} recipe{filteredRecipes.length === 1 ? '' : 's'}
          {showSelectedOnly ? ' from the current selection' : ''}.
        </div>
        <button
          className={cx(
            secondaryButtonClass,
            showSelectedOnly && 'border-[color:var(--accent)] bg-[color:var(--accent-soft)]/40 text-[var(--accent-strong)]',
          )}
          type="button"
          onClick={() => setShowSelectedOnly((current) => !current)}
        >
          {showSelectedOnly ? 'Show full catalog' : 'Selected only'}
        </button>
      </div>

      {mode === 'local' ? (
        <form className={cx(insetPanelClass, 'space-y-3')} onSubmit={handleLocalSubmit}>
          <div className="flex items-center gap-2">
            <div className={sectionHeadingClass}>Local Checkpoint</div>
            <HelpPopover entry={helpIndex.get('benchmark.import.local')} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <input
              className={textInputClass}
              type="text"
              value={localPath}
              placeholder="/path/to/best.pt"
              onChange={(event) => setLocalPath(event.target.value)}
            />
            <input
              className={textInputClass}
              type="text"
              value={localLabel}
              placeholder="Optional display label"
              onChange={(event) => setLocalLabel(event.target.value)}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <button className={primaryButtonClass} type="submit" disabled={isImportingLocal || !localPath.trim()}>
              {isImportingLocal ? 'Importing…' : 'Import local detector'}
            </button>
            <button className={secondaryButtonClass} type="button" onClick={() => setMode(null)}>
              Cancel
            </button>
          </div>
        </form>
      ) : null}

      {mode === 'hf' ? (
        <form className={cx(insetPanelClass, 'space-y-3')} onSubmit={handleHfSubmit}>
          <div className="flex items-center gap-2">
            <div className={sectionHeadingClass}>Hugging Face Import</div>
            <HelpPopover entry={helpIndex.get('benchmark.import.huggingface')} />
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <input
              className={textInputClass}
              type="text"
              value={hfRepoId}
              placeholder="org/model-id"
              onChange={(event) => setHfRepoId(event.target.value)}
            />
            <input
              className={textInputClass}
              type="text"
              value={hfFilename}
              placeholder="best.pt"
              onChange={(event) => setHfFilename(event.target.value)}
            />
            <input
              className={textInputClass}
              type="text"
              value={hfLabel}
              placeholder="Optional display label"
              onChange={(event) => setHfLabel(event.target.value)}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <button className={primaryButtonClass} type="submit" disabled={isImportingHf || !hfRepoId.trim()}>
              {isImportingHf ? 'Importing…' : 'Import Hugging Face detector'}
            </button>
            <button className={secondaryButtonClass} type="button" onClick={() => setMode(null)}>
              Cancel
            </button>
          </div>
        </form>
      ) : null}

      {error ? <div className="error-box">{error}</div> : null}

      <div className="space-y-5">
        <div className="space-y-3">
          {filteredRecipes.map((recipe) => {
            const selected = selectedRecipeIds.includes(recipe.id);
            const supportedCount = selectedSuiteIds.filter((suiteId) => recipe.compatibleSuiteIds.includes(suiteId)).length;
            const supportsAll = recipeSupportsAllSuites(recipe, selectedSuiteIds);
            const sourceAssets = recipe.sourceAssetIds
              .map((assetId) => assetById.get(assetId))
              .filter(Boolean) as BenchmarkAsset[];
            return (
              <article
                key={recipe.id}
                className={cx('rounded-[1.3rem] border p-4 text-left transition', selectionTone(selected))}
              >
                <button className="w-full text-left" type="button" onClick={() => onToggleRecipe(recipe.id)}>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-base font-semibold text-[var(--text-strong)]">{recipe.label}</span>
                        <span className={cx('rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]', statusTone(recipe.available ? 'ready' : 'unavailable'))}>
                          {recipe.available ? 'Available' : 'Unavailable'}
                        </span>
                        {selectedSuiteIds.length > 0 ? (
                          <span className={cx('rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]', statusTone(supportsAll ? 'ready' : 'not_supported'))}>
                            Fits {supportedCount}/{selectedSuiteIds.length}
                          </span>
                        ) : null}
                      </div>
                      <div className="text-sm text-[var(--text)]">{formatRecipeStack(recipe, sourceAssets)}</div>
                      <div className="text-sm text-[var(--text-muted)]">
                        {formatRecipeFamilyLabel(recipe.kind)} · pipeline {recipe.pipeline || 'classic'} · binding {recipe.runtimeBinding}
                      </div>
                    </div>
                    <div className="text-right text-xs text-[var(--text-muted)]">
                      <div>{sourceAssets.length} source asset{sourceAssets.length === 1 ? '' : 's'}</div>
                      <div>{selected ? 'Selected' : 'Click to select'}</div>
                    </div>
                  </div>
                </button>

                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                      Capabilities
                    </div>
                    <div className="mt-2 text-sm text-[var(--text)]">
                      {formatCapabilitySummary(recipe.capabilities, 'No declared capabilities')}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                      Artifact
                    </div>
                    <div className="mt-2">
                      <CompactLocation
                        value={recipe.artifactPath || sourceAssets[0]?.artifactPath}
                        fallback="No artifact path"
                        detailsLabel="Show full artifact path"
                      />
                    </div>
                  </div>
                </div>

                {selected ? (
                  <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                    <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                        Source assets
                      </div>
                      <div className="mt-3 space-y-2">
                        {sourceAssets.map((asset) => (
                          <div key={asset.assetId} className="rounded-xl border border-[color:var(--line)] bg-[var(--surface)]/85 px-3 py-2">
                            <div className="flex items-center justify-between gap-3">
                              <div className="text-sm font-semibold text-[var(--text-strong)]">{asset.label}</div>
                              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                                {asset.kind}
                              </div>
                            </div>
                            <div className="mt-1 text-xs text-[var(--text-muted)]">{formatAssetSource(asset)}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                          Detector mapping
                        </div>
                        <div className="mt-2 text-sm text-[var(--text)]">
                          Players: {formatClassIds((sourceAssets[0]?.classMapping?.player_class_ids as number[] | undefined) || [])}
                        </div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          Ball: {formatClassIds((sourceAssets[0]?.classMapping?.ball_class_ids as number[] | undefined) || [])}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-[color:var(--line)] bg-[var(--surface)]/70 p-3">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                          Recipe identity
                        </div>
                        <div className="mt-2 text-sm text-[var(--text)]">{recipe.id}</div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          {recipe.bundleMode || 'separable'} · {recipe.runtimeBinding}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}

                {sourceAssets.some((asset) => asset.availabilityError) ? (
                  <div className="mt-4 rounded-2xl border border-amber-500/30 bg-amber-500/12 px-4 py-3 text-sm text-amber-700 dark:text-amber-300">
                    {sourceAssets.find((asset) => asset.availabilityError)?.availabilityError}
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
