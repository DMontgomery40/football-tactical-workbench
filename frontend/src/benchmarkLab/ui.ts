export function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ');
}

export const panelClass =
  'min-w-0 rounded-[1.5rem] border border-[color:var(--line)] bg-[var(--surface)]/95 p-5 text-[var(--text)] shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm';

export const insetPanelClass =
  'min-w-0 rounded-[1.1rem] border border-[color:var(--line)] bg-[color:var(--surface-muted)]/70 p-4';

export const sectionHeadingClass =
  "text-[0.72rem] font-semibold uppercase tracking-[0.22em] text-[var(--text-muted)] [font-family:'IBM_Plex_Sans',ui-sans-serif,sans-serif]";

export const displayTitleClass =
  "text-xl font-semibold tracking-[-0.02em] text-[var(--text-strong)] [font-family:'Iowan_Old_Style','Palatino_Linotype','Book_Antiqua',Palatino,serif]";

export const primaryButtonClass =
  'inline-flex items-center justify-center rounded-full border border-[color:var(--accent)] bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-[1.05] disabled:cursor-not-allowed disabled:opacity-50';

export const secondaryButtonClass =
  'inline-flex items-center justify-center rounded-full border border-[color:var(--line)] bg-[var(--surface)] px-4 py-2 text-sm font-medium text-[var(--text)] transition hover:border-[color:var(--accent)] hover:text-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-50';

export const textInputClass =
  'w-full rounded-2xl border border-[color:var(--line)] bg-[var(--surface)] px-4 py-3 text-sm text-[var(--text)] outline-none transition placeholder:text-[var(--text-muted)] focus:border-[color:var(--accent)] focus:ring-2 focus:ring-[color:var(--accent-soft)]';

export function statusTone(status: string): string {
  switch (String(status || '').toLowerCase()) {
    case 'completed':
    case 'ready':
      return 'border-emerald-500/30 bg-emerald-500/12 text-emerald-700 dark:text-emerald-300';
    case 'running':
    case 'queued':
    case 'pending':
      return 'border-sky-500/30 bg-sky-500/12 text-sky-700 dark:text-sky-300';
    case 'failed':
    case 'error':
      return 'border-rose-500/30 bg-rose-500/12 text-rose-700 dark:text-rose-300';
    case 'blocked':
    case 'dataset_missing':
    case 'unavailable':
    case 'not_supported':
      return 'border-amber-500/30 bg-amber-500/12 text-amber-700 dark:text-amber-300';
    default:
      return 'border-[color:var(--line)] bg-[color:var(--surface-muted)] text-[var(--text-muted)]';
  }
}

export function selectionTone(active: boolean): string {
  return active
    ? 'border-[color:var(--accent)] bg-[color:var(--accent-soft)] text-[var(--accent-strong)] shadow-[0_0_0_1px_var(--accent)]'
    : 'border-[color:var(--line)] bg-[var(--surface)] text-[var(--text)] hover:border-[color:var(--accent)]/50 hover:bg-[color:var(--accent-soft)]/40';
}
