import { describeLocation } from './presentation.js';

interface CompactLocationProps {
  value: string | null | undefined;
  fallback?: string;
  detailsLabel?: string;
}

export default function CompactLocation({
  value,
  fallback = 'Not provided',
  detailsLabel = 'Show full location',
}: CompactLocationProps) {
  const display = describeLocation(value);

  if (!display.full) {
    return <div className="text-sm text-[var(--text)]">{fallback}</div>;
  }

  return (
    <div className="min-w-0 space-y-1">
      {display.href ? (
        <a
          className="text-sm font-medium text-[var(--accent-strong)] [overflow-wrap:anywhere] hover:text-[var(--accent)] hover:underline"
          href={display.href}
          rel="noreferrer"
          target="_blank"
        >
          {display.primary}
        </a>
      ) : (
        <div className="text-sm font-medium text-[var(--text-strong)] [overflow-wrap:anywhere]">{display.primary}</div>
      )}
      {display.secondary ? (
        <div className="text-xs text-[var(--text-muted)] [overflow-wrap:anywhere]">{display.secondary}</div>
      ) : null}
      {display.full !== display.primary ? (
        <details className="group">
          <summary className="cursor-pointer list-none text-xs font-medium text-[var(--accent-strong)] hover:text-[var(--accent)]">
            {detailsLabel}
          </summary>
          <div className="mt-1 text-xs text-[var(--text-muted)] [overflow-wrap:anywhere]">{display.full}</div>
        </details>
      ) : null}
    </div>
  );
}
