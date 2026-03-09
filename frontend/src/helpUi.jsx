import { useId, useState } from 'react';
import {
  FloatingFocusManager,
  FloatingPortal,
  autoUpdate,
  flip,
  offset,
  safePolygon,
  shift,
  useClick,
  useDismiss,
  useFloating,
  useFocus,
  useHover,
  useInteractions,
  useRole,
} from '@floating-ui/react';

export function buildHelpIndex(helpCatalog) {
  const index = new Map();
  for (const entry of helpCatalog || []) {
    for (const target of entry?.targets || []) {
      if (target && !index.has(target)) {
        index.set(target, entry);
      }
    }
  }
  return index;
}

function formatHelpLinkMeta(link) {
  const bits = [link.kind, link.published_at].filter(Boolean);
  return bits.join(' · ');
}

export function HelpPopover({ entry }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const titleId = `${panelId}-title`;
  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: setOpen,
    placement: 'right-start',
    whileElementsMounted: autoUpdate,
    middleware: [offset(10), flip({ padding: 16 }), shift({ padding: 16 })],
  });
  const hover = useHover(context, {
    move: false,
    handleClose: safePolygon({ buffer: 4 }),
  });
  const focus = useFocus(context);
  const click = useClick(context, { toggle: true });
  const dismiss = useDismiss(context);
  const role = useRole(context, { role: 'dialog' });
  const { getReferenceProps, getFloatingProps } = useInteractions([hover, focus, click, dismiss, role]);

  if (!entry) {
    return null;
  }

  return (
    <span className={`help-popover ${open ? 'open' : ''}`}>
      <button
        ref={refs.setReference}
        type="button"
        className="help-trigger"
        aria-label={`More about ${entry.title}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        {...getReferenceProps()}
      >
        i
      </button>
      {open ? (
        <FloatingPortal>
          <FloatingFocusManager context={context} modal={false} initialFocus={-1} returnFocus={false}>
            <div
              ref={refs.setFloating}
              id={panelId}
              aria-labelledby={titleId}
              className="help-panel"
              style={floatingStyles}
              {...getFloatingProps()}
            >
              <div id={titleId} className="help-panel-title">{entry.title}</div>
              {entry.summary ? <div className="help-panel-summary">{entry.summary}</div> : null}
              {(entry.body || []).length ? (
                <div className="help-panel-body">
                  {entry.body.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}
                </div>
              ) : null}
              {(entry.links || []).length ? (
                <div className="help-panel-links">
                  <div className="micro-label">References</div>
                  {(entry.links || []).map((link) => (
                    <a key={link.url} href={link.url} target="_blank" rel="noreferrer" className="help-link">
                      <span>{link.label}</span>
                      {formatHelpLinkMeta(link) ? <span className="help-link-meta">{formatHelpLinkMeta(link)}</span> : null}
                    </a>
                  ))}
                </div>
              ) : null}
            </div>
          </FloatingFocusManager>
        </FloatingPortal>
      ) : null}
    </span>
  );
}

export function FieldLabel({ label, entry }) {
  return (
    <span className="field-label-row">
      <span>{label}</span>
      <HelpPopover entry={entry} />
    </span>
  );
}

export function SectionTitleWithHelp({ title, entry, className = '' }) {
  return (
    <div className={`section-title ${className}`.trim()}>
      <span className="label-with-help">
        <span>{title}</span>
        <HelpPopover entry={entry} />
      </span>
    </div>
  );
}

export function MicroLabelWithHelp({ label, entry }) {
  return (
    <span className="micro-label-with-help">
      <span className="micro-label">{label}</span>
      <HelpPopover entry={entry} />
    </span>
  );
}
