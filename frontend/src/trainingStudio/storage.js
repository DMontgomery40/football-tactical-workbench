export function readStoredString(key, fallback = '') {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    // INTENTIONAL_SWALLOW: localStorage can be unavailable in privacy or embedded contexts; falling back keeps the UI bootable.
    return fallback;
  }
}

export function readStoredJson(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    // INTENTIONAL_SWALLOW: persisted drafts are best-effort only; unreadable storage must fall back to the in-code default.
    return fallback;
  }
}

export function writeStoredValue(key, value) {
  try {
    if (value === undefined || value === null || value === '') {
      window.localStorage.removeItem(key);
      return;
    }
    window.localStorage.setItem(key, String(value));
  } catch {
    // INTENTIONAL_SWALLOW: persistence is best-effort; write failures must not block interactive state updates.
  }
}

export function writeStoredJson(key, value) {
  try {
    if (value === undefined || value === null) {
      window.localStorage.removeItem(key);
      return;
    }
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // INTENTIONAL_SWALLOW: persistence is best-effort; write failures must not block interactive state updates.
  }
}
