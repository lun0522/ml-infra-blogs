const SHORT_DATE_FMT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});

const MONTH_DAY_FMT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
});

const TIMESTAMP_FMT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export function effectiveTimestamp(publishedAt: string | null, firstSeen: string): number {
  const iso = publishedAt || firstSeen;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? 0 : ms;
}

export function formatPostDate(iso: string | null, fallbackIso: string): string {
  const ms = effectiveTimestamp(iso, fallbackIso);
  if (!ms) return "—";
  const dt = new Date(ms);
  const now = new Date();
  return dt.getFullYear() === now.getFullYear()
    ? MONTH_DAY_FMT.format(dt)
    : SHORT_DATE_FMT.format(dt);
}

export function formatRelativeAge(iso: string | null, fallbackIso: string, referenceMs: number): string {
  const ms = effectiveTimestamp(iso, fallbackIso);
  if (!ms) return "";
  const diffHours = Math.max(0, Math.floor((referenceMs - ms) / 3_600_000));
  if (diffHours < 1) return "just now";
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays === 1) return "1d ago";
  if (diffDays < 30) return `${diffDays}d ago`;
  const diffMonths = Math.floor(diffDays / 30);
  if (diffMonths < 12) return `${diffMonths}mo ago`;
  return `${Math.floor(diffDays / 365)}y ago`;
}

export function formatLastUpdated(iso: string | null): string {
  if (!iso) return "Not yet updated";
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return iso;
  return TIMESTAMP_FMT.format(new Date(ms));
}
