import { useEffect, useMemo, useRef, useState } from "react";
import MiniSearch from "minisearch";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Cpu,
  ExternalLink,
  Moon,
  Search,
  Sun,
  X,
} from "lucide-react";
import type { ArchivePayload, PostRecord } from "./types";
import {
  effectiveTimestamp,
  formatLastUpdated,
  formatPostDate,
  formatRelativeAge,
} from "./utils/format";

type Theme = "light" | "dark";

const THEME_STORAGE_KEY = "ml-infra-blogs-theme";
const PAGE_SIZE = 20;

function getPaginationItems(current: number, total: number): (number | "ellipsis")[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, idx) => idx + 1);
  }
  if (current <= 3) {
    return [1, 2, 3, 4, "ellipsis", total];
  }
  if (current >= total - 2) {
    return [1, "ellipsis", total - 3, total - 2, total - 1, total];
  }
  return [1, "ellipsis", current - 1, current, current + 1, "ellipsis", total];
}

function readInitialTheme(): Theme {
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // Ignore storage errors in restricted contexts.
  }
  return "light";
}

function resolveAssetPath(relPath: string): string {
  const base = import.meta.env.BASE_URL || "./";
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}${relPath.replace(/^\/+/, "")}`;
}

export function App() {
  const [theme, setTheme] = useState<Theme>(readInitialTheme);
  const isDark = theme === "dark";

  const [payload, setPayload] = useState<ArchivePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSource, setSelectedSource] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);

  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore storage errors.
    }
  }, [isDark, theme]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.key === "/" &&
        document.activeElement?.tagName !== "INPUT" &&
        document.activeElement?.tagName !== "TEXTAREA"
      ) {
        event.preventDefault();
        searchInputRef.current?.focus();
      } else if (
        event.key === "Escape" &&
        document.activeElement === searchInputRef.current
      ) {
        setSearchQuery("");
        searchInputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(resolveAssetPath("data/posts.json"), { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status} loading blog archive`);
        return res.json() as Promise<ArchivePayload>;
      })
      .then((data) => {
        if (!cancelled) {
          setPayload(data);
          setErrorMsg(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setErrorMsg(err instanceof Error ? err.message : "Failed to load posts");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, selectedSource]);

  const referenceMs = useMemo(() => {
    if (payload?.generated_at) {
      const parsed = Date.parse(payload.generated_at);
      if (!Number.isNaN(parsed)) return Math.max(parsed, Date.now());
    }
    return Date.now();
  }, [payload?.generated_at]);

  const sortedSources = useMemo(
    () =>
      [...(payload?.sources ?? [])].sort((a, b) =>
        a.name.localeCompare(b.name, "en", { sensitivity: "base" }),
      ),
    [payload?.sources],
  );

  const searchIndex = useMemo(() => {
    const index = new MiniSearch<PostRecord>({
      fields: ["title", "summary", "source_name"],
      searchOptions: {
        boost: { title: 3, source_name: 2, summary: 1 },
        prefix: true,
        fuzzy: (term) => (term.length >= 4 ? 0.2 : false),
        combineWith: "AND",
      },
    });
    const postById = new Map<string, PostRecord>();
    const uniquePosts: PostRecord[] = [];

    for (const post of payload?.posts ?? []) {
      if (postById.has(post.id)) continue;
      postById.set(post.id, post);
      uniquePosts.push(post);
    }
    if (uniquePosts.length > 0) {
      index.addAll(uniquePosts);
    }

    return { index, postById };
  }, [payload?.posts]);

  const filteredPosts = useMemo(() => {
    if (!payload) return [];
    const q = searchQuery.trim();

    if (!q) {
      if (selectedSource === "all") return payload.posts;
      return payload.posts.filter((post) => post.source_id === selectedSource);
    }

    return searchIndex.index
      .search(q)
      .map((result) => searchIndex.postById.get(String(result.id)))
      .filter(
        (post): post is PostRecord =>
          post !== undefined &&
          (selectedSource === "all" || post.source_id === selectedSource),
      );
  }, [payload, searchIndex, searchQuery, selectedSource]);

  const totalPages = Math.max(1, Math.ceil(filteredPosts.length / PAGE_SIZE));
  const safePage = Math.min(currentPage, totalPages);
  const startIdx = (safePage - 1) * PAGE_SIZE;

  const pagedPosts = useMemo(
    () => filteredPosts.slice(startIdx, startIdx + PAGE_SIZE),
    [filteredPosts, startIdx],
  );

  const pageItems = useMemo(
    () => getPaginationItems(safePage, totalPages),
    [safePage, totalPages],
  );

  const goToPage = (page: number) => {
    const next = Math.min(Math.max(1, page), totalPages);
    setCurrentPage(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const clearFilters = () => {
    setSearchQuery("");
    setSelectedSource("all");
  };

  return (
    <div className="min-h-screen text-slate-900 dark:text-slate-100">
      {/* Sticky Top Header */}
      <header className="sticky top-0 z-30 border-b border-slate-200/90 bg-white/85 backdrop-blur-md dark:border-slate-800/80 dark:bg-[#090F18]/90">
        <div className="mx-auto max-w-[1360px] px-4 py-3 md:px-6">
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
            <div className="order-1 flex shrink-0 items-center gap-2.5">
              <div
                aria-hidden="true"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-sky-500/12 text-sky-600 ring-1 ring-sky-500/25 dark:bg-sky-400/12 dark:text-sky-400 dark:ring-sky-400/25"
              >
                <Cpu className="h-[18px] w-[18px]" />
              </div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl dark:text-white">
                AI &amp; LLM Infra Blogs
              </h1>
            </div>

            <div className="order-3 flex w-full flex-wrap items-baseline gap-x-2 gap-y-0.5 text-xs text-slate-500 md:order-2 md:w-auto md:flex-1 md:justify-end dark:text-slate-400">
              <span>Not affiliated with or endorsed by any listed source</span>
              <span aria-hidden="true" className="hidden text-slate-400 sm:inline dark:text-slate-600">
                ·
              </span>
              <span>Up to 50 posts per source</span>
              {payload?.generated_at && (
                <>
                  <span aria-hidden="true" className="hidden text-slate-400 sm:inline dark:text-slate-600">
                    ·
                  </span>
                  <span className="whitespace-nowrap tabular-nums">
                    Updated {formatLastUpdated(payload.generated_at)}
                  </span>
                </>
              )}
            </div>

            {/* Theme Toggle */}
            <button
              type="button"
              onClick={() => setTheme(isDark ? "light" : "dark")}
              aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
              className="order-2 shrink-0 cursor-pointer rounded-lg border border-slate-200 bg-white p-1.5 text-slate-600 transition hover:border-sky-400 hover:text-slate-900 md:order-3 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-sky-600 dark:hover:text-white"
            >
              {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>

          <div className="mt-2.5 flex items-center gap-2.5">
            {/* Search Box */}
            <div className="relative min-w-0 flex-1">
              <Search className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                ref={searchInputRef}
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter posts (press / to focus)"
                className="w-full rounded-xl border border-slate-200 bg-slate-50/70 py-2 pr-9 pl-9 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-sky-500/20 dark:border-slate-700/80 dark:bg-slate-950/70 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-sky-400 dark:focus:bg-slate-950"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  aria-label="Clear search"
                  className="absolute top-1/2 right-2.5 -translate-y-1/2 cursor-pointer rounded p-0.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Source Dropdown */}
            <select
              aria-label="Filter by blog source"
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.target.value)}
              className="max-w-[160px] shrink-0 cursor-pointer rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-800 focus:border-sky-500 focus:outline-none sm:max-w-none dark:border-slate-700/80 dark:bg-slate-950 dark:text-slate-200"
            >
              <option value="all">All Sources ({payload?.posts.length ?? 0})</option>
              {sortedSources.map((src) => (
                <option key={src.id} value={src.id}>
                  {src.name} ({src.total_count})
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1360px] px-4 pt-6 pb-10 md:px-6">

        {/* Loading & Error States */}
        {loading && (
          <div className="rounded-2xl border border-slate-200 bg-white/70 p-12 text-center text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900/60 dark:text-slate-400">
            Loading aggregated posts...
          </div>
        )}

        {errorMsg && !loading && (
          <div className="flex items-center gap-3 rounded-2xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/70 dark:bg-rose-950/40 dark:text-rose-200">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Empty State */}
        {!loading && !errorMsg && filteredPosts.length === 0 && (
          <div className="rounded-2xl border border-slate-200 bg-white/80 p-12 text-center dark:border-slate-800 dark:bg-slate-900/60">
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
              No posts match the current filters.
            </p>
            <button
              type="button"
              onClick={clearFilters}
              className="mt-3 cursor-pointer rounded-lg bg-sky-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
            >
              Reset all filters
            </button>
          </div>
        )}

        {/* Posts Timeline */}
        {!loading && !errorMsg && filteredPosts.length > 0 && (
          <div className="space-y-6">
            <div className="divide-y divide-slate-200/80 overflow-hidden rounded-2xl border border-slate-200/90 bg-white/90 shadow-xs dark:divide-slate-800/80 dark:border-slate-800/90 dark:bg-slate-900/75">
              {pagedPosts.map((post) => (
                <PostRow
                  key={post.id}
                  post={post}
                  referenceMs={referenceMs}
                />
              ))}
            </div>

            {totalPages > 1 && (
              <nav
                aria-label="Pagination"
                className="flex flex-wrap items-center justify-center gap-1 text-xs"
              >
                <button
                  type="button"
                  onClick={() => goToPage(safePage - 1)}
                  disabled={safePage <= 1}
                  aria-label="Previous page"
                  className="inline-flex cursor-pointer items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 font-medium text-slate-700 transition hover:border-sky-400 hover:text-sky-600 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-sky-600 dark:hover:text-sky-400"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Prev
                </button>

                {pageItems.map((item, idx) =>
                  item === "ellipsis" ? (
                    <span
                      key={`ellipsis-${idx}`}
                      className="px-2 py-1 text-slate-400 select-none dark:text-slate-500"
                    >
                      …
                    </span>
                  ) : (
                    <button
                      key={item}
                      type="button"
                      onClick={() => goToPage(item)}
                      aria-current={item === safePage ? "page" : undefined}
                      className={`min-w-[2rem] cursor-pointer rounded-lg border px-2 py-1.5 font-medium tabular-nums transition ${
                        item === safePage
                          ? "border-sky-600 bg-sky-600 text-white dark:border-sky-500 dark:bg-sky-500"
                          : "border-slate-200 bg-white text-slate-700 hover:border-sky-400 hover:text-sky-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-sky-600 dark:hover:text-sky-400"
                      }`}
                    >
                      {item}
                    </button>
                  ),
                )}

                <button
                  type="button"
                  onClick={() => goToPage(safePage + 1)}
                  disabled={safePage >= totalPages}
                  aria-label="Next page"
                  className="inline-flex cursor-pointer items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 font-medium text-slate-700 transition hover:border-sky-400 hover:text-sky-600 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-sky-600 dark:hover:text-sky-400"
                >
                  Next
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </nav>
            )}
          </div>
        )}

      </main>
    </div>
  );
}

interface PostRowProps {
  post: PostRecord;
  referenceMs: number;
}

function PostRow({
  post,
  referenceMs,
}: PostRowProps) {
  const ts = effectiveTimestamp(post.published_at, post.first_seen);
  const isRecent = referenceMs - ts <= 2 * 86_400_000;

  return (
    <article className="group px-4 py-3.5 transition hover:bg-sky-50/45 sm:px-5 dark:hover:bg-slate-800/40">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs">
          <span className="font-semibold text-sky-700 dark:text-sky-400">
            {post.source_name}
          </span>
          {isRecent && (
            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-400/15 dark:text-emerald-300">
              New
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2 text-xs text-slate-500 tabular-nums dark:text-slate-400">
          <span>{formatPostDate(post.published_at, post.first_seen)}</span>
          <span className="text-slate-300 dark:text-slate-700">·</span>
          <span>{formatRelativeAge(post.published_at, post.first_seen, referenceMs)}</span>
        </div>
      </div>

      <h3 className="mt-1 text-[15px] leading-snug font-semibold text-slate-900 dark:text-slate-100">
        <a
          href={post.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-baseline gap-1.5 transition group-hover:text-sky-700 dark:group-hover:text-sky-300"
        >
          <span>{post.title}</span>
          <ExternalLink className="inline h-3.5 w-3.5 shrink-0 translate-y-0.5 opacity-0 transition group-hover:opacity-70" />
        </a>
      </h3>

      {post.summary && (
        <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
          {post.summary}
        </p>
      )}
    </article>
  );
}

