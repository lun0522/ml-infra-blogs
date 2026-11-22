import { useEffect, useMemo, useRef, useState } from "react";
import MiniSearch from "minisearch";
import {
  AlertCircle,
  Bookmark,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Cpu,
  ExternalLink,
  Moon,
  Search,
  Sun,
  X,
} from "lucide-react";
import { PINNED_REFERENCES, type PinnedReference } from "./data/references";
import type { ArchivePayload, PillarId, PostRecord } from "./types";
import {
  effectiveTimestamp,
  formatLastUpdated,
  formatPostDate,
  formatRelativeAge,
} from "./utils/format";

type Theme = "light" | "dark";
type TopicFilter = PillarId | "all" | "filtered";

const THEME_STORAGE_KEY = "ml-infra-blogs-theme";
const PAGE_SIZE = 20;

const TOPIC_PILLARS: readonly { id: PillarId; label: string }[] = [
  { id: "inference", label: "Inference & Serving" },
  { id: "training", label: "Training & RL" },
  { id: "hardware", label: "Hardware & Silicon" },
  { id: "systems", label: "Clusters, Sandboxes & Storage" },
  { id: "agents", label: "Agents, Harnesses & Evals" },
];

const TOPIC_LABEL_MAP = new Map<PillarId, string>(
  TOPIC_PILLARS.map((item) => [item.id, item.label]),
);

function isPostFiltered(post: PostRecord): boolean {
  if (Array.isArray(post.topics)) {
    return post.topics.length === 0;
  }
  return post.label === "drop";
}

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

interface IndexedCollection<T extends { id: string }> {
  index: MiniSearch<T>;
  byId: Map<string, T>;
}

function createSearchIndex<T extends { id: string }>(
  items: readonly T[],
  fields: (keyof T & string)[],
  boost: Record<string, number>,
): IndexedCollection<T> {
  const index = new MiniSearch<T>({
    fields,
    searchOptions: {
      boost,
      prefix: true,
      fuzzy: (term) => (term.length >= 4 ? 0.2 : false),
      combineWith: "AND",
    },
  });
  const byId = new Map<string, T>();
  const uniqueItems: T[] = [];

  for (const item of items) {
    if (byId.has(item.id)) continue;
    byId.set(item.id, item);
    uniqueItems.push(item);
  }
  if (uniqueItems.length > 0) {
    index.addAll(uniqueItems);
  }

  return { index, byId };
}

function querySearchIndex<T extends { id: string }>(
  collection: IndexedCollection<T>,
  query: string,
): T[] {
  const matches: T[] = [];
  for (const result of collection.index.search(query)) {
    const item = collection.byId.get(String(result.id));
    if (item) matches.push(item);
  }
  return matches;
}

interface SourceDropdownProps {
  value: string;
  totalCount: number;
  sources: Array<{ id: string; name: string }>;
  sourceCounts: Map<string, number>;
  onChange: (value: string) => void;
}

function SourceDropdown({
  value,
  totalCount,
  sources,
  sourceCounts,
  onChange,
}: SourceDropdownProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleMouseDown(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const activeSource = value === "all" ? null : sources.find((s) => s.id === value);
  const activeLabel = activeSource ? activeSource.name : "All Sources";
  const activeCount = activeSource ? (sourceCounts.get(activeSource.id) ?? 0) : totalCount;

  const selectOption = (nextValue: string) => {
    onChange(nextValue);
    setOpen(false);
  };

  return (
    <div ref={containerRef} className="relative w-full shrink-0 sm:w-56">
      <button
        type="button"
        aria-label="Filter by blog source"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="inline-flex h-10 w-full cursor-pointer items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50/70 px-3.5 text-sm font-medium text-slate-800 transition hover:border-slate-300 focus:border-sky-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-sky-500/20 dark:border-slate-700/80 dark:bg-slate-950/70 dark:text-slate-200 dark:hover:border-slate-600 dark:focus:border-sky-400 dark:focus:bg-slate-950"
      >
        <span className="truncate">{activeLabel}</span>
        <span className="flex shrink-0 items-center gap-1.5">
          <span className="text-xs font-normal tabular-nums text-slate-400 dark:text-slate-500">
            {activeCount}
          </span>
          <ChevronDown
            className={`h-4 w-4 text-slate-400 transition-transform duration-150 ${
              open ? "rotate-180" : ""
            }`}
          />
        </span>
      </button>

      {open && (
        <div
          role="listbox"
          aria-label="Blog sources"
          className="absolute right-0 z-30 mt-1.5 w-full min-w-[240px] overflow-hidden rounded-xl border border-slate-200 bg-white p-1 shadow-lg shadow-slate-900/10 sm:w-64 dark:border-slate-700/80 dark:bg-slate-900 dark:shadow-black/40"
        >
          <div className="max-h-80 overflow-y-auto">
            <button
              type="button"
              role="option"
              aria-selected={value === "all"}
              onClick={() => selectOption("all")}
              className={`flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition ${
                value === "all"
                  ? "bg-sky-50 font-medium text-sky-700 dark:bg-sky-500/15 dark:text-sky-300"
                  : "text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
              }`}
            >
              <span className="flex min-w-0 items-center gap-2">
                <Check className={`h-3.5 w-3.5 shrink-0 ${value === "all" ? "opacity-100" : "opacity-0"}`} />
                <span className="truncate">All Sources</span>
              </span>
              <span className="shrink-0 text-xs tabular-nums text-slate-400 dark:text-slate-500">
                {totalCount}
              </span>
            </button>

            <div className="my-1 border-t border-slate-100 dark:border-slate-800" />

            {sources.map((src) => {
              const isSelected = value === src.id;
              const count = sourceCounts.get(src.id) ?? 0;
              return (
                <button
                  key={src.id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => selectOption(src.id)}
                  className={`flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm transition ${
                    isSelected
                      ? "bg-sky-50 font-medium text-sky-700 dark:bg-sky-500/15 dark:text-sky-300"
                      : "text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
                  }`}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <Check className={`h-3.5 w-3.5 shrink-0 ${isSelected ? "opacity-100" : "opacity-0"}`} />
                    <span className="truncate">{src.name}</span>
                  </span>
                  <span className="shrink-0 text-xs tabular-nums text-slate-400 dark:text-slate-500">
                    {count}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export function App() {
  const [theme, setTheme] = useState<Theme>(readInitialTheme);
  const isDark = theme === "dark";

  const [payload, setPayload] = useState<ArchivePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSource, setSelectedSource] = useState("all");
  const [selectedTopic, setSelectedTopic] = useState<TopicFilter>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [referencesOpen, setReferencesOpen] = useState(false);

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
  }, [searchQuery, selectedSource, selectedTopic]);

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

  const trimmedQuery = searchQuery.trim();

  const visiblePosts = useMemo(
    () => (payload?.posts ?? []).filter((post) => !isPostFiltered(post)),
    [payload?.posts],
  );

  const droppedPosts = useMemo(
    () => (payload?.posts ?? []).filter(isPostFiltered),
    [payload?.posts],
  );

  const visibleSourceCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const post of visiblePosts) {
      counts.set(post.source_id, (counts.get(post.source_id) ?? 0) + 1);
    }
    return counts;
  }, [visiblePosts]);

  const droppedSourceCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const post of droppedPosts) {
      counts.set(post.source_id, (counts.get(post.source_id) ?? 0) + 1);
    }
    return counts;
  }, [droppedPosts]);

  const visibleTopicCounts = useMemo(() => {
    const counts = new Map<PillarId, number>();
    for (const post of visiblePosts) {
      if (selectedSource !== "all" && post.source_id !== selectedSource) continue;
      for (const topic of post.topics ?? []) {
        counts.set(topic, (counts.get(topic) ?? 0) + 1);
      }
    }
    return counts;
  }, [visiblePosts, selectedSource]);

  const filteredTopicCount = useMemo(() => {
    if (selectedSource === "all") return droppedPosts.length;
    return droppedSourceCounts.get(selectedSource) ?? 0;
  }, [droppedPosts.length, droppedSourceCounts, selectedSource]);

  const searchIndex = useMemo(
    () =>
      createSearchIndex(payload?.posts ?? [], ["title", "summary", "source_name"], {
        title: 3,
        source_name: 2,
        summary: 1,
      }),
    [payload?.posts],
  );

  const referenceIndex = useMemo(
    () =>
      createSearchIndex(PINNED_REFERENCES, ["title", "summary", "author"], {
        title: 3,
        author: 2,
        summary: 1,
      }),
    [],
  );

  const filteredReferences = useMemo(() => {
    if (selectedSource !== "all" || selectedTopic !== "all") return [];
    if (!trimmedQuery) return PINNED_REFERENCES;
    return querySearchIndex(referenceIndex, trimmedQuery);
  }, [referenceIndex, selectedSource, selectedTopic, trimmedQuery]);

  const filteredPosts = useMemo(() => {
    if (!payload) return [];
    const base = trimmedQuery
      ? querySearchIndex(searchIndex, trimmedQuery)
      : selectedTopic === "filtered"
        ? droppedPosts
        : visiblePosts;
    return base.filter((post) => {
      if (selectedSource !== "all" && post.source_id !== selectedSource) return false;
      if (selectedTopic === "filtered") return isPostFiltered(post);
      if (isPostFiltered(post)) return false;
      if (selectedTopic === "all") return true;
      return (post.topics ?? []).includes(selectedTopic);
    });
  }, [droppedPosts, payload, searchIndex, selectedSource, selectedTopic, trimmedQuery, visiblePosts]);

  const totalPages = Math.max(1, Math.ceil(filteredPosts.length / PAGE_SIZE));
  const safePage = Math.min(currentPage, totalPages);
  const startIdx = (safePage - 1) * PAGE_SIZE;
  const showReferencesSection =
    filteredReferences.length > 0 && (Boolean(trimmedQuery) || safePage === 1);
  const referencesExpanded = referencesOpen || Boolean(trimmedQuery);
  const referenceGridCols =
    filteredReferences.length === 1
      ? "grid-cols-1"
      : filteredReferences.length === 2
        ? "grid-cols-1 md:grid-cols-2 md:divide-x md:divide-y-0"
        : "grid-cols-1 md:grid-cols-3 md:divide-x md:divide-y-0";
  const referenceHeaderLabel = (
    <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-0.5">
      <div className="flex items-center gap-2">
        <Bookmark className="h-4 w-4 shrink-0 text-sky-600 dark:text-sky-400" />
        <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          Evergreen
        </span>
        <span className="rounded-full bg-sky-500/12 px-2 py-0.5 text-[11px] font-semibold text-sky-700 tabular-nums dark:bg-sky-400/12 dark:text-sky-300">
          {filteredReferences.length}
        </span>
      </div>
      <span className="text-xs text-slate-500 dark:text-slate-400">
        Timeless guides &amp; deep dives
      </span>
    </div>
  );

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

  const selectTopic = (topicId: Exclude<TopicFilter, "all">) => {
    setSelectedTopic((prev) => (prev === topicId ? "all" : topicId));
  };

  const clearFilters = () => {
    setSearchQuery("");
    setSelectedSource("all");
    setSelectedTopic("all");
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

          <div className="mt-2.5 flex flex-wrap items-center gap-2 sm:flex-nowrap sm:gap-2.5">
            {/* Search Box */}
            <div className="relative min-w-[200px] flex-1">
              <Search className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                ref={searchInputRef}
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter posts (press / to focus)"
                className="h-10 w-full rounded-xl border border-slate-200 bg-slate-50/70 pr-9 pl-9 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-sky-500/20 dark:border-slate-700/80 dark:bg-slate-950/70 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-sky-400 dark:focus:bg-slate-950"
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
            <SourceDropdown
              value={selectedSource}
              totalCount={selectedTopic === "filtered" ? droppedPosts.length : visiblePosts.length}
              sources={sortedSources}
              sourceCounts={selectedTopic === "filtered" ? droppedSourceCounts : visibleSourceCounts}
              onChange={setSelectedSource}
            />
          </div>

          <div
            role="group"
            aria-label="Filter by topic pillar"
            className="mt-2 flex flex-wrap items-center gap-1.5"
          >
            <button
              type="button"
              onClick={() => setSelectedTopic("all")}
              aria-pressed={selectedTopic === "all"}
              className={`cursor-pointer rounded-lg border px-2.5 py-1 text-xs font-medium transition ${
                selectedTopic === "all"
                  ? "border-sky-500/45 bg-sky-500/12 text-sky-700 dark:border-sky-400/45 dark:bg-sky-400/15 dark:text-sky-300"
                  : "border-slate-200 bg-white text-slate-600 hover:border-sky-400 hover:text-slate-900 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400 dark:hover:border-sky-600 dark:hover:text-slate-200"
              }`}
            >
              All Topics
            </button>
            {TOPIC_PILLARS.map((pillar) => {
              const active = selectedTopic === pillar.id;
              const count = visibleTopicCounts.get(pillar.id) ?? 0;
              return (
                <button
                  key={pillar.id}
                  type="button"
                  onClick={() => selectTopic(pillar.id)}
                  aria-pressed={active}
                  className={`cursor-pointer rounded-lg border px-2.5 py-1 text-xs font-medium transition ${
                    active
                      ? "border-sky-500/45 bg-sky-500/12 text-sky-700 dark:border-sky-400/45 dark:bg-sky-400/15 dark:text-sky-300"
                      : "border-slate-200 bg-white text-slate-600 hover:border-sky-400 hover:text-slate-900 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400 dark:hover:border-sky-600 dark:hover:text-slate-200"
                  }`}
                >
                  {pillar.label}{" "}
                  <span className="text-[11px] opacity-75 tabular-nums">({count})</span>
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => selectTopic("filtered")}
              aria-pressed={selectedTopic === "filtered"}
              className={`cursor-pointer rounded-lg border px-2.5 py-1 text-xs font-medium transition ${
                selectedTopic === "filtered"
                  ? "border-amber-500/45 bg-amber-500/12 text-amber-800 dark:border-amber-400/45 dark:bg-amber-400/15 dark:text-amber-300"
                  : "border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:text-slate-800 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400 dark:hover:border-slate-700 dark:hover:text-slate-200"
              }`}
            >
              Filtered{" "}
              <span className="text-[11px] opacity-75 tabular-nums">({filteredTopicCount})</span>
            </button>
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

        {/* Evergreen */}
        {!loading && !errorMsg && showReferencesSection && (
          <section
            aria-label="Evergreen"
            className="mb-5 overflow-hidden rounded-2xl border border-slate-200/90 bg-white/90 shadow-xs dark:border-slate-800/90 dark:bg-slate-900/75"
          >
            {trimmedQuery ? (
              <div className="flex items-center px-4 py-3 sm:px-5">
                {referenceHeaderLabel}
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setReferencesOpen((prev) => !prev)}
                aria-expanded={referencesOpen}
                className="flex w-full cursor-pointer items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-sky-50/45 sm:px-5 dark:hover:bg-slate-800/40"
              >
                {referenceHeaderLabel}
                <span className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600 dark:border-slate-700/80 dark:bg-slate-800/80 dark:text-slate-300">
                  <span>{referencesOpen ? "Collapse" : "Expand"}</span>
                  {referencesOpen ? (
                    <ChevronUp className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5" />
                  )}
                </span>
              </button>
            )}

            {referencesExpanded && (
              <div
                className={`grid divide-y divide-slate-200/80 border-t border-slate-200/80 dark:divide-slate-800/80 dark:border-slate-800/80 ${referenceGridCols}`}
              >
                {filteredReferences.map((item) => (
                  <ReferenceCard key={item.id} item={item} />
                ))}
              </div>
            )}
          </section>
        )}

        {/* Empty State */}
        {!loading && !errorMsg && filteredPosts.length === 0 && !showReferencesSection && (
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
          {isPostFiltered(post) ? (
            <span className="rounded-full bg-amber-500/12 px-2 py-0.5 text-[10px] font-medium text-amber-800 dark:bg-amber-400/15 dark:text-amber-300">
              Filtered
            </span>
          ) : (
            (post.topics ?? []).map((topicId) => (
              <span
                key={topicId}
                className="rounded-full bg-sky-500/12 px-2 py-0.5 text-[10px] font-medium text-sky-700 dark:bg-sky-400/12 dark:text-sky-300"
              >
                {TOPIC_LABEL_MAP.get(topicId) ?? topicId}
              </span>
            ))
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

function ReferenceCard({ item }: { item: PinnedReference }) {
  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex flex-col px-4 py-3.5 transition hover:bg-sky-50/45 sm:px-5 dark:hover:bg-slate-800/40"
    >
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="truncate font-semibold text-sky-700 dark:text-sky-400">
          {item.author}
        </span>
        <ExternalLink className="h-3.5 w-3.5 shrink-0 text-slate-400 opacity-0 transition group-hover:opacity-70 dark:text-slate-500" />
      </div>
      <h2 className="mt-1 line-clamp-2 text-[15px] leading-snug font-semibold text-slate-900 transition group-hover:text-sky-700 dark:text-slate-100 dark:group-hover:text-sky-300">
        {item.title}
      </h2>
      <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
        {item.summary}
      </p>
    </a>
  );
}

