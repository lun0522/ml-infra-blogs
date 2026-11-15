export interface SourceSummary {
  id: string;
  name: string;
  total_count: number;
}

export interface PostRecord {
  id: string;
  source_id: string;
  source_name: string;
  url: string;
  title: string;
  summary: string;
  published_at: string | null;
  first_seen: string;
}

export interface ArchivePayload {
  generated_at: string | null;
  sources: SourceSummary[];
  posts: PostRecord[];
}
