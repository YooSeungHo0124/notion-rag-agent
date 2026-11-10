export type Support = "supported" | "partial" | "unsupported";
export type Confidence = "high" | "medium" | "low" | "none";

export interface Sentence {
  text: string;
  citations: number[];
  support: Support;
  support_score: number;
}

export interface SourceHit {
  chunk_id: string;
  page_id: string;
  title: string;
  path: string[];
  heading_path: string[];
  cite_url: string;
  page_url: string;
  text: string;
  tags: string[];
  last_edited_time: string;
  rerank_score: number;
}

export interface Answer {
  question: string;
  answerable: boolean;
  sentences: Sentence[];
  sources: SourceHit[];
  confidence: Confidence;
  retrieval_score: number;
  supported_ratio: number;
  dropped_citations: number;
  latency_ms: number;
  total_ms?: number;
}

export interface PageSource {
  page_id: string;
  title: string;
  path: string[];
  tags: string[];
  last_edited_time: string | null;
  url: string | null;
  chunks: number;
}

export interface Health {
  ready: boolean;
  error: string | null;
  chunks: number;
  pages: number;
  last_sync: string | null;
  syncing: boolean;
}

export interface SyncStatus {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  ok: boolean | null;
  summary: string | null;
  errors: string[];
  error: string | null;
  last_sync: string | null;
}

export interface SavedNote {
  id: string;
  question?: string;
  text: string;
  confidence?: Confidence;
  savedAt: string;
}

export interface AgentStep {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
  ok: boolean;
  observation: unknown;
  ms: number;
}

export interface AgentRun {
  thread_id: string;
  status: "awaiting_approval" | "done";
  approval: { tool: string; args: Record<string, unknown>; thought: string } | null;
  steps: AgentStep[];
  answer: (Answer & { kind: "answer" }) | { kind: "action"; steps: AgentStep[] } | null;
  done_reason: string | null;
  total_ms: number;
}
