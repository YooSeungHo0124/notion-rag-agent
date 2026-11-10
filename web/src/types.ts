export type Support = "supported" | "partial" | "unsupported";
export type Confidence = "high" | "medium" | "low" | "none";

export type BlockType =
  | "summary"
  | "heading"
  | "text"
  | "callout"
  | "table"
  | "steps"
  | "code"
  | "chart"
  | "flow";

export interface Sentence {
  text: string;
  citations: number[];
  support: Support;
  support_score: number;
  type?: BlockType;
  data?: {
    columns?: string[];
    rows?: string[][];
    items?: string[];
    lang?: string;
    code?: string;
    kind?: "bar";
    labels?: string[];
    values?: number[];
    unit?: string;
    level?: "note" | "warning" | "conflict";
    nodes?: string[];
    notes?: string[];
  };
  note?: string;
}

export interface Stage {
  name: "retrieving" | "retrieved" | "generating" | "verifying" | "collecting" | "collected" | "draft";
  blocks_draft?: { type: BlockType; text: string; data: Sentence["data"] }[];
  n?: number;
  chunks?: number;
  pages_used?: number;
  pages_total?: number;
  truncated?: boolean;
  top?: number;
  titles?: string[];
  blocks?: number;
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
  block_ids?: string[];
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
  followups?: string[];
  closest?: SourceHit | null;
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

export interface PageBlock {
  id: string;
  type: string;
  text: string;
  level: number | null;
  block_ids?: string[];
}

export interface PageDoc {
  page_id: string;
  title: string;
  url: string;
  path?: string[];
  last_edited_time: string | null;
  blocks: PageBlock[];
  source: "notion" | "index";
}

export interface Turn {
  id: string;
  q: string;
  mode: "ask" | "agent" | "studio";
  studio?: { kind: StudioKind; title: string; scopeLabel: string; topic?: string; page_ids: string[]; coverage?: Coverage };
  mindmap?: MindNode;
  startedAt: number;
  stage?: Stage;
  retrieved?: Stage;
  images?: AnswerImage[];
  draft?: Sentence[];
  answer?: Answer;
  run?: AgentRun;
  error?: string;
  stopped?: boolean;
  doneMs?: number;
}

export interface Conv {
  id: string;
  title: string;
  updatedAt: number;
  turns: Turn[];
  scope?: { label: string; page_ids: string[] } | null;
}

export interface AnswerImage {
  block_id: string;
  page_id: string;
  source: number;
  after_block: number;
  caption: string;
  context: string;
  heading: string;
  score: number;
  src: string;
}

export type StudioKind = "briefing" | "study" | "faq" | "timeline" | "guide" | "mindmap";

export interface Coverage {
  pages_total: number;
  pages_used: number;
  chunks_used: number;
  truncated: boolean;
  topic: string;
}

export interface MindNode {
  label: string;
  citations: number[];
  children: MindNode[];
}

export interface StudioResult {
  kind: StudioKind;
  title: string;
  coverage: Coverage;
  latency_ms: number;
  answer?: Answer | null;
  tree?: MindNode;
  sources?: SourceHit[];
}

export interface Note {
  id: string;
  title: string;
  text: string;
  updatedAt: string;
  fromQ?: string;
}
