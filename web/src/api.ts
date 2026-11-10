import type { AgentRun, Answer, AnswerImage, PageDoc, Stage, Health, PageSource, SyncStatus } from "./types";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export const api = {
  health: () => req<Health>("/api/health"),
  sources: (refresh = false) =>
    req<{ sources: PageSource[] }>(`/api/sources${refresh ? "?refresh=true" : ""}`).then(
      (r) => r.sources,
    ),
  ask: (question: string, page_ids: string[]) =>
    req<Answer>("/api/ask", { method: "POST", body: JSON.stringify({ question, page_ids }) }),
  /** SSE: stage events while the pipeline runs, resolves with the final answer. */
  askStream: async (
    question: string,
    page_ids: string[],
    onStage: (s: Stage) => void,
    signal?: AbortSignal,
  ) => {
    const r = await fetch("/api/ask/stream", {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, page_ids }),
    });
    if (!r.ok || !r.body) throw new Error(`${r.status} ${r.statusText}`);
    const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const ev = /^event: (.*)$/m.exec(frame)?.[1];
        const data = JSON.parse(/^data: (.*)$/m.exec(frame)?.[1] ?? "{}");
        if (ev === "answer") return data as Answer;
        if (ev === "error") throw new Error(data.detail ?? "error");
        if (ev) onStage({ name: ev, ...data } as Stage);
      }
    }
    throw new Error("스트림이 답변 없이 끝났습니다");
  },
  agent: (question: string, thread_id?: string) =>
    req<AgentRun>("/api/agent", { method: "POST", body: JSON.stringify({ question, thread_id }) }),
  resume: (thread_id: string, approved: boolean, args?: Record<string, unknown>) =>
    req<AgentRun>("/api/agent/resume", {
      method: "POST",
      body: JSON.stringify({ thread_id, approved, args: args ?? null }),
    }),
  answerImages: (a: Answer) =>
    req<{ images: AnswerImage[] }>("/api/answer/images", {
      method: "POST",
      body: JSON.stringify({
        sources: a.sources.map((h) => ({ page_id: h.page_id, block_ids: h.block_ids ?? [] })),
        sentences: a.sentences.map((x) => ({ text: x.text, citations: x.citations, type: x.type })),
      }),
    }).then((r) => r.images),
  page: (page_id: string, live: boolean) =>
    req<PageDoc>(`/api/page/${encodeURIComponent(page_id)}?live=${live ? 1 : 0}`),
  sync: () => req<SyncStatus>("/api/sync", { method: "POST", body: "{}" }),
  syncStatus: () => req<SyncStatus>("/api/sync/status"),
};
