import type { Answer, Health, PageSource, SyncStatus } from "./types";

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
  sync: () => req<SyncStatus>("/api/sync", { method: "POST", body: "{}" }),
  syncStatus: () => req<SyncStatus>("/api/sync/status"),
};
