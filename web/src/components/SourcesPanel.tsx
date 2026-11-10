import { useMemo, useState } from "react";
import type { Health, PageSource, SyncStatus } from "../types";
import { relTime } from "../util";

interface Props {
  sources: PageSource[];
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  health: Health | null;
  sync: SyncStatus | null;
  onSync: () => void;
  loadError: string | null;
}

export default function SourcesPanel({ sources, selected, setSelected, health, sync, onSync, loadError }: Props) {
  const [q, setQ] = useState("");
  const filtered = useMemo(() => {
    const k = q.trim().toLowerCase();
    if (!k) return sources;
    return sources.filter((s) =>
      [s.title, ...s.path, ...s.tags].some((t) => t.toLowerCase().includes(k)),
    );
  }, [sources, q]);

  const groups = useMemo(() => {
    const m = new Map<string, PageSource[]>();
    for (const s of filtered) {
      const key = s.path.length ? s.path.join(" / ") : "(최상위)";
      m.set(key, [...(m.get(key) ?? []), s]);
    }
    return [...m.entries()];
  }, [filtered]);

  const allSelected = filtered.length > 0 && filtered.every((s) => selected.has(s.page_id));
  const toggleMany = (ids: string[], on: boolean) => {
    const next = new Set(selected);
    ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
    setSelected(next);
  };

  const running = sync?.running ?? health?.syncing ?? false;
  const lastSync = sync?.last_sync ?? health?.last_sync ?? null;

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-stone-200 p-4 dark:border-stone-800">
        <h2 className="text-sm font-semibold">소스</h2>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="페이지 검색"
          className="mt-3 w-full rounded-lg border border-stone-200 bg-stone-50 px-3 py-1.5 text-sm outline-none focus:border-indigo-400 dark:border-stone-700 dark:bg-stone-800"
        />
        <label className="mt-3 flex cursor-pointer items-center gap-2 text-sm text-stone-600 dark:text-stone-300">
          <input
            type="checkbox"
            className="accent-indigo-600"
            checked={allSelected}
            onChange={(e) => toggleMany(filtered.map((s) => s.page_id), e.target.checked)}
          />
          모두 선택
          <span className="ml-auto text-xs text-stone-400">
            {selected.size}/{sources.length}
          </span>
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {loadError && <p className="px-2 text-xs text-rose-600">{loadError}</p>}
        {!loadError && sources.length === 0 && (
          <p className="px-2 py-4 text-xs text-stone-400">
            {health?.ready ? "색인된 페이지가 없습니다. 동기화를 실행하세요." : "불러오는 중…"}
          </p>
        )}
        {groups.map(([group, items]) => {
          const ids = items.map((s) => s.page_id);
          const all = ids.every((id) => selected.has(id));
          return (
            <div key={group} className="mb-2">
              <label className="flex cursor-pointer items-center gap-2 px-2 py-1 text-xs font-semibold text-stone-500">
                <input
                  type="checkbox"
                  className="accent-indigo-600"
                  checked={all}
                  onChange={(e) => toggleMany(ids, e.target.checked)}
                />
                <span className="truncate">{group}</span>
              </label>
              {items.map((s) => (
                <label
                  key={s.page_id}
                  className="flex cursor-pointer items-center gap-2 rounded-lg py-1 pl-6 pr-2 text-sm hover:bg-stone-100 dark:hover:bg-stone-800"
                  title={`${s.chunks}개 조각 · ${s.tags.join(", ")}`}
                >
                  <input
                    type="checkbox"
                    className="accent-indigo-600"
                    checked={selected.has(s.page_id)}
                    onChange={(e) => toggleMany([s.page_id], e.target.checked)}
                  />
                  <span className="min-w-0 flex-1 truncate">{s.title}</span>
                  <span className="text-[10px] text-stone-400">{s.chunks}</span>
                </label>
              ))}
            </div>
          );
        })}
      </div>

      <div className="border-t border-stone-200 p-4 text-xs dark:border-stone-800">
        <div className="flex items-center justify-between text-stone-500">
          <span>
            {running ? "동기화 중…" : `동기화: ${relTime(lastSync)}`}
          </span>
          <span>{health ? `${health.pages}페이지 · ${health.chunks}조각` : ""}</span>
        </div>
        {sync && !sync.running && sync.ok === false && (
          <p className="mt-1 text-rose-600">{sync.error ?? `실패: ${sync.summary ?? ""}`}</p>
        )}
        <button
          onClick={onSync}
          disabled={running || !health?.ready}
          className="mt-2 w-full rounded-lg border border-stone-300 px-3 py-1.5 font-medium hover:bg-stone-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800"
        >
          {running ? "동기화 중…" : "지금 동기화"}
        </button>
      </div>
    </div>
  );
}
