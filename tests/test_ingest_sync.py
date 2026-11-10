from notion_fakes import FakeNotion, blk, page

from notion_agent.ingest.models import NotionPage
from notion_agent.ingest.notion_reader import NotionReader
from notion_agent.ingest.state import SyncState
from notion_agent.ingest.sync import sync


class RecSink:
    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.upserts: list[str] = []
        self.deletes: list[str] = []
        self.fail_on = fail_on or set()

    def upsert_page(self, p: NotionPage) -> int:
        if p.page_id in self.fail_on:
            raise RuntimeError("boom")
        self.upserts.append(p.page_id)
        return len(p.blocks)

    def delete_page(self, page_id: str) -> None:
        self.deletes.append(page_id)


def setup():
    fake = FakeNotion()
    for pid in ("a", "b", "c"):
        fake.page_store[pid] = page(pid, pid.upper())
        fake.children[pid] = [blk(f"{pid}1", "paragraph", f"text {pid}")]
    return (
        fake,
        NotionReader(fake, [], min_interval_s=0, sleep=lambda _: None),
        SyncState(":memory:"),
    )


def test_first_then_unchanged():
    fake, r, st = setup()
    s1 = RecSink()
    rep = sync(r, s1, st)
    assert (rep.scanned, rep.changed, rep.chunks) == (3, 3, 3)
    assert st.known_ids() == {"a", "b", "c"}
    s2 = RecSink()
    rep = sync(r, s2, st)
    assert (rep.changed, rep.unchanged) == (0, 3) and s2.upserts == []


def test_edited_reprocessed_and_deleted_removed():
    fake, r, st = setup()
    sync(r, RecSink(), st)
    fake.page_store["a"]["last_edited_time"] = "2026-02-01T00:00:00.000Z"
    fake.children["a"] = [blk("a1", "paragraph", "new text")]
    del fake.page_store["b"]
    fake.page_store["c"]["archived"] = True
    sink = RecSink()
    rep = sync(r, sink, st)
    assert sink.upserts == ["a"]
    assert sorted(sink.deletes) == ["b", "c"]
    assert (rep.changed, rep.deleted) == (1, 2)
    assert st.known_ids() == {"a"}


def test_touched_but_same_content_not_reembedded():
    fake, r, st = setup()
    sync(r, RecSink(), st)
    fake.page_store["a"]["last_edited_time"] = "2026-03-01T00:00:00.000Z"
    sink = RecSink()
    rep = sync(r, sink, st)
    assert sink.upserts == [] and rep.unchanged == 3
    assert st.get("a").last_edited_time.startswith("2026-03-01")


def test_sink_failure_leaves_state_uncommitted():
    fake, r, st = setup()
    rep = sync(r, RecSink(fail_on={"b"}), st)
    assert rep.failed == 1 and rep.changed == 2 and "b" in rep.errors[0]
    assert st.get("b") is None
    sink = RecSink()
    rep = sync(r, sink, st)
    assert sink.upserts == ["b"]


def test_full_and_dry_run():
    fake, r, st = setup()
    sync(r, RecSink(), st)
    sink = RecSink()
    assert sync(r, sink, st, full=True).changed == 3
    st2 = SyncState(":memory:")
    sync(r, RecSink(), st2, commit=False)
    assert st2.known_ids() == set()


def test_enumeration_error_skips_deletions():
    fake, r, st = setup()
    sync(r, RecSink(), st)

    def broken(**_):
        raise RuntimeError("network down")

    fake.search = broken
    sink = RecSink()
    rep = sync(r, sink, st)
    assert sink.deletes == [] and rep.failed == 1
    assert st.known_ids() == {"a", "b", "c"}
