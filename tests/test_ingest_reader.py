from notion_fakes import FakeAPIError, FakeNotion, blk, page, rt

from notion_agent.ingest.notion_reader import (
    NotionReader,
    block_to_text,
    extract_tags,
    extract_title,
)


def reader(fake: FakeNotion, roots: list[str] | None = None) -> tuple[NotionReader, list[float]]:
    sleeps: list[float] = []
    return NotionReader(fake, roots, min_interval_s=0, sleep=sleeps.append), sleeps


def test_block_to_text_types():
    assert block_to_text(blk("1", "heading_2", "H")) == "## H"
    assert block_to_text(blk("1", "paragraph", "p")) == "p"
    assert block_to_text(blk("1", "bulleted_list_item", "b")) == "- b"
    assert block_to_text(blk("1", "numbered_list_item", "n"), number=3) == "3. n"
    assert block_to_text(blk("1", "to_do", "t", to_do={"checked": True})) == "- [x] t"
    assert block_to_text(blk("1", "to_do", "t", to_do={"checked": False})) == "- [ ] t"
    code = blk("1", "code", "print(1)", code={"language": "python"})
    assert block_to_text(code) == "```python\nprint(1)\n```"
    assert block_to_text(blk("1", "quote", "q")) == "> q"
    assert block_to_text(blk("1", "callout", "c", callout={"icon": {"emoji": "!"}})) == "> ! c"
    row = {"id": "r", "type": "table_row", "table_row": {"cells": [rt("a"), rt("b|c")]}}
    assert block_to_text(row) == "| a | b\\|c |"
    child = {"id": "c", "type": "child_page", "child_page": {"title": "Sub"}}
    assert "Sub" in block_to_text(child)
    assert block_to_text({"id": "x", "type": "unsupported", "unsupported": {}}) is None
    assert block_to_text(blk("1", "paragraph", "")) is None
    assert block_to_text(blk("1", "paragraph", "f", code=True)) == "f"


def test_title_and_tags():
    p = page(
        "p",
        "Hello",
        props={
            "Tags": {"type": "multi_select", "multi_select": [{"name": "a"}, {"name": "b"}]},
            "Status": {"type": "select", "select": {"name": "done"}},
            "Empty": {"type": "select", "select": None},
        },
    )
    p["properties"]["이름"] = p["properties"].pop("Name")
    assert extract_title(p) == "Hello"
    assert extract_tags(p) == ["a", "b", "done"]


def test_fetch_blocks_pagination_and_nesting():
    fake = FakeNotion(page_size=2)
    fake.children["p"] = [
        blk("h", "heading_1", "Title"),
        blk("n1", "numbered_list_item", "one"),
        blk("n2", "numbered_list_item", "two", children=True),
        blk("t", "toggle", "Toggle", children=True),
        blk("tbl", "table", children=True),
    ]
    fake.children["n2"] = [blk("n2a", "bulleted_list_item", "nested")]
    fake.children["t"] = [blk("t1", "paragraph", "inside"), blk("t2", "paragraph", "more")]
    fake.children["tbl"] = [
        {"id": "r1", "type": "table_row", "table_row": {"cells": [rt("x"), rt("y")]}}
    ]
    r, _ = reader(fake)
    blocks = r.fetch_blocks("p")
    assert [b.block_id for b in blocks] == ["h", "n1", "n2", "t", "tbl"]
    assert blocks[0].heading_level == 1
    assert blocks[2].text == "2. two\n  - nested"
    assert blocks[3].text == "Toggle\ninside\nmore"
    assert blocks[4].text == "| x | y |"
    # 5 top-level blocks with page_size 2 -> 3 requests
    assert sum(c.startswith("children:p:") for c in fake.calls) == 3


def test_enumerate_roots_recursive_with_database():
    fake = FakeNotion()
    fake.page_store["root"] = page("root", "Root")
    fake.page_store["sub"] = page("sub", "Sub")
    fake.children["root"] = [
        {"id": "sub", "type": "child_page", "has_children": True, "child_page": {"title": "Sub"}},
        {"id": "db", "type": "child_database", "has_children": False, "child_database": {}},
    ]
    fake.db_store["db"] = {"id": "db", "title": rt("Tasks"), "data_sources": [{"id": "ds"}]}
    fake.ds_rows["ds"] = [page(f"row{i}", f"Row {i}") for i in range(3)]
    r, _ = reader(fake, ["root"])
    metas = list(r.iter_page_metas())
    assert [m[0]["id"] for m in metas] == ["root", "sub", "row0", "row1", "row2"]
    assert dict((m[0]["id"], m[1]) for m in metas)["row2"] == ["Root", "Tasks"]
    np = r.to_notion_page(*metas[1])
    assert np.path == ["Root"] and np.title == "Sub"


def test_search_when_no_roots():
    fake = FakeNotion()
    for i in range(3):
        fake.page_store[f"p{i}"] = page(f"p{i}", f"P{i}")
    r, _ = reader(fake)
    assert len(list(r.iter_page_metas())) == 3


def test_retry_on_429_respects_retry_after():
    fake = FakeNotion()
    fake.fail_queue = [FakeAPIError(429, {"retry-after": "2"}), FakeAPIError(503)]
    fake.children["p"] = [blk("a", "paragraph", "ok")]
    r, sleeps = reader(fake)
    assert [b.text for b in r.fetch_blocks("p")] == ["ok"]
    assert sleeps == [2.0, 2.0]  # Retry-After, then backoff 2**1


def test_non_retriable_error_raises():
    fake = FakeNotion()
    fake.fail_queue = [FakeAPIError(401)]
    r, _ = reader(fake)
    try:
        r.fetch_blocks("p")
    except FakeAPIError as e:
        assert e.status == 401
    else:
        raise AssertionError("expected error")
