from rag_fakes import make_page

from notion_agent.config import ChunkingConfig
from notion_agent.rag.chunker import chunk_page

CFG = ChunkingConfig(max_chars=200, overlap_chars=30)


def _page():
    return make_page(
        "p1",
        "Title",
        [
            ("paragraph", "intro text", None),
            ("heading_1", "H1", 1),
            ("paragraph", "alpha " * 20, None),
            ("heading_2", "H2", 2),
            ("paragraph", "beta " * 30, None),
            ("paragraph", "gamma " * 25, None),
            ("heading_2", "H2b", 2),
            ("code", "x = 1\n" * 30, None),
            ("heading_1", "Other", 1),
            ("paragraph", "omega", None),
        ],
    )


def test_heading_paths_and_prefix():
    chunks = chunk_page(_page(), CFG)
    paths = [c.heading_path for c in chunks]
    assert paths[0] == []
    assert ["H1"] in paths and ["H1", "H2"] in paths and ["H1", "H2b"] in paths
    assert paths[-1] == ["Other"]
    c = next(c for c in chunks if c.heading_path == ["H1", "H2"])
    assert c.context_text.startswith("[Title > Root > H1 > H2 | 2026-01-01]\n")
    assert all(len(c.text) <= CFG.max_chars for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_packing_overlap_and_block_ids():
    chunks = [c for c in chunk_page(_page(), CFG) if c.heading_path == ["H1", "H2"]]
    assert len(chunks) == 2
    assert chunks[0].block_ids == ["p1-b3", "p1-b4"]
    assert chunks[1].block_ids == ["p1-b5"]  # overlap does not add provenance
    assert "beta" in chunks[1].text and "gamma" in chunks[1].text  # overlap carried


def test_code_block_kept_whole_unless_oversized():
    chunks = chunk_page(_page(), CFG)
    code = [c for c in chunks if c.heading_path == ["H1", "H2b"]]
    assert len(code) == 1 and code[0].text.count("x = 1") == 30
    small = make_page(
        "p2", "T", [("paragraph", "a " * 80, None), ("code", "print(1)\n" * 10, None)]
    )
    cs = chunk_page(small, CFG)
    assert any(c.text.endswith("print(1)\n" * 10) or c.text == "print(1)\n" * 10 for c in cs)
    assert sum(c.text.count("print(1)") for c in cs) == 10  # never split, no overlap dup
    big = make_page("p3", "T", [("code", "line\n" * 100, None)])
    parts = chunk_page(big, CFG)
    assert len(parts) > 1 and "".join(p.text for p in parts) == "line\n" * 100


def test_ids_deterministic():
    a = chunk_page(_page(), CFG)
    b = chunk_page(_page(), CFG)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_toggle_heading_label_is_first_line_only():
    from notion_agent.rag.chunker import _heading_label

    assert _heading_label("### `Version 2`\nargs.yaml\n```\ntask: detect\n```") == "`Version 2`"
    assert len(_heading_label("## " + "가" * 200)) == 80
