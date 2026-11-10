"""Image selection must reject anything not clearly tied to the cited text."""

from notion_agent.qa.images import MAX_IMAGES, find_candidates, select_images


def p(i, text):
    return {"id": i, "type": "paragraph",
            "paragraph": {"rich_text": [{"plain_text": text}]}}


def h(i, text):
    return {"id": i, "type": "heading_2", "heading_2": {"rich_text": [{"plain_text": text}]}}


def img(i, caption=""):
    cap = [{"plain_text": caption}] if caption else []
    return {"id": i, "type": "image", "image": {"type": "file", "caption": cap,
                                                 "file": {"url": "https://x/y.png"}}}


class Overlap:
    def score(self, q, passages):
        qs = set(q.lower().split())
        return [len(qs & set(x.lower().split())) / max(len(qs), 1) for x in passages]


BLOCKS = [
    h("h1", "GPU OOM"),
    p("p1", "batch size reduced to 8 to fix OOM"),
    img("i1", "nvidia-smi memory after batch size change"),   # near cited, same section
    p("p2", "unrelated note"),
    h("h2", "Other topic"),
    img("i2", "loss curve"),                                    # other section
    p("p3", "far away text"),
    img("i3"),                                                  # no caption, context only
]


def test_candidates_respect_distance_and_section():
    c = find_candidates(BLOCKS, {"p1"}, "pg", 1)
    assert [x.block_id for x in c] == ["i1"]  # i2 behind a heading, i3 too far


def test_image_without_caption_or_context_is_dropped():
    assert find_candidates([img("lone")], {"lone"}, "pg", 1) == []  # nothing around it
    thin = [p("a", "gemma4-e4b"), img("lone")]  # one-word neighbour: too little to judge
    assert find_candidates(thin, {"a"}, "pg", 1) == []
    rich = [p("a", "극성분리와 스케일분리로 구분되는지 확인한 이미지"), img("lone")]
    assert [x.block_id for x in find_candidates(rich, {"a"}, "pg", 1)] == ["lone"]


def test_relevance_threshold_and_insert_position():
    c = find_candidates(BLOCKS, {"p1"}, "pg", 1)
    sents = [{"text": "결론", "citations": [2]},
             {"text": "batch size reduced to 8 to fix OOM", "citations": [1]}]
    out = select_images(c, sents, Overlap())
    assert len(out) == 1 and out[0].after_block == 1 and out[0].score >= 0.5
    unrelated = [{"text": "kubernetes ingress config", "citations": [1]}]
    assert select_images(find_candidates(BLOCKS, {"p1"}, "pg", 1), unrelated, Overlap()) == []
    nobody_cites = [{"text": "batch size reduced to 8", "citations": [2]}]
    assert select_images(find_candidates(BLOCKS, {"p1"}, "pg", 1), nobody_cites, Overlap()) == []


def test_at_most_max_images():
    blocks = [p("c", "gpu memory batch size")] + [img(f"i{k}", "gpu memory batch size")
                                                   for k in range(1, 3)]
    blocks = [img("i0", "gpu memory batch size")] + blocks
    c = find_candidates(blocks, {"c"}, "pg", 1)
    out = select_images(c, [{"text": "gpu memory batch size", "citations": [1]}], Overlap())
    assert len(c) == 3 and len(out) == MAX_IMAGES


def test_nested_images_inside_cited_toggle():
    from notion_agent.qa.images import find_nested_candidates

    toggle = {"id": "t", "type": "toggle", "has_children": True,
              "toggle": {"rich_text": [{"plain_text": "VLM torn frame 원인"}]}}
    kids = {"t": [p("k1", "ring buffer torn frame capture"), img("ki", ""),
                  {"id": "inner", "type": "toggle", "has_children": True,
                   "toggle": {"rich_text": [{"plain_text": "more"}]}}],
            "inner": [img("deep", "second capture")]}
    c = find_nested_candidates(toggle, lambda b: kids[b], "pg", 1, "H")
    assert [x.block_id for x in c] == ["ki", "deep"]
    assert "VLM torn frame 원인" in c[0].context and c[0].heading == "H"
    sents = [{"text": "torn frame in ring buffer capture", "citations": [1]}]
    assert [x.block_id for x in select_images(c, sents, Overlap())] == ["ki"]
