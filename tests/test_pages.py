from notion_agent.api.pages import PageLoader


def test_live_then_cache_then_fallback():
    calls = {"live": 0}
    t = [0.0]

    def live(pid):
        calls["live"] += 1
        if pid in ("down", "missing"):
            raise RuntimeError("notion down")
        return {"page_id": pid, "blocks": [{"id": "b1", "type": "paragraph", "text": "x"}]}

    def index(pid):
        return {"page_id": pid, "blocks": []} if pid == "down" else None

    pl = PageLoader(live, index, clock=lambda: t[0])
    assert pl.get("p")["source"] == "notion"
    pl.get("p")
    assert calls["live"] == 1  # cached
    t[0] = 1000
    pl.get("p")
    assert calls["live"] == 2  # expired
    assert pl.get("down") is None  # live failure: UI keeps the index version it already has
    assert pl.get("down", live=False)["source"] == "index"
    assert pl.get("missing") is None
    assert PageLoader(None, index).get("down")["source"] == "index"  # no token -> index
