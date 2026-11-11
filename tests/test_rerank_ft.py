from notion_agent.rag.factory import _adapter
from training.rerank_ft import paired, score_key


def test_paired_counts_moves_and_treats_missing_as_worst():
    base = {"_ranks": {"test": [1, 3, None, 2], "eval40": [1]}}
    ft = {"_ranks": {"test": [1, 1, 5, None], "eval40": [2]}}
    p = paired(base, ft)
    assert p["test"] == {"wins": 2, "losses": 1} and p["eval40"] == {"wins": 0, "losses": 1}


def test_score_key_averages_both_sets():
    assert score_key({"test": {"MRR": 0.8}, "eval40": {"MRR": 0.6}}) == 0.7


def test_missing_adapter_falls_back_to_base():
    assert _adapter("") == "" and _adapter("data/does-not-exist") == ""
