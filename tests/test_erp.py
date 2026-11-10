from pathlib import Path

import pytest

from notion_agent.erp.odoo import OdooClient, OdooConfig, OdooError
from notion_agent.erp.tools import build_erp_tools
from notion_agent.ingest.local_docs import load_docs
from notion_agent.tools.registry import ToolRegistry


class FakeOdoo(OdooClient):
    def __init__(self):
        super().__init__(OdooConfig("http://erp", "db", "u", "p"))
        self.calls = []
        self.created = []
        self._uid = 2

    def _call(self, model, method, args, **kw):
        self.calls.append((model, method, args, kw))
        if model == "crm.lead" and method == "search_read":
            return [{"id": 19, "name": "Desk", "partner_id": [15, "Azure"],
                     "stage_id": [3, "Proposition"], "expected_revenue": 15000.0,
                     "probability": 65.5, "user_id": [2, "Marc"], "date_deadline": False}]
        if model == "product.product":
            return [{"id": 1, "display_name": "Chair", "qty_available": 3.0,
                     "virtual_available": 3.0, "list_price": 70.0},
                    {"id": 2, "display_name": "Desk", "qty_available": 40.0,
                     "virtual_available": 40.0, "list_price": 300.0}]
        if model == "crm.lead" and method == "read":
            return [{"name": "Desk"}]
        if method == "search":
            return [7]
        if model == "mail.activity" and method == "create":
            self.created.append(args[0])
            return 99
        return []


@pytest.fixture
def setup():
    odoo, sink, reg = FakeOdoo(), {}, ToolRegistry()
    build_erp_tools(reg, odoo, sink)
    return odoo, sink, reg


def test_pipeline_becomes_citable_source_with_totals(setup):
    odoo, sink, reg = setup
    r = reg.run("erp_pipeline", {"partner": "Azure"})
    assert r.ok
    (hit,) = sink.values()
    assert hit.page_id.startswith("erp:pipeline:") and hit.cite_url == "http://erp"
    assert "예상매출:15000" in hit.text and "확률 가중 합 9825" in hit.text
    domain = odoo.calls[0][2][0]
    assert ("partner_id", "ilike", "Azure") in domain and ("type", "=", "opportunity") in domain


def test_stock_filters_computed_field_client_side(setup):
    _, sink, reg = setup
    out = reg.run("erp_stock", {"below": 5}).output
    assert len(out["rows"]) == 1 and "Chair" in out["rows"][0]


def test_only_allowlisted_models_and_fields_are_read():
    odoo = FakeOdoo()
    with pytest.raises(OdooError):
        odoo.search_read("res.users", [])
    odoo.search_read("res.partner", [], fields=["name", "password"])
    assert odoo.calls[-1][3]["fields"] == ["name"]


def test_activity_write_is_approval_gated_tool(setup):
    odoo, _, reg = setup
    assert reg.get("erp_log_activity").write
    assert not any(t.write for n, t in reg.tools.items() if n != "erp_log_activity")
    out = reg.run("erp_log_activity", {"lead_id": 19, "summary": "견적 재발송"}).output
    assert out["activity_id"] == 99 and odoo.created[0]["res_id"] == 19


def test_company_docs_parse_to_pages():
    root = Path(__file__).resolve().parents[1] / "scenario" / "docs"
    pages = {p.page_id: p for p in load_docs(root)}
    assert "doc:README" not in pages and len(pages) >= 8
    p = pages["doc:07_주간_영업회의_2026-11-09"]
    assert p.title.startswith("주간 영업회의") and p.tags == ["company"]
    assert p.last_edited_time.date().isoformat() == "2026-11-09"
    assert any(b.type == "heading_2" for b in p.blocks)
