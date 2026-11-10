"""ERP/CRM tools for the agent.

Every read tool turns its rows into ONE pseudo source (`Hit`, page_id `erp:<tool>:<n>`)
and drops it in the shared `hit_sink`, so the writer cites ERP rows exactly like note
chunks and the verifier checks every number in the answer against the rows we fetched.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from notion_agent.erp.odoo import OdooClient
from notion_agent.rag.retriever import Hit
from notion_agent.tools.registry import Tool, ToolRegistry

SALE_STATE = {"draft": "견적", "sent": "견적 발송", "sale": "확정", "cancel": "취소"}
PO_STATE = {"draft": "RFQ", "sent": "RFQ 발송", "to approve": "승인 대기", "purchase": "발주",
            "done": "완료", "cancel": "취소"}
PAY_STATE = {"not_paid": "미결제", "partial": "부분 결제", "paid": "결제 완료",
             "in_payment": "결제 중", "reversed": "역분개"}


def _n(v: Any) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _m2o(v: Any) -> str:
    return v[1] if isinstance(v, list | tuple) and len(v) > 1 else ""


def _d(v: Any) -> str:
    return str(v)[:10] if v else "-"


class PartnerArgs(BaseModel):
    name: str = Field(description="거래처(회사) 이름 일부")


class PipelineArgs(BaseModel):
    partner: str = Field("", description="거래처 이름 일부(선택)")
    stage: str = Field("", description="단계 이름 일부(선택), 예: Proposition, Won")
    min_probability: float = Field(0, ge=0, le=100)
    limit: int = Field(20, ge=1, le=50)


class SalesArgs(BaseModel):
    partner: str = Field("", description="거래처 이름 일부(선택)")
    state: str = Field("", description="draft|sent|sale|cancel (선택)")
    with_lines: bool = Field(False, description="주문 품목까지 볼지")
    limit: int = Field(20, ge=1, le=50)


class StockArgs(BaseModel):
    product: str = Field("", description="제품명/코드 일부(선택)")
    below: float | None = Field(None, description="가용 재고가 이 값 미만인 제품만")
    limit: int = Field(25, ge=1, le=60)


class InvoiceArgs(BaseModel):
    partner: str = Field("", description="거래처 이름 일부(선택)")
    unpaid_only: bool = True
    limit: int = Field(20, ge=1, le=50)


class PurchaseArgs(BaseModel):
    partner: str = Field("", description="공급처 이름 일부(선택)")
    limit: int = Field(20, ge=1, le=50)


class ActivityArgs(BaseModel):
    lead_id: int = Field(description="crm.lead id (erp_pipeline 결과의 id)")
    summary: str = Field(description="활동 제목, 예: '견적 재발송'")
    note: str = Field("", description="상세 메모")
    due: str | None = Field(None, description="마감일 YYYY-MM-DD (선택)")


def build_erp_tools(reg: ToolRegistry, odoo: OdooClient, hit_sink: dict[str, Hit] | None) -> None:
    seq = itertools.count(1)

    def source(tool: str, title: str, lines: list[str], url: str) -> str:
        sid = f"erp:{tool}:{next(seq)}"
        if hit_sink is not None and lines:
            hit_sink[sid] = Hit(
                chunk_id=sid, page_id=sid, page_title=f"ERP · {title}", page_url=url,
                path=["ERP (Odoo)"], text="\n".join(lines)[:6000],
                last_edited_time=datetime.now(UTC), fusion_score=0.0, rerank_score=1.0)
        return sid

    def like(field: str, v: str) -> list[Any]:
        return [(field, "ilike", v)] if v else []

    def erp_find_partner(a: PartnerArgs) -> dict[str, Any]:
        rows = odoo.search_read("res.partner", [("name", "ilike", a.name)], limit=10)
        lines = [f"거래처 {r['name']} (id {r['id']}) | 회사:{'예' if r['is_company'] else '아니오'}"
                 f" | 도시:{r['city'] or '-'} | 국가:{_m2o(r['country_id']) or '-'}"
                 f" | 이메일:{r['email'] or '-'}" for r in rows]
        sid = source("partner", f"거래처 검색 '{a.name}'", lines, odoo.base)
        return {"source": sid, "rows": lines}

    def erp_pipeline(a: PipelineArgs) -> dict[str, Any]:
        dom = [("type", "=", "opportunity"), *like("partner_id", a.partner),
               *like("stage_id", a.stage)]
        if a.min_probability:
            dom.append(("probability", ">=", a.min_probability))
        rows = odoo.search_read("crm.lead", dom, limit=a.limit, order="expected_revenue desc")
        lines = [f"영업기회 {r['name']} (id {r['id']}) | 거래처:{_m2o(r['partner_id']) or '-'}"
                 f" | 단계:{_m2o(r['stage_id'])} | 예상매출:{_n(r['expected_revenue'])}"
                 f" | 확률:{_n(r['probability'])}% | 담당:{_m2o(r['user_id']) or '-'}"
                 f" | 마감:{_d(r['date_deadline'])}" for r in rows]
        total = sum(r["expected_revenue"] for r in rows)
        weighted = sum(r["expected_revenue"] * r["probability"] / 100 for r in rows)
        if rows:
            lines.append(f"합계: {len(rows)}건, 예상매출 합 {_n(float(total))},"
                         f" 확률 가중 합 {_n(round(weighted, 1))}")
        sid = source("pipeline", "CRM 영업기회" + (f" ({a.partner})" if a.partner else ""),
                     lines, odoo.base)
        return {"source": sid, "rows": lines}

    def erp_sales(a: SalesArgs) -> dict[str, Any]:
        dom = [*like("partner_id", a.partner)] + ([("state", "=", a.state)] if a.state else [])
        rows = odoo.search_read("sale.order", dom, limit=a.limit, order="date_order desc")
        lines = []
        for r in rows:
            lines.append(f"판매주문 {r['name']} (id {r['id']}) | 거래처:{_m2o(r['partner_id'])}"
                         f" | 상태:{SALE_STATE.get(r['state'], r['state'])}"
                         f" | 금액:{_n(r['amount_total'])} {_m2o(r['currency_id'])}"
                         f" | 주문일:{_d(r['date_order'])} | 청구:{r['invoice_status']}")
        if a.with_lines and rows:
            ol = odoo.search_read("sale.order.line", [("order_id", "in", [r["id"] for r in rows])],
                                  limit=80)
            for x in ol:
                lines.append(f"  - {_m2o(x['order_id'])}: {_m2o(x['product_id'])}"
                             f" 수량 {_n(x['product_uom_qty'])}, 출고 {_n(x['qty_delivered'])},"
                             f" 단가 {_n(x['price_unit'])}, 소계 {_n(x['price_subtotal'])}")
        if rows:
            lines.append(f"합계: {len(rows)}건, 금액 합 {_n(float(sum(r['amount_total'] for r in rows)))}")
        sid = source("sales", "판매주문" + (f" ({a.partner})" if a.partner else ""), lines,
                     odoo.base)
        return {"source": sid, "rows": lines}

    def erp_stock(a: StockArgs) -> dict[str, Any]:
        dom: list[Any] = [("type", "=", "consu")]
        if a.product:
            dom += ["|", ("name", "ilike", a.product), ("default_code", "ilike", a.product)]
        # qty_available is computed (not stored): filter/sort client-side
        rows = odoo.search_read("product.product", dom, limit=80, order="default_code")
        if a.below is not None:
            rows = [r for r in rows if r["qty_available"] < a.below]
        rows = sorted(rows, key=lambda r: r["qty_available"])[: a.limit]
        lines = [f"제품 {r['display_name']} (id {r['id']}) | 가용재고:{_n(r['qty_available'])}"
                 f" | 예측재고:{_n(r['virtual_available'])} | 판매가:{_n(r['list_price'])}"
                 for r in rows]
        sid = source("stock", "재고" + (f" ({a.product})" if a.product else ""), lines, odoo.base)
        return {"source": sid, "rows": lines}

    def erp_invoices(a: InvoiceArgs) -> dict[str, Any]:
        dom = [("move_type", "=", "out_invoice"), ("state", "=", "posted"),
               *like("partner_id", a.partner)]
        if a.unpaid_only:
            dom.append(("payment_state", "in", ["not_paid", "partial"]))
        rows = odoo.search_read("account.move", dom, limit=a.limit, order="invoice_date_due asc")
        lines = [f"송장 {r['name']} (id {r['id']}) | 거래처:{_m2o(r['partner_id'])}"
                 f" | 금액:{_n(r['amount_total'])} | 미수금:{_n(r['amount_residual'])}"
                 f" | 결제:{PAY_STATE.get(r['payment_state'], r['payment_state'])}"
                 f" | 발행:{_d(r['invoice_date'])} | 만기:{_d(r['invoice_date_due'])}" for r in rows]
        if rows:
            lines.append(f"합계: {len(rows)}건, 미수금 합 {_n(float(sum(r['amount_residual'] for r in rows)))}")
        sid = source("invoices", "고객 송장" + (f" ({a.partner})" if a.partner else ""), lines,
                     odoo.base)
        return {"source": sid, "rows": lines}

    def erp_purchases(a: PurchaseArgs) -> dict[str, Any]:
        rows = odoo.search_read("purchase.order", like("partner_id", a.partner), limit=a.limit,
                                order="date_order desc")
        lines = [f"구매주문 {r['name']} (id {r['id']}) | 공급처:{_m2o(r['partner_id'])}"
                 f" | 상태:{PO_STATE.get(r['state'], r['state'])} | 금액:{_n(r['amount_total'])}"
                 f" | 입고예정:{_d(r['date_planned'])}" for r in rows]
        sid = source("purchases", "구매주문" + (f" ({a.partner})" if a.partner else ""), lines,
                     odoo.base)
        return {"source": sid, "rows": lines}

    def erp_log_activity(a: ActivityArgs) -> dict[str, Any]:
        return odoo.log_activity(a.lead_id, a.summary, a.note, a.due)

    for name, desc, args, fn in [
        ("erp_find_partner", "ERP 거래처(고객/공급처) 검색.", PartnerArgs, erp_find_partner),
        ("erp_pipeline", "CRM 영업기회: 단계·예상매출·확률·담당자. 거래처/단계로 필터.",
         PipelineArgs, erp_pipeline),
        ("erp_sales", "판매주문(견적/확정): 금액·상태·주문일, 선택 시 품목.", SalesArgs, erp_sales),
        ("erp_stock", "제품 재고(가용/예측)와 판매가. below로 부족 품목만.", StockArgs, erp_stock),
        ("erp_invoices", "고객 송장과 미수금(기본: 미결제만).", InvoiceArgs, erp_invoices),
        ("erp_purchases", "구매주문(공급처 발주)과 입고 예정일.", PurchaseArgs, erp_purchases),
    ]:
        reg.register(Tool(name, desc, args, fn))
    reg.register(Tool("erp_log_activity", "CRM 영업기회에 후속 활동(할 일) 등록.", ActivityArgs,
                      erp_log_activity, write=True))
