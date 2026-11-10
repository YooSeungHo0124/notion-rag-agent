"""Thin, read-mostly Odoo client over the external XML-RPC API.

Only models/fields in `READ_FIELDS` can be read, so the agent can never pull arbitrary
tables (e.g. res.users, ir.config_parameter). The only write is `log_activity`, which
posts a note + schedules a to-do on a CRM opportunity; the agent graph puts it behind the
human approval gate like every other write tool.
"""

from __future__ import annotations

import xmlrpc.client
from dataclasses import dataclass
from typing import Any

READ_FIELDS: dict[str, list[str]] = {
    "res.partner": ["name", "is_company", "city", "country_id", "email", "phone",
                    "customer_rank", "supplier_rank"],
    "crm.lead": ["name", "partner_id", "stage_id", "expected_revenue", "probability",
                 "user_id", "date_deadline", "type", "write_date", "activity_summary"],
    "sale.order": ["name", "partner_id", "state", "amount_total", "currency_id",
                   "date_order", "commitment_date", "invoice_status", "user_id"],
    "sale.order.line": ["order_id", "product_id", "product_uom_qty", "qty_delivered",
                        "price_unit", "price_subtotal"],
    "purchase.order": ["name", "partner_id", "state", "amount_total", "date_planned",
                       "date_order"],
    "product.product": ["display_name", "default_code", "list_price", "qty_available",
                        "virtual_available", "type"],
    "account.move": ["name", "partner_id", "move_type", "state", "amount_total",
                     "amount_residual", "payment_state", "invoice_date", "invoice_date_due"],
}


class OdooError(RuntimeError):
    pass


@dataclass
class OdooConfig:
    url: str
    db: str
    user: str
    password: str


class OdooClient:
    def __init__(self, cfg: OdooConfig, timeout_s: float = 20) -> None:
        self.cfg = cfg
        self.base = cfg.url.rstrip("/")
        self._timeout = timeout_s
        self._uid: int | None = None

    def _proxy(self, path: str) -> xmlrpc.client.ServerProxy:
        t = _TimeoutTransport(self._timeout)
        return xmlrpc.client.ServerProxy(f"{self.base}{path}", allow_none=True, transport=t)

    @property
    def uid(self) -> int:
        if self._uid is None:
            uid = self._proxy("/xmlrpc/2/common").authenticate(
                self.cfg.db, self.cfg.user, self.cfg.password, {})
            if not uid:
                raise OdooError("Odoo authentication failed")
            self._uid = int(uid)
        return self._uid

    def _call(self, model: str, method: str, args: list[Any], **kw: Any) -> Any:
        try:
            return self._proxy("/xmlrpc/2/object").execute_kw(
                self.cfg.db, self.uid, self.cfg.password, model, method, args, kw)
        except xmlrpc.client.Fault as e:
            raise OdooError(e.faultString.strip().splitlines()[-1][:300]) from e

    # -- reads (allow-listed) ------------------------------------------------------
    def search_read(self, model: str, domain: list[Any], *, limit: int = 20,
                    order: str | None = None, fields: list[str] | None = None) -> list[dict]:
        allowed = READ_FIELDS.get(model)
        if allowed is None:
            raise OdooError(f"model {model!r} is not readable")
        fields = [f for f in (fields or allowed) if f in allowed]
        kw: dict[str, Any] = {"fields": fields, "limit": max(1, min(limit, 80))}
        if order:
            kw["order"] = order
        return self._call(model, "search_read", [domain], **kw)

    def read_group(self, model: str, domain: list[Any], fields: list[str],
                   groupby: list[str]) -> list[dict]:
        if model not in READ_FIELDS:
            raise OdooError(f"model {model!r} is not readable")
        return self._call(model, "read_group", [domain, fields, groupby], lazy=False)

    def record_url(self, model: str, rid: int) -> str:
        return f"{self.base}/web#model={model}&id={rid}&view_type=form" if rid else self.base

    # -- the single write ----------------------------------------------------------
    def log_activity(self, lead_id: int, summary: str, note: str = "",
                     due: str | None = None) -> dict[str, Any]:
        lead = self._call("crm.lead", "read", [[lead_id]], fields=["name"])
        if not lead:
            raise OdooError(f"crm.lead {lead_id} not found")
        model_id = self._call("ir.model", "search", [[("model", "=", "crm.lead")]], limit=1)
        todo = self._call("ir.model.data", "search_read",
                          [[("module", "=", "mail"), ("name", "=", "mail_activity_data_todo")]],
                          fields=["res_id"], limit=1)
        todo = [todo[0]["res_id"]] if todo else self._call(
            "mail.activity.type", "search", [[("category", "=", "default")]], limit=1)
        vals: dict[str, Any] = {"res_model_id": model_id[0], "res_id": lead_id,
                                "summary": summary[:120], "note": note[:2000],
                                "user_id": self.uid}
        if todo:
            vals["activity_type_id"] = todo[0]
        if due:
            vals["date_deadline"] = due
        act = self._call("mail.activity", "create", [vals])
        return {"activity_id": act, "lead": lead[0]["name"],
                "url": self.record_url("crm.lead", lead_id)}


class _TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._t = timeout

    def make_connection(self, host):  # type: ignore[no-untyped-def]
        c = super().make_connection(host)
        c.timeout = self._t
        return c
