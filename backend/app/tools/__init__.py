"""Agent tools. Read tools execute immediately; write tools are gated by approval."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..db import Approval, Order, Ticket
from ..errors import APIError, ORDER_NOT_FOUND, TICKET_NOT_FOUND


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)
    awaiting_approval: bool = False
    message: str = ""


class BaseTool:
    name = "base"
    description = ""
    needs_approval = False  # write tools set True; executed only after approval

    def execute(self, session, args: dict) -> ToolResult:
        raise NotImplementedError


class OrderLookupTool(BaseTool):
    name = "order_lookup"
    description = "Look up an order by ID (ORD-1234). Read-only."

    def execute(self, session, args: dict) -> ToolResult:
        order_id = str(args.get("order_id", "")).upper()
        order = session.get(Order, order_id)
        if order is None:
            raise APIError(ORDER_NOT_FOUND[0], f"Order {order_id} not found.", ORDER_NOT_FOUND[1])
        return ToolResult(ok=True, data={
            "found": True, "id": order.id, "customer_name": order.customer_name,
            "item": order.item, "total_cents": order.total_cents, "status": order.status,
            "tracking_number": order.tracking_number,
        })


class TicketStatusTool(BaseTool):
    name = "ticket_status"
    description = "Check a support ticket's status by ID (TKT-2001). Read-only."

    def execute(self, session, args: dict) -> ToolResult:
        ticket_id = str(args.get("ticket_id", "")).upper()
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise APIError(TICKET_NOT_FOUND[0], f"Ticket {ticket_id} not found.", TICKET_NOT_FOUND[1])
        return ToolResult(ok=True, data={
            "found": True, "id": ticket.id, "subject": ticket.subject,
            "description": ticket.description, "status": ticket.status,
            "priority": ticket.priority, "order_id": ticket.order_id,
        })


class TicketCreateTool(BaseTool):
    """WRITE tool — never executes without an approved Approval record."""

    name = "ticket_create"
    description = "Draft a support ticket. Requires human approval before creation."
    needs_approval = True

    def draft(self, session, args: dict) -> ToolResult:
        """Stage 1: create a pending Approval, do NOT create the ticket."""
        last = session.query(Approval).order_by(Approval.id.desc()).first()
        next_n = 3001 if last is None else int(last.id.split("-")[1]) + 1
        approval_id = f"APR-{next_n}"
        payload = {
            "subject": str(args.get("subject", "Support request"))[:200],
            "description": str(args.get("description", ""))[:2000],
            "priority": str(args.get("priority", "normal")),
            "order_id": args.get("order_id"),
        }
        session.add(Approval(id=approval_id, action="ticket_create",
                             payload_json=json.dumps(payload), status="pending"))
        session.commit()
        return ToolResult(
            ok=True,
            awaiting_approval=True,
            data={"approval_id": approval_id, "action": "ticket_create", "payload": payload},
            message=(f"I've drafted a ticket (approval {approval_id}) for your review. "
                     f"Nothing has been created yet — approve it to proceed."),
        )

    def execute(self, session, args: dict) -> ToolResult:
        """Stage 2: actually create the ticket. Called only by the approval resolver."""
        last = session.query(Ticket).order_by(Ticket.id.desc()).first()
        next_n = 2004 if last is None else int(last.id.split("-")[1]) + 1
        ticket_id = f"TKT-{next_n}"
        ticket = Ticket(
            id=ticket_id,
            subject=str(args.get("subject", "Support request"))[:200],
            description=str(args.get("description", ""))[:2000],
            status="open",
            priority=str(args.get("priority", "normal")),
            order_id=args.get("order_id"),
        )
        session.add(ticket)
        session.commit()
        return ToolResult(ok=True, data={
            "found": True, "id": ticket.id, "subject": ticket.subject,
            "status": ticket.status, "priority": ticket.priority,
            "order_id": ticket.order_id,
        })


class KBSearchTool(BaseTool):
    name = "kb_search"
    description = "Search the support knowledge base. Read-only."

    def __init__(self, retriever):
        self.retriever = retriever

    def execute(self, session, args: dict) -> ToolResult:
        query = str(args.get("query", ""))
        snippets = self.retriever.search(query, top_k=3)
        return ToolResult(ok=True, data={"snippets": snippets})


TOOLS: dict[str, BaseTool] = {}


def register_tools(retriever) -> dict[str, BaseTool]:
    tools = {
        "order_lookup": OrderLookupTool(),
        "ticket_status": TicketStatusTool(),
        "ticket_create": TicketCreateTool(),
        "kb_search": KBSearchTool(retriever),
    }
    TOOLS.clear()
    TOOLS.update(tools)
    return tools
