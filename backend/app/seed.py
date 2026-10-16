"""Seed orders, tickets, and knowledge-base documents."""
from __future__ import annotations

from sqlalchemy.orm import Session

from .db import Approval, Order, Ticket

ORDERS = [
    {"id": "ORD-1001", "customer_name": "Ava Chen", "item": "Wireless Noise-Cancelling Headphones",
     "total_cents": 24999, "status": "shipped", "tracking_number": "1Z8842WQ1001"},
    {"id": "ORD-1002", "customer_name": "Liam Park", "item": "Mechanical Keyboard (TKL)",
     "total_cents": 12999, "status": "in_transit", "tracking_number": "1Z8842WQ1002"},
    {"id": "ORD-1003", "customer_name": "Mia Rossi", "item": "4K USB-C Monitor 27in",
     "total_cents": 39999, "status": "delivered", "tracking_number": "1Z8842WQ1003"},
    {"id": "ORD-1004", "customer_name": "Noah Kim", "item": "Ergonomic Office Chair",
     "total_cents": 29999, "status": "shipped", "tracking_number": "1Z8842WQ1004"},
    {"id": "ORD-1005", "customer_name": "Zoe Patel", "item": "Portable SSD 2TB",
     "total_cents": 15999, "status": "cancelled", "tracking_number": ""},
]

TICKETS = [
    {"id": "TKT-2001", "subject": "Headphones arrived with dented case", "description": "Packaging damaged in transit",
     "status": "in_progress", "priority": "high", "order_id": "ORD-1001"},
    {"id": "TKT-2002", "subject": "Keyboard keycap missing", "description": "Right Shift keycap missing from box",
     "status": "open", "priority": "normal", "order_id": "ORD-1002"},
    {"id": "TKT-2003", "subject": "Monitor dead pixel", "description": "One dead pixel near center",
     "status": "resolved", "priority": "normal", "order_id": "ORD-1003"},
]


def seed_db(session: Session) -> None:
    if session.query(Order).count() == 0:
        for row in ORDERS:
            session.add(Order(**row))
    if session.query(Ticket).count() == 0:
        for row in TICKETS:
            session.add(Ticket(**row))
    session.commit()
