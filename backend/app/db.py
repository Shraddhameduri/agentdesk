"""Database engine, session, models.

SQLite (file) is the default so tests and the local demo run offline with no
external services. Set DATABASE_URL to a Postgres DSN for the pgvector path.
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import DateTime, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. ORD-1001
    customer_name: Mapped[str] = mapped_column(String(120))
    item: Mapped[str] = mapped_column(String(200))
    total_cents: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String(40))  # shipped|in_transit|delivered|cancelled
    tracking_number: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. TKT-2001
    subject: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40))  # open|in_progress|resolved
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    order_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. APR-3001
    action: Mapped[str] = mapped_column(String(64))  # currently: ticket_create
    payload_json: Mapped[str] = mapped_column(Text)  # JSON of the tool call to execute
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|approved|rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(20))  # user|assistant|system
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def get_engine(database_url: str | None = None):
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///./data/app.db")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    pool_size_args = {}
    if url == "sqlite:///:memory:":
        # StaticPool keeps one shared connection so in-memory DB survives across sessions (tests).
        from sqlalchemy.pool import StaticPool
        pool_size_args = {"poolclass": StaticPool}
    elif not url.startswith("sqlite"):
        pool_size_args = {"pool_size": 10, "max_overflow": 20, "pool_pre_ping": True}
    return create_engine(url, connect_args=connect_args, **pool_size_args)


def session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
