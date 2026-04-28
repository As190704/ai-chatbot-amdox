"""
database/db.py — Database engine creation, session factory, and helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from database.models import Base

# ─── Engine ───────────────────────────────────────────────────────────────────
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # needed for SQLite
    echo=settings.debug,
)

# Enable WAL mode for better concurrent read performance (SQLite only)
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ─── Session factory ──────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees cleanup.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context-manager variant for use outside FastAPI dependency injection.

    Usage:
        with db_session() as db:
            db.add(...)
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
