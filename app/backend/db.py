"""SQLite persistence for the demo app's memory stores.

Addresses a real gap flagged during review prep: memory previously lived
only in the Python process's RAM and was lost on every restart -- a genuine
scalability/sustainability limitation, not just a cosmetic one. This module
lets both stores survive a server restart by saving their contents to a
local SQLite file after every mutation, and reloading them at startup.

Kept deliberately simple: one `memories` table (one row per stored note)
and one `store_state` table (the bookkeeping counters each store needs --
step counter, compression counts, etc.). No ORM, just the standard library
sqlite3 module, so this adds zero new dependencies.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent / "memory.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            trust REAL NOT NULL,
            hits INTEGER NOT NULL,
            created_at_step INTEGER NOT NULL,
            correct INTEGER,
            is_poison INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS store_state (
            store_name TEXT PRIMARY KEY,
            step INTEGER NOT NULL,
            n_added INTEGER NOT NULL,
            n_compressions INTEGER NOT NULL,
            n_poison_laundered INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history ("
        "domain TEXT NOT NULL, idx INTEGER NOT NULL, entry TEXT NOT NULL, PRIMARY KEY (domain, idx))"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return conn


def init_db() -> None:
    _connect().close()


def save_store(store_name: str, store, embedder_id: str) -> None:
    """Overwrites this store's saved rows with its current in-memory state,
    and records which embedder produced the vectors."""
    conn = _connect()
    conn.execute("DELETE FROM memories WHERE store_name = ?", (store_name,))
    conn.executemany(
        "INSERT INTO memories (store_name, text, embedding, trust, hits, "
        "created_at_step, correct, is_poison) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                store_name,
                m.text,
                json.dumps(np.asarray(m.embedding).tolist()),
                m.trust,
                m.hits,
                m.created_at_step,
                None if m.correct is None else int(m.correct),
                int(m.is_poison),
            )
            for m in store.items
        ],
    )
    conn.execute(
        """
        INSERT INTO store_state (store_name, step, n_added, n_compressions, n_poison_laundered)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(store_name) DO UPDATE SET
            step=excluded.step, n_added=excluded.n_added,
            n_compressions=excluded.n_compressions, n_poison_laundered=excluded.n_poison_laundered
        """,
        (
            store_name,
            getattr(store, "_step", 0),
            getattr(store, "_n_added", 0),
            getattr(store, "n_compressions", 0),
            getattr(store, "n_poison_laundered", 0),
        ),
    )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('embedder', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (embedder_id,),
    )
    conn.commit()
    conn.close()


def load_store(store_name: str, store, embedder_id: str) -> bool:
    """Loads saved rows into an already-constructed store object, in place.
    Returns True if anything was found and loaded.

    Vectors from one embedder are meaningless to another (different dimension
    and geometry), so data saved under a different embedder is discarded
    instead of loaded."""
    conn = _connect()
    saved = conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
    if saved and saved[0] != embedder_id:
        conn.close()
        clear_store(store_name)
        return False
    rows = conn.execute(
        "SELECT text, embedding, trust, hits, created_at_step, correct, is_poison "
        "FROM memories WHERE store_name = ? ORDER BY id",
        (store_name,),
    ).fetchall()
    state = conn.execute(
        "SELECT step, n_added, n_compressions, n_poison_laundered FROM store_state WHERE store_name = ?",
        (store_name,),
    ).fetchone()
    conn.close()

    if not rows:
        return False

    from memory import MemoryItem  # local import: avoids a circular import at module load time

    store.items = [
        MemoryItem(
            text=text,
            embedding=np.array(json.loads(embedding)),
            trust=trust,
            hits=hits,
            created_at_step=created_at_step,
            correct=None if correct is None else bool(correct),
            is_poison=bool(is_poison),
        )
        for (text, embedding, trust, hits, created_at_step, correct, is_poison) in rows
    ]
    if state:
        step, n_added, n_compressions, n_poison_laundered = state
        if hasattr(store, "_step"):
            store._step = step
        if hasattr(store, "_n_added"):
            store._n_added = n_added
        if hasattr(store, "n_compressions"):
            store.n_compressions = n_compressions
        if hasattr(store, "n_poison_laundered"):
            store.n_poison_laundered = n_poison_laundered
    return True


def clear_store(store_name: str) -> None:
    conn = _connect()
    conn.execute("DELETE FROM memories WHERE store_name = ?", (store_name,))
    conn.execute("DELETE FROM store_state WHERE store_name = ?", (store_name,))
    conn.commit()
    conn.close()


def save_history(domain: str, history: list[dict]) -> None:
    """The agent's per-step log -- needed so the curriculum still knows which
    tasks it has already attempted after a restart."""
    conn = _connect()
    conn.execute("DELETE FROM history WHERE domain = ?", (domain,))
    conn.executemany(
        "INSERT INTO history (domain, idx, entry) VALUES (?, ?, ?)",
        [(domain, i, json.dumps(h)) for i, h in enumerate(history)],
    )
    conn.commit()
    conn.close()


def load_history(domain: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT entry FROM history WHERE domain = ? ORDER BY idx", (domain,)).fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]


def clear_history(domain: str) -> None:
    conn = _connect()
    conn.execute("DELETE FROM history WHERE domain = ?", (domain,))
    conn.commit()
    conn.close()
