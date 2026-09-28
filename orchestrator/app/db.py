"""SQLite persistence. Everything a task needs to resume after a restart lives here."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,            -- queued, running, waiting, paused, done, failed, cancelled
    round INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    budget_usd REAL NOT NULL,
    resume_at REAL,
    note TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,              -- user, lead, system
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    key TEXT NOT NULL,
    title TEXT NOT NULL,
    agent TEXT NOT NULL,
    instructions TEXT NOT NULL,
    depends_on TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',   -- pending, running, reviewing, done, failed, skipped
    output TEXT,
    reviews INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_id INTEGER,
    agent TEXT,
    kind TEXT NOT NULL,              -- tool, blocked, review, error, info
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    step_id INTEGER,
    agent TEXT,
    model TEXT,
    cost_usd REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(task_id);
CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_usage_time ON usage(created_at);
"""

ACTIVE = ("queued", "running")


class DB:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur.lastrowid

    # tasks -----------------------------------------------------------------
    def create_task(self, title: str, budget_usd: float) -> str:
        tid = uuid.uuid4().hex[:12]
        now = time.time()
        self._x(
            "INSERT INTO tasks (id, title, status, budget_usd, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (tid, title, "queued", budget_usd, now, now),
        )
        return tid

    def get_task(self, tid: str) -> dict | None:
        rows = self._q("SELECT * FROM tasks WHERE id=?", (tid,))
        return rows[0] if rows else None

    def list_tasks(self, limit: int = 100) -> list[dict]:
        return self._q("SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,))

    def update_task(self, tid: str, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), tid))

    def runnable_tasks(self, now: float) -> list[dict]:
        return self._q(
            "SELECT * FROM tasks WHERE status='queued' OR (status='paused' AND resume_at<=?) ORDER BY created_at",
            (now,),
        )

    def recover_after_restart(self) -> None:
        """Anything that was mid-flight when the server stopped goes back in the queue."""
        self._x("UPDATE steps SET status='pending' WHERE status IN ('running','reviewing')")
        self._x("UPDATE tasks SET status='queued' WHERE status='running'")

    def delete_task(self, tid: str) -> None:
        for table in ("messages", "steps", "events", "usage"):
            self._x(f"DELETE FROM {table} WHERE task_id=?", (tid,))
        self._x("DELETE FROM tasks WHERE id=?", (tid,))

    # messages --------------------------------------------------------------
    def add_message(self, tid: str, role: str, content: str) -> int:
        mid = self._x(
            "INSERT INTO messages (task_id, role, content, created_at) VALUES (?,?,?,?)",
            (tid, role, content, time.time()),
        )
        self.update_task(tid)
        return mid

    def messages(self, tid: str) -> list[dict]:
        return self._q("SELECT * FROM messages WHERE task_id=? ORDER BY id", (tid,))

    # steps -----------------------------------------------------------------
    def add_step(self, tid: str, rnd: int, key: str, title: str, agent: str,
                 instructions: str, depends_on: list[str]) -> int:
        return self._x(
            "INSERT INTO steps (task_id, round, key, title, agent, instructions, depends_on) VALUES (?,?,?,?,?,?,?)",
            (tid, rnd, key, title, agent, instructions, json.dumps(depends_on)),
        )

    def steps(self, tid: str, rnd: int | None = None) -> list[dict]:
        if rnd is None:
            rows = self._q("SELECT * FROM steps WHERE task_id=? ORDER BY id", (tid,))
        else:
            rows = self._q("SELECT * FROM steps WHERE task_id=? AND round=? ORDER BY id", (tid, rnd))
        for r in rows:
            r["depends_on"] = json.loads(r["depends_on"] or "[]")
        return rows

    def get_step(self, sid: int) -> dict | None:
        rows = self._q("SELECT * FROM steps WHERE id=?", (sid,))
        if not rows:
            return None
        rows[0]["depends_on"] = json.loads(rows[0]["depends_on"] or "[]")
        return rows[0]

    def update_step(self, sid: int, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE steps SET {cols} WHERE id=?", (*fields.values(), sid))

    # events ----------------------------------------------------------------
    def add_event(self, tid: str, kind: str, content: str, step_id: int | None = None,
                  agent: str | None = None) -> None:
        self._x(
            "INSERT INTO events (task_id, step_id, agent, kind, content, created_at) VALUES (?,?,?,?,?,?)",
            (tid, step_id, agent, kind, content[:4000], time.time()),
        )

    def events(self, tid: str, step_id: int | None = None, limit: int = 50) -> list[dict]:
        if step_id is None:
            rows = self._q("SELECT * FROM events WHERE task_id=? ORDER BY id DESC LIMIT ?", (tid, limit))
        else:
            rows = self._q("SELECT * FROM events WHERE step_id=? ORDER BY id DESC LIMIT ?", (step_id, limit))
        return list(reversed(rows))

    # usage / cost ----------------------------------------------------------
    def add_usage(self, tid: str | None, step_id: int | None, agent: str, model: str, cost: float) -> None:
        self._x(
            "INSERT INTO usage (task_id, step_id, agent, model, cost_usd, created_at) VALUES (?,?,?,?,?,?)",
            (tid, step_id, agent, model, cost, time.time()),
        )
        if tid:
            with self._lock:
                self._conn.execute("UPDATE tasks SET cost_usd=cost_usd+?, updated_at=? WHERE id=?",
                                   (cost, time.time(), tid))
                if step_id:
                    self._conn.execute("UPDATE steps SET cost_usd=cost_usd+? WHERE id=?", (cost, step_id))
                self._conn.commit()

    def spent_since(self, since: float) -> float:
        rows = self._q("SELECT COALESCE(SUM(cost_usd),0) AS s FROM usage WHERE created_at>=?", (since,))
        return float(rows[0]["s"])

    # settings --------------------------------------------------------------
    def model_overrides(self) -> dict[str, str]:
        rows = self._q("SELECT key, value FROM settings WHERE key LIKE 'model:%'")
        return {r["key"][6:]: r["value"] for r in rows}

    def set_model_override(self, agent: str, model: str | None) -> None:
        if model:
            self._x("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (f"model:{agent}", model))
        else:
            self._x("DELETE FROM settings WHERE key=?", (f"model:{agent}",))

    def get_setting(self, key: str) -> str | None:
        rows = self._q("SELECT value FROM settings WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    def set_setting(self, key: str, value: str) -> None:
        self._x("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))
