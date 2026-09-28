"""Web server: the chat UI plus a small JSON API, behind a single password."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import logging
import os
import secrets
import shutil
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import DATA_DIR, load_team_file, settings
from .db import DB
from .engine import Engine

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC = Path(__file__).parent / "static"
COOKIE = "orch_session"
SESSION_DAYS = 30

# The database and session secret stay private to the server process; agent
# commands run as another user (see tools.agent_identity).
os.umask(0o077)
db = DB(DATA_DIR / "app.db")
engine = Engine(db, load_team_file())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not settings.app_password:
        log.warning("APP_PASSWORD is not set - the app will refuse all logins until it is.")
    if not settings.openrouter_key:
        log.warning("OPENROUTER_API_KEY is not set - agents cannot run.")
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(title="AI Orchestrator", lifespan=lifespan, docs_url=None, redoc_url=None)


# auth ------------------------------------------------------------------------
def _secret() -> bytes:
    stored = db.get_setting("session_secret")
    if not stored:
        stored = secrets.token_hex(32)
        db.set_setting("session_secret", stored)
    # Changing APP_PASSWORD logs every device out.
    return hashlib.sha256((stored + settings.app_password).encode()).digest()


def _sign(issued: int) -> str:
    mac = hmac.new(_secret(), str(issued).encode(), hashlib.sha256).hexdigest()
    return f"{issued}.{mac}"


def _valid(token: str | None) -> bool:
    if not token or "." not in token or not settings.app_password:
        return False
    issued_s, _ = token.split(".", 1)
    if not issued_s.isdigit() or time.time() - int(issued_s) > SESSION_DAYS * 86400:
        return False
    return hmac.compare_digest(token, _sign(int(issued_s)))


def require_auth(request: Request) -> None:
    if not _valid(request.cookies.get(COOKIE)):
        raise HTTPException(401, "Not logged in")


class Login(BaseModel):
    password: str


@app.post("/api/login")
async def login(body: Login, request: Request, response: Response):
    if not settings.app_password:
        raise HTTPException(503, "Set the APP_PASSWORD environment variable on the server first.")
    if not hmac.compare_digest(body.password.encode(), settings.app_password.encode()):
        await asyncio.sleep(1.5)  # slows down password guessing
        raise HTTPException(401, "Wrong password")
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(COOKIE, _sign(int(time.time())), max_age=SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=secure)
    return {"ok": True}


@app.post("/api/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


# pages ------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# tasks ------------------------------------------------------------------------
def _progress(task: dict, steps: list[dict]) -> dict:
    total = len(steps)
    finished = sum(1 for s in steps if s["status"] in ("done", "failed", "skipped"))
    if task["status"] == "done":
        pct = 100
    elif total == 0:
        pct = 3 if task["status"] == "running" else 0
    else:
        partial = sum(0.5 for s in steps if s["status"] in ("running", "reviewing"))
        pct = min(95, round((finished + partial) / total * 100))
    return {"percent": pct, "finished": finished, "total": total}


def _task_summary(task: dict) -> dict:
    steps = db.steps(task["id"])
    return {**task, "progress": _progress(task, steps)}


def _get_task(tid: str) -> dict:
    task = db.get_task(tid)
    if not task:
        raise HTTPException(404, "No such task")
    return task


class NewTask(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    budget_usd: float | None = Field(default=None, gt=0, le=1000)


class NewMessage(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)


class Budget(BaseModel):
    budget_usd: float = Field(gt=0, le=1000)


class ModelChoice(BaseModel):
    agent: str
    model: str = ""


@app.get("/api/tasks", dependencies=[Depends(require_auth)])
async def list_tasks():
    return [_task_summary(t) for t in db.list_tasks()]


@app.post("/api/tasks", dependencies=[Depends(require_auth)])
async def create_task(body: NewTask):
    title = " ".join(body.message.split())[:60]
    tid = db.create_task(title, body.budget_usd or settings.task_budget_usd)
    db.add_message(tid, "user", body.message)
    engine.workspace(tid)
    engine.tick()
    return _task_summary(db.get_task(tid))


@app.get("/api/tasks/{tid}", dependencies=[Depends(require_auth)])
async def get_task(tid: str):
    task = _get_task(tid)
    steps = db.steps(tid)
    for s in steps:
        s["output"] = (s["output"] or "")[:3000]
        s["full_output_length"] = len(db.get_step(s["id"])["output"] or "")
    return {
        "task": {**task, "progress": _progress(task, steps)},
        "messages": db.messages(tid),
        "steps": steps,
        "events": db.events(tid, limit=40),
    }


@app.get("/api/tasks/{tid}/steps/{sid}", dependencies=[Depends(require_auth)])
async def get_step(tid: str, sid: int):
    step = db.get_step(sid)
    if not step or step["task_id"] != tid:
        raise HTTPException(404, "No such step")
    return {"step": step, "events": db.events(tid, step_id=sid, limit=300)}


@app.post("/api/tasks/{tid}/messages", dependencies=[Depends(require_auth)])
async def post_message(tid: str, body: NewMessage):
    task = _get_task(tid)
    db.add_message(tid, "user", body.message)
    if task["status"] not in ("running", "queued"):
        db.update_task(tid, status="queued", resume_at=None)
        engine.tick()
    return {"ok": True}


@app.post("/api/tasks/{tid}/cancel", dependencies=[Depends(require_auth)])
async def cancel_task(tid: str):
    _get_task(tid)
    engine.cancel(tid)
    return {"ok": True}


@app.post("/api/tasks/{tid}/budget", dependencies=[Depends(require_auth)])
async def set_budget(tid: str, body: Budget):
    _get_task(tid)
    db.update_task(tid, budget_usd=body.budget_usd)
    return {"ok": True}


@app.delete("/api/tasks/{tid}", dependencies=[Depends(require_auth)])
async def delete_task(tid: str):
    _get_task(tid)
    running = engine.running.get(tid)
    engine.cancel(tid)
    if running:
        await asyncio.gather(running, return_exceptions=True)
    db.delete_task(tid)
    shutil.rmtree(DATA_DIR / "workspaces" / tid, ignore_errors=True)
    return {"ok": True}


# workspace files ---------------------------------------------------------------
ZIP_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", ".home"}


def _workspace_file(tid: str, rel: str) -> Path:
    root = engine.workspace(tid).resolve()
    p = (root / rel).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "No such file")
    return p


@app.get("/api/tasks/{tid}/files", dependencies=[Depends(require_auth)])
async def list_files(tid: str):
    _get_task(tid)
    root = engine.workspace(tid)
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in ZIP_SKIP for part in rel.parts) or not p.is_file():
            continue
        files.append({"path": str(rel), "size": p.stat().st_size})
        if len(files) >= 2000:
            break
    return files


@app.get("/api/tasks/{tid}/file", dependencies=[Depends(require_auth)])
async def read_file(tid: str, path: str):
    p = _workspace_file(tid, path)
    if p.stat().st_size > 2_000_000:
        raise HTTPException(413, "File too large to preview; download the zip instead.")
    return {"path": path, "content": p.read_text(encoding="utf-8", errors="replace")}


@app.get("/api/tasks/{tid}/download", dependencies=[Depends(require_auth)])
async def download(tid: str):
    _get_task(tid)
    root = engine.workspace(tid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if p.is_file() and not any(part in ZIP_SKIP for part in rel.parts):
                zf.write(p, str(rel))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="task-{tid}.zip"'})


# team -----------------------------------------------------------------------------
@app.get("/api/team", dependencies=[Depends(require_auth)])
async def get_team():
    team = engine.team()
    overrides = db.model_overrides()
    spent_today = db.spent_since(time.time() - (time.time() % 86400))
    return {
        "agents": [{"name": n, "model": a["model"], "description": a["description"],
                    "tools": a.get("tools", []), "review_by": a.get("review_by"),
                    "overridden": n in overrides} for n, a in team["agents"].items()],
        "daily_budget_usd": settings.daily_budget_usd,
        "task_budget_usd": settings.task_budget_usd,
        "spent_today_usd": spent_today,
        "api_key_set": bool(settings.openrouter_key),
    }


@app.post("/api/team", dependencies=[Depends(require_auth)])
async def set_model(body: ModelChoice):
    if body.agent not in engine.team()["agents"]:
        raise HTTPException(404, "No such agent")
    db.set_model_override(body.agent, body.model.strip() or None)
    return {"ok": True}
