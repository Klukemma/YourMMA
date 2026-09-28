"""The tools agents can call. Every file operation is confined to the task's workspace."""

from __future__ import annotations

import asyncio
import html
import json
import os
import pwd
import re
import signal
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from . import safety

MAX_TOOL_OUTPUT = 15_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
AGENT_USER = "agent"


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


SPECS: dict[str, tuple[str, dict]] = {
    "web_search": ("web", _fn(
        "web_search", "Search the web. Returns findings with source URLs.",
        {"query": {"type": "string"}}, ["query"])),
    "fetch_url": ("web", _fn(
        "fetch_url", "Read a web page or text/JSON document and return its text.",
        {"url": {"type": "string"}}, ["url"])),
    "list_files": ("read", _fn(
        "list_files", "List files in the workspace (recursively, skipping .git/node_modules/venvs).",
        {"path": {"type": "string", "description": "Directory relative to the workspace root", "default": "."}}, [])),
    "read_file": ("read", _fn(
        "read_file", "Read a text file from the workspace, with line numbers.",
        {"path": {"type": "string"},
         "start_line": {"type": "integer", "default": 1},
         "max_lines": {"type": "integer", "default": 400}}, ["path"])),
    "write_file": ("write", _fn(
        "write_file", "Create or overwrite a file in the workspace (parent folders are created).",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])),
    "edit_file": ("write", _fn(
        "edit_file", "Replace an exact snippet in a file. old_text must match exactly once unless replace_all.",
        {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
         "replace_all": {"type": "boolean", "default": False}}, ["path", "old_text", "new_text"])),
    "run_command": ("shell", _fn(
        "run_command",
        "Run a bash command in the workspace root (Linux container; python3, node, npm, git available). "
        "Non-interactive: pass -y flags. Long-running servers must be started in the background with a "
        "timeout or they will be killed. Returns exit code and combined output.",
        {"command": {"type": "string"},
         "timeout_seconds": {"type": "integer", "default": 300}}, ["command"])),
}


def tool_specs(groups: list[str]) -> list[dict]:
    return [spec for group, spec in SPECS.values() if group in groups]


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... [{len(text) - limit} characters omitted] ...\n\n{text[-half:]}"


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def agent_identity() -> tuple[int, int] | None:
    """(uid, gid) to run agent commands as, when we are root and the sandbox user exists."""
    if os.geteuid() != 0:
        return None
    try:
        pw = pwd.getpwnam(AGENT_USER)
    except KeyError:
        return None
    return pw.pw_uid, pw.pw_gid


def give_to_agent(path: Path, stop_at: Path) -> None:
    """chown files the server created so the unprivileged agent shell can edit them."""
    ident = agent_identity()
    if not ident:
        return
    path = path.resolve()
    stop_at = stop_at.resolve()
    while True:
        try:
            os.chown(path, *ident)
        except OSError:
            pass
        if path == stop_at or stop_at not in path.parents:
            break
        path = path.parent


class Toolbox:
    def __init__(self, workspace: Path, trusted_domains: list[str], web_search: Callable[[str], Awaitable[str]],
                 github_token: str = "", command_timeout: int = 600):
        self.root = workspace.resolve()
        self.trusted = trusted_domains
        self._web_search = web_search
        self.github_token = github_token
        self.command_timeout = command_timeout

    # helpers ---------------------------------------------------------------
    def _path(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes the workspace: {rel}")
        return p

    async def execute(self, name: str, args: dict) -> tuple[str, str | None]:
        """Run a tool. Returns (result, blocked_reason)."""
        handler = getattr(self, f"t_{name}", None)
        if handler is None:
            return f"Unknown tool: {name}", None
        try:
            return await handler(**args)
        except TypeError as exc:
            return f"Bad arguments for {name}: {exc}", None
        except ValueError as exc:
            return f"Error: {exc}", None
        except OSError as exc:
            return f"Error: {exc}", None

    # web -------------------------------------------------------------------
    async def t_web_search(self, query: str):
        return _truncate(await self._web_search(query)), None

    async def t_fetch_url(self, url: str):
        ok, reason = safety.check_fetch_url(url)
        if not ok:
            return f"BLOCKED: {reason}", reason
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-orchestrator/1.0)"}
        async with httpx.AsyncClient(timeout=45, follow_redirects=True, headers=headers) as c:
            async with c.stream("GET", url) as resp:
                ok, reason = safety.check_fetch_url(str(resp.url))
                if not ok:
                    return f"BLOCKED after redirect: {reason}", reason
                ctype = resp.headers.get("content-type", "").lower()
                textual = any(t in ctype for t in ("text/", "json", "xml", "javascript", "yaml", "markdown")) or not ctype
                if not textual:
                    msg = (f"Not fetched: content type '{ctype}' is binary. Binary downloads go through "
                           "run_command from a trusted domain.")
                    return msg, None
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > 3_000_000:
                        break
        text = body.decode(resp.encoding or "utf-8", errors="replace")
        if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            text = html_to_text(text)
        return _truncate(f"HTTP {resp.status_code} {resp.url}\n\n{text}", 20_000), None

    # files -----------------------------------------------------------------
    async def t_list_files(self, path: str = "."):
        base = self._path(path)
        if not base.exists():
            return f"No such directory: {path}", None
        out = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for f in sorted(filenames):
                p = Path(dirpath) / f
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                out.append(f"{p.relative_to(self.root)}  ({size} B)")
                if len(out) >= 500:
                    out.append("... (truncated at 500 files)")
                    return "\n".join(out), None
        return "\n".join(out) or "(workspace is empty)", None

    async def t_read_file(self, path: str, start_line: int = 1, max_lines: int = 400):
        p = self._path(path)
        if not p.is_file():
            return f"No such file: {path}", None
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + int(max_lines)]
        body = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start))
        more = len(lines) - (start - 1 + len(chunk))
        if more > 0:
            body += f"\n... {more} more lines (use start_line={start + len(chunk)})"
        return _truncate(body or "(empty file)", 30_000), None

    async def t_write_file(self, path: str, content: str):
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        give_to_agent(p, self.root)
        return f"Wrote {len(content)} characters to {path}", None

    async def t_edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False):
        p = self._path(path)
        if not p.is_file():
            return f"No such file: {path}", None
        text = p.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_text)
        if count == 0:
            return "old_text not found; read the file again and copy the snippet exactly.", None
        if count > 1 and not replace_all:
            return f"old_text matches {count} times; add surrounding lines to make it unique or set replace_all.", None
        text = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        p.write_text(text, encoding="utf-8")
        return f"Edited {path} ({count if replace_all else 1} replacement(s))", None

    # shell -----------------------------------------------------------------
    def _env(self) -> dict:
        ident = agent_identity()
        home = pwd.getpwuid(ident[0]).pw_dir if ident else str(self.root / ".home")
        Path(home).mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": f"{home}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": home, "LANG": "C.UTF-8", "TERM": "dumb", "CI": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "GIT_TERMINAL_PROMPT": "0",
            "DEBIAN_FRONTEND": "noninteractive", "npm_config_yes": "true",
        }
        if self.github_token:
            env["GH_TOKEN"] = env["GITHUB_TOKEN"] = self.github_token
        return env

    async def t_run_command(self, command: str, timeout_seconds: int = 300):
        ok, reason = safety.check_command(command, self.trusted)
        if not ok:
            return f"BLOCKED by safety policy: {reason}", reason
        timeout = max(5, min(int(timeout_seconds), self.command_timeout))
        kwargs = {}
        ident = agent_identity()
        if ident:
            kwargs = {"user": ident[0], "group": ident[1], "extra_groups": []}
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command, cwd=str(self.root), env=self._env(),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True, **kwargs,
        )
        timed_out = False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            out = b""
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
        text = out.decode("utf-8", errors="replace")
        status = f"TIMED OUT after {timeout}s (process killed)" if timed_out else f"exit code {proc.returncode}"
        return _truncate(f"[{status}]\n{text}"), None


def parse_args(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        raise ValueError(f"tool arguments were not valid JSON: {raw[:200]}")
