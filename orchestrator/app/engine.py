"""The orchestration loop.

A task moves through rounds:

    lead plans a round of steps  ->  steps run (in parallel where independent)
         ^                               each step: worker agent with tools,
         |                               optionally checked by a reviewer who
         |                               can send it back for fixes
         +--- lead sees the results and either plans another round,
              asks the user something, or declares the task done.

Every state change is written to SQLite first, so a restart resumes where the
task left off instead of starting over.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import llm, safety
from .config import DATA_DIR, Settings, build_team, settings as default_settings
from .db import DB
from .tools import Toolbox, give_to_agent, parse_args, tool_specs

log = logging.getLogger("engine")


class Cancelled(Exception):
    pass


class BudgetHit(Exception):
    def __init__(self, daily: bool, limit: float):
        super().__init__("daily budget" if daily else "task budget")
        self.daily = daily
        self.limit = limit


@dataclass
class Decision:
    reply: str
    status: str
    steps: list[dict] = field(default_factory=list)
    title: str | None = None


LEAD_SYSTEM = """You are the lead of a team of AI agents that works autonomously in a cloud workspace \
(a Linux container with a shared project folder) on behalf of one user. You never do the hands-on work \
yourself: you break the user's goal into steps, give each step to the teammate best suited for it, \
check the results, and decide what happens next. The user may be away for hours or days; keep going \
until the goal is truly achieved.

Your team:
{roster}

Reply with ONLY a JSON object, no prose around it:
{{
  "title": "3-6 word name for this task (first round only, else null)",
  "reply": "message shown to the user in the chat",
  "status": "continue" | "done" | "need_input",
  "steps": [
    {{"id": "s1", "agent": "<teammate name>", "title": "short title",
      "instructions": "complete, self-contained instructions", "depends_on": []}}
  ]
}}

How to plan:
- Steps in one round run in parallel unless linked with depends_on (ids from the same round). Link steps \
that edit the same files or need another step's output.
- A teammate sees ONLY its instructions, the outputs of the steps it depends on, and the workspace files. \
Spell out goals, file paths, constraints and what "done" means.
- Typical flow: research unknowns first, then build, then verify. Use the critic for reviews of plans, \
architecture or finished work when quality matters. 1-6 steps per round is usually right.
- After each round you get the results. Failed or weak steps: re-plan them with better instructions or a \
different teammate. Don't repeat work that already succeeded.
- status "continue": steps must be non-empty; keep reply to one or two sentences on what happens next.
- status "done": only when the goal is met and verified. steps empty. reply is the final report: what was \
built or found, where it is in the workspace, how to run or use it, and any caveats.
- status "need_input": only when blocked on a decision only the user can make (credentials, a real \
preference between very different options). steps empty. reply asks the question clearly.
- The user can talk to you mid-task; new user messages override earlier instructions."""

WORKER_SYSTEM = """You are "{name}", one member of a team of AI agents working autonomously for a user. \
{prompt}

Team members: {team}

Environment:
- You work in a shared project folder; it is your current directory. Always use relative paths.
- Linux container, unprivileged user, no sudo. python3, pip, node, npm and git are installed. Install \
project dependencies locally (python -m venv .venv, npm install).
- Security policy: downloads are only allowed from trusted official sources (PyPI, npm, GitHub, other \
official registries). Other domains are blocked. Never try to work around a blocked command, never run \
scripts from unverified sources, never run obfuscated code. If something you need is blocked, say so in \
your report.
- Commands are non-interactive and time-limited. Start servers in the background and stop them when done.
{github}
Your tools: {tools}

When you are finished, reply WITHOUT calling tools. That reply is your report and it is all the lead \
sees, so make it complete but tight: what you did, files created or changed, how to run or verify it, \
results of any tests, and anything unresolved."""

REVIEW_BRIEF = """Review the work below.

## Step: {title}
## Instructions {agent} was given
{instructions}

## {agent}'s report
{output}

Check the actual files in the workspace (and run code or tests if you can) instead of trusting the \
report. Judge correctness, completeness against the instructions, security and robustness. Ignore \
trivial style issues.

End your reply with one line of JSON and nothing after it:
{{"verdict": "approve" or "revise", "feedback": "numbered list of required fixes, or a one-line summary if approving"}}"""


def _clip(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more characters]"


def extract_json(text: str) -> dict | None:
    """Pull the JSON object out of a model reply (tolerates code fences and surrounding prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for cand in candidates:
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass
    # Fall back to scanning for balanced objects, preferring the last one.
    decoder = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                val, _ = decoder.raw_decode(text[i:])
                if isinstance(val, dict):
                    found = val
            except json.JSONDecodeError:
                continue
    return found


def _summarize_args(args: dict) -> str:
    shown = {}
    for k, v in args.items():
        if isinstance(v, str) and k in ("content", "new_text", "old_text"):
            shown[k] = f"<{len(v)} chars>"
        else:
            shown[k] = v
    return _clip(json.dumps(shown, ensure_ascii=False), 400)


def _next_utc_midnight(now: float) -> float:
    d = dt.datetime.fromtimestamp(now, dt.timezone.utc).date() + dt.timedelta(days=1)
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()


def _utc_day_start(now: float) -> float:
    d = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()


class Engine:
    def __init__(self, db: DB, team_file: dict, chat=llm.chat, data_dir: Path = DATA_DIR,
                 settings: Settings = default_settings):
        self.db = db
        self.team_file = team_file
        self.chat = chat
        self.data_dir = Path(data_dir)
        self.settings = settings
        self.running: dict[str, asyncio.Task] = {}
        self._loop_task: asyncio.Task | None = None

    # setup -----------------------------------------------------------------
    def team(self) -> dict:
        return build_team(self.team_file, self.db.model_overrides())

    def workspace(self, tid: str) -> Path:
        root = self.data_dir / "workspaces"
        path = root / tid
        path.mkdir(parents=True, exist_ok=True)
        give_to_agent(path, root)
        return path

    def trusted_domains(self, team: dict) -> list[str]:
        return [*safety.DEFAULT_TRUSTED_DOMAINS, *(team.get("trusted_domains") or []),
                *self.settings.extra_trusted_domains]

    async def start(self) -> None:
        self.db.recover_after_restart()
        self._loop_task = asyncio.create_task(self.worker_loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        for t in list(self.running.values()):
            t.cancel()
        await asyncio.gather(*self.running.values(), return_exceptions=True)

    async def worker_loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("worker loop error")
            await asyncio.sleep(2)

    def tick(self) -> None:
        for t in self.db.runnable_tasks(time.time()):
            if t["id"] in self.running or len(self.running) >= self.settings.max_concurrent_tasks:
                continue
            self.launch(t["id"])

    def launch(self, tid: str) -> asyncio.Task:
        self.db.update_task(tid, status="running", resume_at=None, note=None)
        task = asyncio.create_task(self.run_task(tid))
        self.running[tid] = task
        task.add_done_callback(lambda _t, tid=tid: self.running.pop(tid, None))
        return task

    def cancel(self, tid: str) -> None:
        self.db.update_task(tid, status="cancelled")
        task = self.running.get(tid)
        if task:
            task.cancel()

    # task lifecycle ----------------------------------------------------------
    async def run_task(self, tid: str) -> None:
        try:
            await self._run(tid)
        except (Cancelled, asyncio.CancelledError):
            self._reset_active_steps(tid, "skipped")
            self.db.update_task(tid, status="cancelled")
            self.db.add_message(tid, "system", "Stopped. Send a message to pick it back up.")
        except BudgetHit as hit:
            self._reset_active_steps(tid, "pending")
            if hit.daily:
                resume = _next_utc_midnight(time.time())
                self.db.update_task(tid, status="paused", resume_at=resume,
                                    note=f"Daily cap ${hit.limit:.2f} reached")
                self.db.add_message(tid, "system",
                                    f"The daily spending cap (${hit.limit:.2f}, all tasks) is reached. "
                                    "Paused; it resumes automatically at 00:00 UTC.")
            else:
                self.db.update_task(tid, status="waiting", note=f"Budget ${hit.limit:.2f} reached")
                self.db.add_message(tid, "system",
                                    f"This task used its ${hit.limit:.2f} budget. Raise the budget and "
                                    "send a message (for example \"continue\") to resume.")
        except Exception as exc:
            log.error("task %s failed:\n%s", tid, traceback.format_exc())
            self._reset_active_steps(tid, "pending")
            self.db.update_task(tid, status="failed", note=_clip(str(exc), 300))
            self.db.add_message(tid, "system", f"Something went wrong: {_clip(str(exc), 1500)}\n\n"
                                               "Send a message to retry.")

    def _reset_active_steps(self, tid: str, to: str) -> None:
        for s in self.db.steps(tid):
            if s["status"] in ("running", "reviewing"):
                self.db.update_step(s["id"], status=to)

    def _last_user_message_id(self, tid: str) -> int:
        ids = [m["id"] for m in self.db.messages(tid) if m["role"] == "user"]
        return max(ids) if ids else 0

    async def _run(self, tid: str) -> None:
        while True:
            task = self.db.get_task(tid)
            rnd = task["round"]
            if rnd and any(s["status"] == "pending" for s in self.db.steps(tid, rnd)):
                await self.execute_round(tid, rnd)  # resuming an interrupted round

            if rnd >= self.settings.max_rounds:
                self.db.add_message(tid, "system",
                                    f"Reached the limit of {self.settings.max_rounds} rounds. "
                                    "Send a message to keep going.")
                self.db.update_task(tid, status="waiting", note="Round limit reached")
                return

            seen = self._last_user_message_id(tid)
            decision = await self.plan(tid)
            if decision.title and rnd == 0:
                self.db.update_task(tid, title=_clip(decision.title, 80))
            if decision.reply:
                self.db.add_message(tid, "lead", decision.reply)

            if decision.status == "continue" and decision.steps:
                rnd += 1
                self.db.update_task(tid, round=rnd)
                for s in decision.steps:
                    self.db.add_step(tid, rnd, s["id"], s["title"], s["agent"], s["instructions"], s["depends_on"])
                await self.execute_round(tid, rnd)
                continue

            if self._last_user_message_id(tid) > seen:
                continue  # the user wrote while we were deciding; take it into account
            self.db.update_task(tid, status="done" if decision.status == "done" else "waiting",
                                note=None if decision.status == "done" else "Waiting for your reply")
            return

    # budget-aware model call -----------------------------------------------
    def _check_budget(self, tid: str) -> None:
        task = self.db.get_task(tid)
        if task is None or task["status"] == "cancelled":
            raise Cancelled()
        if task["cost_usd"] >= task["budget_usd"]:
            raise BudgetHit(False, task["budget_usd"])
        cap = self.settings.daily_budget_usd
        if cap > 0 and self.db.spent_since(_utc_day_start(time.time())) >= cap:
            raise BudgetHit(True, cap)

    async def call(self, tid: str, step_id: int | None, agent: dict, messages: list[dict],
                   tools: list[dict] | None = None, plugins: list[dict] | None = None,
                   model: str | None = None) -> llm.Reply:
        self._check_budget(tid)
        reply = await self.chat(agent, messages, tools=tools, plugins=plugins, model=model)
        self.db.add_usage(tid, step_id, agent["name"], model or agent["model"], reply.cost)
        return reply

    # planning ----------------------------------------------------------------
    def _roster(self, team: dict) -> str:
        lines = []
        for name, a in team["agents"].items():
            if name == "lead":
                continue
            tools = ", ".join(a.get("tools") or []) or "none (thinking only)"
            review = f"; its work is checked by {a['review_by']}" if a.get("review_by") else ""
            lines.append(f"- {name} ({a['model']}): {a['description'].strip()} Tools: {tools}{review}")
        return "\n".join(lines)

    def _plan_context(self, tid: str) -> str:
        task = self.db.get_task(tid)
        parts = ["## Conversation with the user"]
        for m in self.db.messages(tid):
            if m["role"] in ("user", "lead"):
                parts.append(f"[{m['role']}] {_clip(m['content'], 5000)}")
            else:
                parts.append(f"[system notice] {_clip(m['content'], 500)}")
        steps = self.db.steps(tid)
        if steps:
            parts.append("\n## Work so far")
            last_round = max(s["round"] for s in steps)
            for rnd in sorted({s["round"] for s in steps}):
                parts.append(f"\n### Round {rnd}")
                limit = 6000 if rnd == last_round else 1200
                for s in (x for x in steps if x["round"] == rnd):
                    parts.append(f"- [{s['key']}] {s['title']} -> {s['agent']}: {s['status'].upper()}"
                                 f" (reviews: {s['reviews']})\n  Output:\n{_clip(s['output'], limit)}")
        listing = self._workspace_listing(tid)
        parts.append(f"\n## Workspace files\n{listing}")
        parts.append(
            f"\n## Now\nPlanning round {task['round'] + 1} of at most {self.settings.max_rounds}. "
            f"Spent ${task['cost_usd']:.2f} of the ${task['budget_usd']:.2f} task budget. "
            "Decide the next step and reply with the JSON object.")
        return "\n".join(parts)

    def _workspace_listing(self, tid: str, limit: int = 150) -> str:
        root = self.workspace(tid)
        skip = {".git", "node_modules", ".venv", "venv", "__pycache__", ".home"}
        out = []
        for p in sorted(root.rglob("*")):
            if any(part in skip for part in p.relative_to(root).parts):
                continue
            if p.is_file():
                out.append(str(p.relative_to(root)))
                if len(out) >= limit:
                    out.append("... (more files)")
                    break
        return "\n".join(out) or "(empty)"

    async def plan(self, tid: str) -> Decision:
        team = self.team()
        lead = team["agents"]["lead"]
        workers = [n for n in team["agents"] if n != "lead"]
        if not workers:
            raise ValueError("agents.yaml needs at least one agent besides the lead")
        messages = [
            {"role": "system", "content": LEAD_SYSTEM.format(roster=self._roster(team))},
            {"role": "user", "content": self._plan_context(tid)},
        ]
        data = None
        for _attempt in range(3):
            reply = await self.call(tid, None, lead, messages)
            data = extract_json(reply.content)
            if data and data.get("status") in ("continue", "done", "need_input"):
                break
            messages += [reply.message, {"role": "user", "content":
                         "That was not a valid decision. Reply with ONLY the JSON object described in "
                         "your instructions (keys: title, reply, status, steps)."}]
            data = None
        if data is None:
            raise llm.LLMError("the lead did not return a usable plan after 3 tries")

        steps, keys = [], set()
        for i, raw in enumerate(data.get("steps") or []):
            if not isinstance(raw, dict) or not str(raw.get("instructions", "")).strip():
                continue
            key = str(raw.get("id") or f"s{i + 1}")
            while key in keys:
                key += "_"
            keys.add(key)
            agent = str(raw.get("agent", "")).strip().lower()
            if agent not in workers:
                fallback = "coder" if "coder" in workers else workers[0]
                self.db.add_event(tid, "info", f"Lead assigned unknown agent '{agent}'; using {fallback}")
                agent = fallback
            deps = raw.get("depends_on") or []
            steps.append({"id": key, "agent": agent, "title": _clip(str(raw.get("title") or key), 120),
                          "instructions": str(raw["instructions"]),
                          "depends_on": [str(d) for d in deps if isinstance(d, (str, int))]})
        for s in steps:
            s["depends_on"] = [d for d in s["depends_on"] if d in keys and d != s["id"]]
        status = data["status"]
        if status == "continue" and not steps:
            status = "need_input"
        return Decision(reply=str(data.get("reply") or "").strip(), status=status, steps=steps,
                        title=data.get("title") if isinstance(data.get("title"), str) else None)

    # executing a round -------------------------------------------------------
    async def execute_round(self, tid: str, rnd: int) -> None:
        running: dict[asyncio.Task, int] = {}
        try:
            while True:
                steps = self.db.steps(tid, rnd)
                by_key = {s["key"]: s for s in steps}
                active = set(running.values())
                for s in steps:
                    if s["status"] != "pending" or s["id"] in active:
                        continue
                    deps = [by_key[k] for k in s["depends_on"] if k in by_key]
                    if any(d["status"] in ("failed", "skipped") for d in deps):
                        self.db.update_step(s["id"], status="skipped", finished_at=time.time(),
                                            output="Skipped: a step it depends on did not succeed.")
                        continue
                    if all(d["status"] == "done" for d in deps) and len(running) < self.settings.max_parallel_steps:
                        t = asyncio.create_task(self.run_step(tid, s["id"]))
                        running[t] = s["id"]
                if not running:
                    leftovers = [s for s in self.db.steps(tid, rnd) if s["status"] == "pending"]
                    if not leftovers:
                        return
                    if self._startable(tid, rnd):
                        continue  # a skip just cascaded; the next pass handles it
                    # Only a dependency cycle leaves pending steps with nothing runnable.
                    for s in leftovers:
                        self.db.update_step(s["id"], status="skipped", finished_at=time.time(),
                                            output="Skipped: its dependencies could never complete.")
                    return
                done, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    running.pop(t)
                    exc = t.exception()
                    if exc:
                        raise exc
        finally:
            for t in running:
                t.cancel()
            if running:
                await asyncio.gather(*running.keys(), return_exceptions=True)

    def _startable(self, tid: str, rnd: int) -> bool:
        steps = self.db.steps(tid, rnd)
        by_key = {s["key"]: s for s in steps}
        for s in steps:
            if s["status"] != "pending":
                continue
            deps = [by_key[k] for k in s["depends_on"] if k in by_key]
            if all(d["status"] == "done" for d in deps) or any(d["status"] in ("failed", "skipped") for d in deps):
                return True
        return False

    def _step_brief(self, tid: str, step: dict) -> str:
        users = [m["content"] for m in self.db.messages(tid) if m["role"] == "user"]
        parts = ["## What the user asked for", "\n\n---\n\n".join(_clip(u, 6000) for u in users),
                 f"\n## Your step: {step['title']}", step["instructions"]]
        siblings = self.db.steps(tid, step["round"])
        deps = [s for s in siblings if s["key"] in step["depends_on"]]
        if deps:
            parts.append("\n## Results of the steps you build on")
            for d in deps:
                parts.append(f"### {d['title']} ({d['agent']})\n{_clip(d['output'], 8000)}")
        others = [s for s in siblings if s["id"] != step["id"] and s["key"] not in step["depends_on"]]
        if others:
            parts.append("\n## Other steps in this round (handled by teammates; don't do their work)")
            parts += [f"- {s['title']} ({s['agent']})" for s in others]
        return "\n".join(parts)

    def _worker_system(self, agent: dict, team: dict) -> str:
        names = ", ".join(f"{n} ({a['description'].split('.')[0].strip()})" for n, a in team["agents"].items())
        github = ("- A GitHub token is available as GH_TOKEN for cloning private repos, pushing branches and "
                  "opening pull requests. Never force-push or push directly to main/master.\n"
                  if self.settings.agent_github_token and "shell" in agent.get("tools", []) else "")
        tools = ", ".join(f["function"]["name"] for f in tool_specs(agent.get("tools", []))) or "none"
        return WORKER_SYSTEM.format(name=agent["name"], prompt=(agent.get("prompt") or "").strip(),
                                    team=names, github=github, tools=tools)

    def _toolbox(self, tid: str, step_id: int, agent: dict, team: dict) -> Toolbox:
        async def web_search(query: str) -> str:
            return await self.web_search(tid, step_id, agent, team, query)
        return Toolbox(self.workspace(tid), self.trusted_domains(team), web_search,
                       github_token=self.settings.agent_github_token,
                       command_timeout=self.settings.command_timeout)

    async def web_search(self, tid: str, step_id: int, agent: dict, team: dict, query: str) -> str:
        if self.settings.tavily_key:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post("https://api.tavily.com/search", json={
                    "api_key": self.settings.tavily_key, "query": query,
                    "max_results": 6, "include_answer": True})
            if r.status_code >= 400:
                return f"Search failed: HTTP {r.status_code} {r.text[:300]}"
            data = r.json()
            out = [f"Answer: {data['answer']}"] if data.get("answer") else []
            for res in data.get("results", []):
                out.append(f"- {res.get('title')}\n  {res.get('url')}\n  {_clip(res.get('content'), 600)}")
            return "\n".join(out) or "No results."
        searcher = {**agent, "params": {}}
        reply = await self.call(tid, step_id, searcher, [{"role": "user", "content":
                                f"Search the web and report the most relevant, current findings for: {query}\n"
                                "Give the source URL for every fact."}],
                                plugins=[{"id": "web", "max_results": 6}],
                                model=team.get("web_search_model") or agent["model"])
        return reply.content or "No results."

    async def run_step(self, tid: str, sid: int) -> None:
        step = self.db.get_step(sid)
        team = self.team()
        agent = team["agents"][step["agent"]]
        self.db.update_step(sid, status="running", started_at=time.time())
        try:
            messages = [{"role": "system", "content": self._worker_system(agent, team)},
                        {"role": "user", "content": self._step_brief(tid, step)}]
            output = await self.run_agent(tid, sid, agent, messages, team)
            reviewer = team["agents"].get(agent.get("review_by") or "")
            reviews = 0
            while reviewer and reviewer["name"] != agent["name"]:
                self.db.update_step(sid, status="reviewing", output=output)
                verdict, feedback = await self.review(tid, sid, reviewer, step, output, team)
                reviews += 1
                self.db.update_step(sid, reviews=reviews)
                self.db.add_event(tid, "review", f"{verdict.upper()}: {feedback}", sid, reviewer["name"])
                if verdict == "approve":
                    output += f"\n\n---\nApproved by {reviewer['name']}: {_clip(feedback, 800)}"
                    break
                if reviews > team["max_revisions"]:
                    output += (f"\n\n---\n{reviewer['name']} still wanted changes after {reviews - 1} "
                               f"revision(s):\n{_clip(feedback, 3000)}")
                    break
                self.db.update_step(sid, status="running", output=output)
                messages.append({"role": "user", "content":
                                 f"{reviewer['name']} reviewed your work and requires changes:\n\n{feedback}\n\n"
                                 "Fix them, verify, then reply with your updated full report."})
                output = await self.run_agent(tid, sid, agent, messages, team)
            self.db.update_step(sid, status="done", output=output, finished_at=time.time())
        except (BudgetHit, Cancelled, asyncio.CancelledError):
            raise
        except Exception as exc:
            log.warning("step %s failed: %s", sid, exc)
            self.db.update_step(sid, status="failed", output=f"Failed: {_clip(str(exc), 2000)}",
                                finished_at=time.time())
            self.db.add_event(tid, "error", _clip(str(exc), 1000), sid, agent["name"])

    async def review(self, tid: str, sid: int, reviewer: dict, step: dict, output: str,
                     team: dict) -> tuple[str, str]:
        messages = [{"role": "system", "content": self._worker_system(reviewer, team)},
                    {"role": "user", "content": REVIEW_BRIEF.format(
                        title=step["title"], agent=step["agent"], instructions=step["instructions"],
                        output=_clip(output, 12000))}]
        text = await self.run_agent(tid, sid, reviewer, messages, team)
        data = extract_json(text) or {}
        verdict = str(data.get("verdict", "")).lower()
        feedback = str(data.get("feedback") or "").strip()
        if verdict not in ("approve", "revise"):
            verdict = "revise" if re.search(r"\brevise\b", text, re.I) else "approve"
        if not feedback:
            feedback = _clip(text, 3000)
        return verdict, feedback

    @staticmethod
    def _trim(messages: list[dict], limit: int = 350_000) -> None:
        total = sum(len(str(m.get("content") or "")) for m in messages)
        for m in messages[2:-12]:
            if total <= limit:
                return
            if m.get("role") == "tool" and len(m.get("content") or "") > 500:
                total -= len(m["content"])
                m["content"] = "[older tool output removed to save space]"

    async def run_agent(self, tid: str, sid: int, agent: dict, messages: list[dict], team: dict) -> str:
        tools = tool_specs(agent.get("tools") or []) or None
        toolbox = self._toolbox(tid, sid, agent, team) if tools else None
        for _turn in range(self.settings.max_agent_turns):
            self._trim(messages)
            reply = await self.call(tid, sid, agent, messages, tools)
            messages.append(reply.message)
            if not reply.tool_calls:
                if reply.content.strip():
                    return reply.content
                messages.append({"role": "user", "content": "Please write your final report now."})
                continue
            for call in reply.tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = parse_args(fn.get("arguments"))
                except ValueError as exc:
                    result = str(exc)
                else:
                    self.db.add_event(tid, "tool", f"{name} {_summarize_args(args)}", sid, agent["name"])
                    result, blocked = await toolbox.execute(name, args)
                    if blocked:
                        self.db.add_event(tid, "blocked", f"{name}: {blocked}", sid, agent["name"])
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name,
                                 "content": result})
        messages.append({"role": "user", "content":
                         "You have reached the step's turn limit. Do not call any more tools. Write your "
                         "final report: what is done, what is not, and what should happen next."})
        reply = await self.call(tid, sid, agent, messages, tools)
        messages.append(reply.message)
        for call in reply.tool_calls:  # every tool call needs an answer or the next request is rejected
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "name": (call.get("function") or {}).get("name", ""),
                             "content": "Not executed: turn limit reached."})
        return reply.content or "(turn limit reached without a report)"
