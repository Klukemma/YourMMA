"""Settings from the environment plus the team definition in agents.yaml."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
AGENTS_FILE = Path(os.environ.get("AGENTS_FILE", ROOT / "agents.yaml"))


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        self.app_password = os.environ.get("APP_PASSWORD", "")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.tavily_key = os.environ.get("TAVILY_API_KEY", "")
        # Handed to agents' shell so they can push branches / open PRs.
        self.agent_github_token = os.environ.get("AGENT_GITHUB_TOKEN", "")
        self.task_budget_usd = _float("TASK_BUDGET_USD", 5.0)
        self.daily_budget_usd = _float("DAILY_BUDGET_USD", 25.0)
        self.max_rounds = _int("MAX_ROUNDS", 12)
        self.max_parallel_steps = _int("MAX_PARALLEL_STEPS", 3)
        self.max_concurrent_tasks = _int("MAX_CONCURRENT_TASKS", 2)
        self.max_agent_turns = _int("MAX_AGENT_TURNS", 40)
        self.command_timeout = _int("COMMAND_TIMEOUT_SECONDS", 600)
        extra = [d.strip() for d in os.environ.get("EXTRA_TRUSTED_DOMAINS", "").split(",") if d.strip()]
        self.extra_trusted_domains = extra


settings = Settings()


def load_team_file(path: Path = AGENTS_FILE) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def build_team(file_data: dict, db_overrides: dict[str, str] | None = None) -> dict:
    """Resolve the team: agents.yaml, then MODEL_<AGENT> env vars, then models set in the UI."""
    data = copy.deepcopy(file_data)
    defaults = data.get("defaults", {})
    agents = {}
    for name, cfg in (data.get("agents") or {}).items():
        merged = {**defaults, **(cfg or {})}
        env_model = os.environ.get(f"MODEL_{name.upper()}")
        if env_model:
            merged["model"] = env_model
        if db_overrides and db_overrides.get(name):
            merged["model"] = db_overrides[name]
        merged["name"] = name
        merged.setdefault("tools", [])
        merged.setdefault("description", "")
        agents[name] = merged
    if "lead" not in agents:
        raise ValueError("agents.yaml must define a 'lead' agent")
    return {
        "agents": agents,
        "web_search_model": data.get("web_search_model"),
        "trusted_domains": data.get("trusted_domains"),
        "max_revisions": int(data.get("max_revisions", 2)),
    }
