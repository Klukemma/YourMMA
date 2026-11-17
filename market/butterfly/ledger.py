"""Append-only prediction ledger (JSONL, one prediction per line).

The rules that make grading honest:

* `created_at` is stamped here, from the clock - a caller cannot pass one.
  Nothing can be predicted "as of" yesterday.
* Every record carries the hash of the record before it, so editing or
  deleting an old prediction breaks `verify()`. Git commits of the file add an
  outside timestamp on top.
* Evidence may only be material fetched before the prediction: an evidence
  item stamped after `created_at` is refused.

A prediction says: over `horizon_days` trading days, `ticker` will beat
(direction +1) or trail (direction -1) the benchmark, with probability
`confidence` that the sign is right.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config

HORIZONS = (1, 5, 21, 63)          # a day, a week, a month, a quarter (trading days)
GENESIS = "0" * 64


class LedgerError(ValueError):
    pass


def _canonical(rec: dict) -> bytes:
    return json.dumps({k: v for k, v in rec.items() if k != "hash"},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _hash(rec: dict) -> str:
    return hashlib.sha256(_canonical(rec)).hexdigest()


def _parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


@dataclass
class Ledger:
    path: Path = config.LEDGER_PATH

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def verify(self) -> list[str]:
        """Problems found, empty when the chain is intact."""
        problems, prev, last_time = [], GENESIS, None
        for i, rec in enumerate(self.records()):
            if rec.get("prev_hash") != prev:
                problems.append(f"line {i + 1} ({rec.get('id')}): prev_hash does not match the line before")
            if rec.get("hash") != _hash(rec):
                problems.append(f"line {i + 1} ({rec.get('id')}): contents changed after it was written")
            t = _parse_time(rec["created_at"])
            if last_time and t < last_time:
                problems.append(f"line {i + 1} ({rec.get('id')}): created before the line above it")
            prev, last_time = rec.get("hash"), t
        return problems

    def add(self, *, ticker: str, direction: int, horizon_days: int, confidence: float,
            event: str, chain: list[str], evidence: list[dict] | None = None,
            expected_move: float | None = None, author: str = "unspecified",
            tags: list[str] | None = None, now: datetime | None = None) -> dict:
        """`now` exists for tests only; it must not be earlier than the last record."""
        if direction not in (1, -1):
            raise LedgerError("direction must be +1 (beat the benchmark) or -1 (trail it)")
        if horizon_days not in HORIZONS:
            raise LedgerError(f"horizon_days must be one of {HORIZONS}")
        if not 0.5 <= confidence <= 0.99:
            raise LedgerError("confidence is the probability the direction is right: 0.50-0.99")
        if not event.strip() or not chain:
            raise LedgerError("a prediction needs its event and the chain of reasoning")
        ticker = ticker.upper()
        if ticker == config.BENCHMARK:
            raise LedgerError(f"{ticker} is the benchmark; a prediction against itself is always zero")

        problems = self.verify()
        if problems:
            raise LedgerError("ledger is damaged, refusing to append: " + "; ".join(problems))
        recs = self.records()
        created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if recs and created < _parse_time(recs[-1]["created_at"]):
            raise LedgerError("clock is behind the last record")

        for ev in evidence or []:
            if "fetched" in ev and _parse_time(ev["fetched"]) > created:
                raise LedgerError(f"evidence fetched after the prediction: {ev.get('url')}")

        rec = {
            "id": f"P{len(recs) + 1:06d}",
            "created_at": created.isoformat(timespec="seconds"),
            "ticker": ticker,
            "benchmark": config.BENCHMARK,
            "direction": direction,
            "horizon_days": horizon_days,
            "confidence": round(float(confidence), 4),
            "expected_move": expected_move,
            "event": event.strip(),
            "chain": list(chain),
            "evidence": list(evidence or []),
            "author": author,
            "tags": list(tags or []),
            "prev_hash": recs[-1]["hash"] if recs else GENESIS,
        }
        rec["hash"] = _hash(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
