"""Backtest of the same-day rule (sameday.py's docstring), point in time.

Every ET day from 2016-08-01 to the last day all three leg tables cover
(the UFC walk-forward cache ends 2026-08-08), the rule builds at most one
ticket from that day's legs (parlays/data/legs_*.csv.gz, written by
legs_history.py), and the market-only control builds its own. Each is
settled at the legs' prices on file, a void leg (NFL tie, UFC draw or no
contest) paying 1.0. Reported for ALL days and for the LAST 3 YEARS before
the end, and split by how many sports a ticket mixes, by weekday and by
year. Nothing here was tuned: the rule was fixed before this ran.

Writes parlays/data/sameday_backtest.json and parlays/data/sameday_tickets.csv.gz.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "parlays" / "engine"))
import sameday as sdy  # noqa: E402

DATA = ROOT / "parlays" / "data"
START = "2016-08-01"
SPORT_OF = {"ufc": "UFC", "nfl": "NFL", "soccer": "Soccer"}


def load():
    frames = []
    for name, sport in SPORT_OF.items():
        d = pd.read_csv(DATA / f"legs_{name}.csv.gz")
        d["sport"] = sport
        frames.append(d)
    ends = {f["sport"].iloc[0]: f["day"].max() for f in frames}
    end = min(ends.values())
    legs = pd.concat(frames, ignore_index=True)
    legs = legs[(legs["day"] >= START) & (legs["day"] <= end)]
    return legs, end, ends


def to_leg(r):
    result = {"won": {"winner": "pick"}, "lost": {"winner": "other"},
              "void": {"winner": None}}[r.result]
    num = lambda x: None if pd.isna(x) else float(x)
    return sdy.leg(r.sport, r.key, r.day, r.event, r.pick, num(r.p), num(r.decimal),
                   p_market=num(r.p_market), p_model=num(r.p_model), opponent=r.opponent,
                   result=result, price_source=r.price_source)


def run(legs):
    out = {"same_day": [], "market_only": []}
    days = 0
    for day, g in legs.groupby("day"):
        days += 1
        L = [l for l in (to_leg(r) for r in g.itertuples()) if l is not None]
        for name, objective in (("same_day", "p"), ("market_only", "p_market")):
            t = sdy.build(L, objective)
            if t is not None:
                t["day"] = day
                out[name].append(t)
    return out, days


def split(tickets, key):
    by = defaultdict(list)
    for t in tickets:
        by[key(t)].append(t)
    return {k: sdy.summarise(v) for k, v in sorted(by.items())}


def multipliers(tickets):
    m = np.array([t["multiplier"] for t in tickets]) if tickets else np.array([])
    if not len(m):
        return None
    return {"median": round(float(np.median(m)), 2), "min": round(float(m.min()), 2),
            "max": round(float(m.max()), 2)}


def main():
    legs, end, ends = load()
    last3 = (date.fromisoformat(end) - timedelta(days=3 * 365)).isoformat()
    built, days = run(legs)
    results, extra = {}, {}
    for name, tickets in built.items():
        recent = [t for t in tickets if t["day"] > last3]
        results[name] = {"all": sdy.summarise(tickets), "last_3_years": sdy.summarise(recent)}
        extra[name] = {
            "by_sports": split(tickets, lambda t: "1 sport" if len(t["sports"]) == 1 else "2+ sports"),
            "by_mix": split(tickets, lambda t: "+".join(t["sports"])),
            "by_weekday": split(tickets, lambda t: date.fromisoformat(t["day"]).strftime("%a")),
            "by_year": split(tickets, lambda t: t["day"][:4]),
            "multipliers": {"all": multipliers(tickets), "last_3_years": multipliers(recent)},
            "legs_used": dict(pd.Series([l["sport"] for t in tickets for l in t["legs"]]).value_counts()),
        }
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": {"day_tz": sdy.DAY_TZ, "band": list(sdy.BAND), "max_legs": sdy.MAX_LEGS,
                 "pool": sdy.POOL, "min_p": sdy.MIN_P},
        "periods": {"all": [START, end], "last_3_years": [last3, end]},
        "days": days, "leg_tables_end": ends,
        "results": results, "splits": extra,
        "caveats": caveats(results, extra, ends, end),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "sameday_backtest.json").write_text(json.dumps(payload, indent=1, default=_json))
    rows = [{"rule": name, "day": t["day"], "n_legs": t["n_legs"], "sports": "+".join(t["sports"]),
             "multiplier": round(t["multiplier"], 4), "p": round(t["p"], 6),
             "p_market": None if t["p_market"] is None else round(t["p_market"], 6),
             "status": sdy.settle(t)["status"], "profit": sdy.settle(t)["profit"],
             "legs": "|".join(f"{l['sport']}:{l['pick']}@{l['decimal']:.3f}" for l in t["legs"])}
            for name, ts in built.items() for t in ts]
    pd.DataFrame(rows).to_csv(DATA / "sameday_tickets.csv.gz", index=False)
    table(payload)


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


def _pct(x):
    return f"{'+' if x >= 0 else '-'}{abs(x) * 100:.0f}%"


def caveats(results, extra, ends, end):
    a = results["same_day"]["all"]
    r3 = results["same_day"]["last_3_years"]
    mo = results["market_only"]["last_3_years"]
    out = [
        f"The market is hard to beat. Over the last 3 years the rule's tickets won {r3.get('wins', 0)} of "
        f"{r3.get('tickets', 0)} ({_pct(r3.get('roi', 0))} a ticket, {_pct(r3.get('roi_low', 0))} to "
        f"{_pct(r3.get('roi_high', 0))}); the market-only control's won {mo.get('wins', 0)} of "
        f"{mo.get('tickets', 0)} ({_pct(mo.get('roi', 0))}). An interval that spans zero is no evidence of an edge.",
        f"Measured to {end}, the last day all three sports' leg tables cover (UFC {ends['UFC']}, "
        f"NFL {ends['NFL']}, soccer {ends['Soccer']}); the UFC walk-forward needs a runner refit to go further.",
        "Prices: UFC at each card's closing (or, where only that exists, opening) line, NFL at the closing "
        "moneyline, soccer at the pre-match average posted on Friday or Tuesday afternoon. A ticket built "
        "earlier in the day gets different prices; NFL closing lines know the day's news.",
        "Payouts: these files' margins are lower than a single sportsbook's live prices (NFL closing "
        "about 2.8% a game against about 4% live), so a live ticket pays a little less than the record assumes.",
        "Soccer here is the top five leagues only, which play no matches on international-break weekends "
        "and in June-July; on those days a ticket comes from UFC and NFL alone, or there is none.",
        "UFC days are the event date; a card held outside the Americas can fall on another ET date.",
        "The likeliest ticket in 3-5x is mostly favourites, and every leg is the same model's opinion; "
        "the legs are different events, so their outcomes are close to independent.",
    ]
    return out


def table(payload):
    head = f"  {'rule':<14}{'period':<14}{'tickets':>8}{'wins':>6}{'hit':>7}{'model':>7}{'market':>7}{'mult':>7}{'legs':>6}{'ROI':>8}  [95% interval]"
    print(head)
    for name in ("same_day", "market_only"):
        for period in ("all", "last_3_years"):
            s = payload["results"][name][period]
            if not s.get("tickets"):
                print(f"  {name:<14}{period:<14}       0")
                continue
            print(f"  {name:<14}{period:<14}{s['tickets']:>8}{s['wins']:>6}{s['hit_rate']:>7.1%}"
                  f"{s['expected_hit_rate']:>7.1%}{(s['market_hit_rate'] or 0):>7.1%}{s['mean_multiplier']:>6.2f}x"
                  f"{s['mean_legs']:>6.1f}{s['roi']:>+8.3f}  [{s['roi_low']:+.3f}, {s['roi_high']:+.3f}]")
    for k in ("by_sports", "by_mix", "by_weekday", "by_year"):
        print(f"\n  same_day {k}:")
        for g, s in payload["splits"]["same_day"][k].items():
            if s.get("tickets"):
                print(f"    {g:<20}{s['tickets']:>6}{s['wins']:>6}{s['hit_rate']:>7.1%}{s['expected_hit_rate']:>7.1%}"
                      f"{s['roi']:>+8.3f}  [{s['roi_low']:+.3f}, {s['roi_high']:+.3f}]")
    print("\n  multipliers:", payload["splits"]["same_day"]["multipliers"])
    print("  legs used:", payload["splits"]["same_day"]["legs_used"])
    print(f"  days {payload['days']}, periods {payload['periods']}")


if __name__ == "__main__":
    main()
