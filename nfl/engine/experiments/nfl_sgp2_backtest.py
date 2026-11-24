"""The history behind nfl_sgp2's tickets (sgp-v2), and the choice of its two
numbers: every NFL game from nfl_sgp.HISTORY_FROM to the last one played,
read from v1's history inputs (nfl_sgp_backtest.inputs: the same
point-in-time blend, TD walk-forward, closing prices and rookie flag) and
graded as v1 grades, its four tickets built by nfl_sgp2.build_tickets.

    python nfl/engine/experiments/nfl_sgp2_backtest.py --choose
        chooses CONFIDENT and TD_FLOOR from nfl_sgp2's grids on DEV
        (2014-2019) ALONE, by the docstring's objective and tie-breaks,
        writes them into nfl_sgp2.py with the date and the DEV figures that
        chose them, then replays as below with them
    python nfl/engine/experiments/nfl_sgp2_backtest.py
        the replay with the constants nfl_sgp2.py holds
        -> nfl/data/sgp2_backtest.json           the meters, per kind and period
           nfl/data/sgp2_backtest_tickets.csv.gz  every v2 ticket, for audit
    python nfl/engine/experiments/nfl_sgp2_backtest.py --if-stale
        the same, only when stale() says the file on disk is behind - what
        the NFL runner calls before each predict run. It never re-chooses.

WHY THE CHOICE CANNOT SEE 2020 OR LATER. choose() keeps only the DEV
seasons' games before it builds a ticket, and every input of a DEV game is
itself point in time (a season's blend and TD chances come from models
fitted on the seasons before it). The one input chosen with hindsight is
v1's: which nfl_td usage groups get a chance at all was measured on
nfl_td.DEV and CONFIRM walk-forward calibration, so `td_groups` reports how
many v2 tickets it changes and what the DEV choice is with every group's
chance in.

WHAT IS REPORTED, beside the chosen pair's meters (report-only - none of it
can move the choice): every pair of the grid, per period; v1's tickets on
the same confident games; and SMALL with the game's best TD instead of the
pick's team's first (nfl_sgp2.game_best_small). And two things found after
the rule was fixed, with the chosen pair: `rises`, the docstring's "hit
rate always rises as the bars rise" against DEV's own grid (counted here,
never typed), and `untouched` (nfl_sgp2.UNTOUCHED_NOTE), that "untouched"
covers the two numbers, not SMALL's pick's-team clause, whose motivating
statistic's seasons overlap CONFIRM - said beside the SMALL-the-other-way
comparison too.

No anytime-TD price is on file for the past, so the return is missing with
nfl_sgp.NO_TD_PRICE, as in v1; and every TD candidate here took a snap
(nfl_sgp.KNOWN_BIAS, carried into the output).
"""

import argparse
import datetime as dt
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
for _p in (ENGINE, ENGINE / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nfl_data as nd               # noqa: E402
import nfl_export as ne             # noqa: E402
import nfl_sgp as sgp               # noqa: E402
import nfl_sgp2 as sgp2             # noqa: E402
import nfl_sgp_backtest as bk       # noqa: E402
import nfl_td as nt                 # noqa: E402

OUT_NAME = "sgp2_backtest"
TICKETS_OUT = nd.DATA / "sgp2_backtest_tickets.csv.gz"
RULE = ENGINE / "nfl_sgp2.py"
DEV_NAME = f"{sgp2.DEV[0]}-{sgp2.DEV[1]}"
# v1's bins, with its top one (40% and up) split: v2's tickets are the likelier ones.
CHANCE_EDGES = bk.CHANCE_EDGES[:-1] + (0.45, 0.50, 0.60, 1.0)
META = ("game_id", "season", "week", "game_type", "home", "away", "pick", "p_pick")
# The lines --choose writes (and rewrites) in nfl_sgp2.py: its comment, then the three constants.
CHOSEN_BLOCK = re.compile(r"(?:# CHOSEN [^\n]*\n(?:#   [^\n]*\n)*)?"
                          r"CONFIDENT = [^\n]*\nTD_FLOOR = [^\n]*\nCHOSEN_ON = [^\n]*\n")


# ---------------------------------------------------------------- the games

def records(games, blend, td_rows, prices=None, start=sgp.HISTORY_FROM):
    """Every game nfl_sgp_backtest replays (final, its box score on file) as
    a record: its meta, the pick (nfl_sgp2.pick_of the walk-forward blend),
    its legs (nfl_sgp.game_legs) and its outcome - plus the games left out
    and why. Building a record reads nothing of the game's result but the
    outcome it is graded on."""
    on_file, left_out = bk.replayable(games, blend, td_rows, prices, start)
    out = []
    for g, legs, result, meta in on_file:
        pick, p_pick = sgp2.pick_of(g.home, g.away, blend.get(g.game_id))
        out.append(dict(meta, pick=pick, p_pick=p_pick, legs=legs, result=result))
    return out, left_out


def _meta(rec):
    return {k: rec[k] for k in META}


def tickets(recs, confident, td_floor):
    """Every record's four v2 tickets, graded, with the game's meta."""
    return [dict(sgp.grade_ticket(t, r["result"]), **_meta(r)) for r in recs
            for t in sgp2.build_tickets(r["legs"], r["pick"], r["p_pick"], confident, td_floor)]


def v1_tickets(recs):
    """v1's four tickets of the same records, graded (report-only)."""
    return [dict(sgp.grade_ticket(t, r["result"]), **_meta(r)) for r in recs
            for t in sgp.build_tickets(r["legs"])]


def alt_small(recs, confident, td_floor):
    """SMALL with the game's best TD (nfl_sgp2.game_best_small), graded (report-only)."""
    return [dict(sgp.grade_ticket(sgp2.game_best_small(r["legs"], r["pick"], r["p_pick"], confident,
                                                       td_floor), r["result"]), **_meta(r))
            for r in recs]


def _in(rows, span):
    lo, hi = span
    return [r for r in rows if lo <= r["season"] <= hi]


# ---------------------------------------------------------------- the grid and the choice

def cell(recs, built, confident):
    """One pair's figures over one period's records and their tickets: the
    games, the confident ones, SMALL's tickets, coverage (tickets / games),
    won, lost, hit rate and mean chance, and every kind's tickets and hit rate."""
    m = sgp.summarise(built)
    games = len(recs)
    out = {"games": games, "confident_games": sum(sgp2.eligible(r["p_pick"], confident) for r in recs),
           "kinds": {k: {key: m[k][key] for key in ("tickets", "won", "lost", "void", "hit_rate", "mean_chance")}
                     for k in sgp2.KINDS}}
    small = dict(out["kinds"]["small"])
    small["coverage"] = small["tickets"] / games if games else None
    out["small"] = small
    return out


def grid(recs, spans):
    """Every pair of nfl_sgp2's grids, its cell() per period of `spans`."""
    rows = []
    for confident in sgp2.CONFIDENT_GRID:
        for td_floor in sgp2.TD_FLOOR_GRID:
            built = tickets(recs, confident, td_floor)
            rows.append({"confident": confident, "td_floor": td_floor,
                         "periods": {name: cell(_in(recs, span), _in(built, span), confident)
                                     for name, span in spans.items()}})
    return rows


def objective(cells):
    """The docstring's choice among `cells` (one per pair: its DEV cell()
    with `confident` and `td_floor`): of the pairs whose SMALL ticket is
    built in at least COVERAGE_MIN of the games, the highest SMALL hit rate,
    ties to the larger coverage, then the lower CONFIDENT, then the lower
    TD_FLOOR - compared as exact fractions, so a tie is a tie. Returns
    (the chosen cell or None, how many pairs cleared the coverage floor)."""
    floor = Fraction(str(sgp2.COVERAGE_MIN))
    ok = [c for c in cells if c["games"] and Fraction(c["small"]["tickets"], c["games"]) >= floor
          and c["small"]["won"] + c["small"]["lost"] > 0]

    def key(c):
        s = c["small"]
        return (Fraction(s["won"], s["won"] + s["lost"]), Fraction(s["tickets"], c["games"]),
                -Fraction(str(c["confident"])), -Fraction(str(c["td_floor"])))
    return (max(ok, key=key) if ok else None), len(ok)


def dev_cells(rows):
    """The DEV cell of each grid() row, with its pair."""
    return [dict(r["periods"][DEV_NAME], confident=r["confident"], td_floor=r["td_floor"]) for r in rows]


def _rate(cell_):
    s = cell_["small"]
    return Fraction(s["won"], s["won"] + s["lost"]) if s["won"] + s["lost"] else None


def rises(cells):
    """nfl_sgp2.RISES_CLAIM against `cells` (dev_cells: one per pair), found
    after the rule was fixed and report-only - it cannot move the choice.
    Every step of the grid (one bar raised one grid step, the other held)
    whose two SMALL hit rates are on file, compared as exact fractions:
    how many fell (and which), how many stayed level; the best pair with no
    coverage floor, by the objective's own tie-breaks; and whether the floor
    stopped the choice (that pair is not the chosen one). With the words
    nfl_sgp2.rises_note makes of them."""
    by = {(c["confident"], c["td_floor"]): c for c in cells}
    conf, tds = sgp2.CONFIDENT_GRID, sgp2.TD_FLOOR_GRID
    steps, level, fell = 0, 0, []
    for i, c in enumerate(conf):
        for j, t in enumerate(tds):
            for a, b in ((i + 1, j), (i, j + 1)):
                if a >= len(conf) or b >= len(tds) or (c, t) not in by or (conf[a], tds[b]) not in by:
                    continue
                lo, hi = _rate(by[(c, t)]), _rate(by[(conf[a], tds[b])])
                if lo is None or hi is None:
                    continue
                steps += 1
                if hi < lo:
                    fell.append({"from": [c, t], "to": [conf[a], tds[b]],
                                 "hit_rates": [float(lo), float(hi)]})
                level += hi == lo
    rated = [c for c in cells if c["games"] and _rate(c) is not None]
    best = max(rated, key=lambda c: (_rate(c), Fraction(c["small"]["tickets"], c["games"]),
                                     -Fraction(str(c["confident"])), -Fraction(str(c["td_floor"])))) if rated else None
    chosen, _ = objective(cells)
    out = {"report_only": True, "claim": sgp2.RISES_CLAIM, "period": DEV_NAME, "steps": steps,
           "falls": len(fell), "level": level, "fell": fell,
           "best_without_floor": None if best is None else {
               "confident": best["confident"], "td_floor": best["td_floor"],
               "coverage": best["small"]["tickets"] / best["games"], "hit_rate": float(_rate(best))},
           "floor_binds": (best is not None and chosen is not None
                           and (best["confident"], best["td_floor"]) != (chosen["confident"], chosen["td_floor"]))}
    out["note"] = sgp2.rises_note(out)
    return out


def choose(recs):
    """The two numbers from DEV ALONE: the records of seasons outside
    nfl_sgp2.DEV are dropped before any ticket is built, so nothing of 2020
    or later can reach the choice. Returns {confident, td_floor, dev (the
    chosen pair's DEV cell), cleared, pairs}, or None when no pair clears
    COVERAGE_MIN."""
    rows = grid(_in(recs, sgp2.DEV), {DEV_NAME: sgp2.DEV})
    best, cleared = objective(dev_cells(rows))
    if best is None:
        return None
    dev = {k: v for k, v in best.items() if k not in ("confident", "td_floor")}
    return {"confident": best["confident"], "td_floor": best["td_floor"], "dev": dev,
            "cleared": cleared, "pairs": len(rows)}


def chosen_block(choice, today):
    """nfl_sgp2.py's lines for a choice: a comment with the date and the DEV
    figures that chose it, generated from the run, then the constants."""
    s, games = choice["dev"]["small"], choice["dev"]["games"]
    return (f"# CHOSEN {today} by experiments/nfl_sgp2_backtest.py --choose, on {DEV_NAME} alone:\n"
            f"#   SMALL built in {s['coverage']:.1%} of the {games:,} games ({s['tickets']:,} tickets), "
            f"hit {s['hit_rate']:.1%} (won {s['won']:,}, lost {s['lost']:,});\n"
            f"#   {choice['cleared']} of the {choice['pairs']} pairs built SMALL in "
            f"{sgp2.COVERAGE_MIN:.0%} or more of the games.\n"
            f"CONFIDENT = {choice['confident']!r}\nTD_FLOOR = {choice['td_floor']!r}\n"
            f'CHOSEN_ON = "{today}"\n')


def write_choice(choice, path=RULE, today=None):
    """Write the choice into nfl_sgp2.py, replacing the constants (and any
    earlier choice's comment); the docstring and everything else stay."""
    today = today or dt.datetime.now(dt.timezone.utc).date().isoformat()
    path = Path(path)
    text = path.read_text()
    new, n = CHOSEN_BLOCK.subn(lambda _: chosen_block(choice, today), text)
    if n != 1:
        raise RuntimeError(f"{path}: expected one CONFIDENT/TD_FLOOR/CHOSEN_ON block, found {n}")
    path.write_text(new)
    return today


# ---------------------------------------------------------------- the payload

def _coverage(meters, games):
    return {name: (m["tickets"] / games[name] if games.get(name) else None) for name, m in meters.items()}


def alternative(recs, built, alt, spans, no_price):
    """SMALL with the game's best TD beside v2's SMALL, per period: its
    meter, coverage, and how many v2 SMALL tickets it changes."""
    key = lambda t: tuple(leg["key"] for leg in t["legs"])           # noqa: E731
    v2 = {t["game_id"]: key(t) for t in built if t["kind"] == "small" and t["built"]}
    out = {}
    for name, span in spans.items():
        rows, games = _in(alt, span), len(_in(recs, span))
        m = sgp.meter(rows, no_price)
        mine = [t for t in rows if t["built"]]
        out[name] = {"meter": m, "coverage": m["tickets"] / games if games else None,
                     "differs": sum(v2.get(t["game_id"]) != key(t) for t in mine)}
    return out


def payload_of(data, recs, left_out, confident, td_floor, chosen_on, every_recs=None):
    """Everything sgp2_backtest.json holds, for the pair (confident, td_floor)."""
    games, prices = data["games"], data["prices"]
    current = int(games["season"].max())
    spans = bk.periods(current)
    no_price = sgp.NO_TD_PRICE if prices is None else None
    built = tickets(recs, confident, td_floor)
    out = bk.measure(built, current, no_price, CHANCE_EDGES)
    for kind, k in out["kinds"].items():
        k["coverage"] = _coverage(k["meters"], out["games"])
    out["confident_games"] = {name: sum(sgp2.eligible(r["p_pick"], confident) for r in _in(recs, span))
                              for name, span in spans.items()}
    rows = grid(recs, spans)
    now, cleared = objective(dev_cells(rows))
    dev = next(c for c in dev_cells(rows) if (c["confident"], c["td_floor"]) == (confident, td_floor))
    out["chosen"] = {
        "confident": confident, "td_floor": td_floor, "chosen_on": chosen_on, "on": list(sgp2.DEV),
        "objective": sgp2.OBJECTIVE, "coverage_min": sgp2.COVERAGE_MIN,
        "grids": {"confident": list(sgp2.CONFIDENT_GRID), "td_floor": list(sgp2.TD_FLOOR_GRID)},
        "dev": {k: v for k, v in dev.items() if k not in ("confident", "td_floor")},
        "cleared": cleared, "pairs": len(rows),
        "reproduced": now is not None and (now["confident"], now["td_floor"]) == (confident, td_floor),
        "rises": rises(dev_cells(rows)), "untouched": sgp2.UNTOUCHED_NOTE}
    out["grid"] = {"report_only": True, "rows": rows,
                   "note": "every pair of the grid, per period; only the DEV figures chose, and only by "
                           "the objective"}
    same = [r for r in recs if sgp2.eligible(r["p_pick"], confident)]
    v1 = bk.measure(v1_tickets(same), current, no_price, CHANCE_EDGES)
    out["v1_same_games"] = {"report_only": True, "version": sgp.VERSION, "games": v1["games"],
                            "kinds": {k: v["meters"] for k, v in v1["kinds"].items()},
                            "note": f"v1's four tickets of the games v2 builds in (the pick's chance "
                                    f"{confident:.0%} or more), measured as v1's own history is"}
    out["alt_small"] = {"report_only": True,
                        "rule": f"SMALL with the game's best TD at {td_floor:.0%} or more, whichever team, "
                                "instead of the pick's team's first",
                        "untouched": sgp2.UNTOUCHED_NOTE,
                        "periods": alternative(recs, built, alt_small(recs, confident, td_floor), spans,
                                               no_price)}
    groups = {"shipped": data["ships"], "note": bk.groups_note()}
    if every_recs is not None:
        groups.update(bk.changed_by_groups(built, tickets(every_recs, confident, td_floor)))
        every = choose(every_recs)
        groups["dev_choice_with_every_group"] = (None if every is None else
                                                 {"confident": every["confident"], "td_floor": every["td_floor"]})
    played = {r["game_id"] for r in recs}
    out.update(
        version=sgp2.VERSION, rule="nfl/engine/nfl_sgp2.py", v1_rule="nfl/engine/nfl_sgp.py",
        history_from=sgp.HISTORY_FROM, current_season=current, stake=sgp.STAKE_UNIT,
        through=games.loc[games["game_id"].isin(played), "kickoff"].max(),
        left_out=left_out, roi_reason=no_price, known_bias=sgp.KNOWN_BIAS, no_edge=sgp.NO_EDGE,
        no_edge_v2=sgp2.NO_EDGE, payout_basis=sgp.EST_PAYOUT, est_erratum=sgp.EST_ERRATUM,
        td_prices={"file": str(bk.TD_PRICES.relative_to(nd.NFL.parent)), "on_file": prices is not None,
                   "rows": 0 if prices is None else len(prices)},
        td_groups=groups, inputs=dict(bk.INPUTS, pick="nfl_predict's pick of the same blend: the home "
                                                      "side at P(home) 0.5 or more"))
    return out, built


def audit_rows(built):
    """v1's audit row per ticket, with the game's pick and its chance."""
    df = bk.ticket_rows(built)
    df.insert(6, "pick", [t["pick"] for t in built])
    df.insert(7, "p_pick", [t["p_pick"] for t in built])
    return df


# ---------------------------------------------------------------- run

def run(choose_pair=False, out_dir=nd.DATA, tickets_out=TICKETS_OUT, price_path=bk.TD_PRICES, rule=RULE):
    t0 = time.time()
    data = bk.inputs(price_path)
    recs, left_out = records(data["games"], data["blend"], data["td_rows"], data["prices"])
    if choose_pair:
        choice = choose(recs)
        if choice is None:
            raise SystemExit(f"no pair of the grid builds SMALL in {sgp2.COVERAGE_MIN:.0%} of "
                             f"{DEV_NAME}'s games; nothing chosen, nfl_sgp2.py unchanged")
        confident, td_floor = choice["confident"], choice["td_floor"]
        chosen_on = write_choice(choice, rule)
        print(f"chose CONFIDENT {confident} and TD_FLOOR {td_floor} on {DEV_NAME}; wrote {rule}")
    else:
        confident, td_floor, chosen_on = sgp2.CONFIDENT, sgp2.TD_FLOOR, sgp2.CHOSEN_ON
        if confident is None or td_floor is None:
            raise SystemExit(sgp2.NOT_CHOSEN)
    every = bk.td_history(data["td_frame"], nt.GROUPS, data["player_games"], data["preds"])
    every_recs, _ = records(data["games"], data["blend"], every, data["prices"])
    payload, built = payload_of(data, recs, left_out, confident, td_floor, chosen_on, every_recs)
    path = ne.write(OUT_NAME, payload, out_dir=out_dir)
    audit_rows(built).to_csv(tickets_out, index=False, compression="gzip")
    print(f"wrote {path} and {tickets_out} in {time.time() - t0:.0f}s")
    return payload


def stale(path=None, games=None, player_games=None, snaps=None):
    """Why the v2 history on file needs replaying, or None when it is
    current: nfl_sgp_backtest.stale's reasons for v2's version, or a pair
    on file that is not the one nfl_sgp2.py holds."""
    path = Path(path) if path is not None else nd.DATA / f"{OUT_NAME}.json"
    why = bk.stale(path, games, player_games, snaps, version=sgp2.VERSION)
    if why is not None:
        return why
    chosen = json.loads(path.read_text()).get("chosen") or {}
    on_file, rule = (chosen.get("confident"), chosen.get("td_floor")), (sgp2.CONFIDENT, sgp2.TD_FLOOR)
    if on_file != rule:
        return f"the history on file is for CONFIDENT, TD_FLOOR = {on_file}, the rule's are {rule}"
    return None


def headline(payload):
    c = payload["chosen"]
    s = c["dev"]["small"]
    lines = [bk.headline(payload),
             f"  chosen on {DEV_NAME} ({c['chosen_on']}): CONFIDENT {c['confident']}, TD_FLOOR "
             f"{c['td_floor']} - SMALL in {s['coverage']:.1%} of {c['dev']['games']:,} games, hit "
             f"{s['hit_rate']:.1%}; {c['cleared']} of {c['pairs']} pairs cleared; reproduced: {c['reproduced']}",
             f"  {c['rises']['note']}" if (c.get("rises") or {}).get("note") else "",
             "  coverage: " + "; ".join(f"{k} " + ", ".join(f"{n} {v:.1%}" for n, v in x["coverage"].items()
                                                               if v is not None)
                                         for k, x in payload["kinds"].items()),
             "  v1 on the same games (report-only):"]
    for kind, meters in payload["v1_same_games"]["kinds"].items():
        lines.append(f"    {kind:7s} " + "  ".join(
            f"{name} {m['tickets']}/{'-' if m['hit_rate'] is None else format(m['hit_rate'], '.1%')}"
            for name, m in meters.items()))
    return "\n".join(line for line in lines if line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--choose", action="store_true",
                        help=f"choose CONFIDENT and TD_FLOOR on {DEV_NAME} and write them into nfl_sgp2.py")
    parser.add_argument("--if-stale", action="store_true",
                        help="replay only when the history on file is behind (stale()); never chooses")
    args = parser.parse_args(argv)
    if args.choose and args.if_stale:
        parser.error("--if-stale never chooses")
    if args.if_stale:
        if sgp2.CONFIDENT is None or sgp2.TD_FLOOR is None:
            print(f"{sgp2.VERSION}: {sgp2.NOT_CHOSEN}; not replayed")
            return None
        why = stale()
        if why is None:
            print(f"{sgp2.VERSION}: the history on file is current; not replayed")
            return None
        print(f"{sgp2.VERSION}: replaying - {why}")
    payload = run(choose_pair=args.choose)
    print(headline(payload))
    return payload


if __name__ == "__main__":
    main()
