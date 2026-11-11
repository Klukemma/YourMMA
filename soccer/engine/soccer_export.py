"""The soccer app's data files: matches.json, tables.json, performance.json.

Shapes are the contract with soccer/app/app.js; tests/test_soccer_export.py
pins them. The other apps' export rules carry over unchanged:

  * NaN and infinity become null (a bare NaN makes JSON.parse throw on the
    phone - a blank screen). A missing number prints as a dash, never as 0.
  * Every payload carries `schema` and `generated`.
  * Every caveat the reader sees is written here, in one place, and every
    number in one is read from the measured backtest - never typed in.

EVERY LEAGUE IN THE APP (soccer_leagues.in_app()). Each league carries what
was decided for it and measured (leagues(): its group, its pick - the
market recalibrated or raw -, the numbers it ships and withholds from each
source, its own periods, the price it was measured on, its season and its
label on its own calendar, and every caveat soccer_leagues holds for it,
never filtered), so a card is read against its own league's measurement,
never against the top five's. Beside those caveats, never in place of
them, is what the live feeds cover for it (soccer_feed.coverage /
coverage_caveat): seven leagues have no feed, so no fixture before
football-data.co.uk posts it and no provisional result. The global caveats
keep their keys - the app shows them all - with their text scoped to what
they were measured on: the pooled numbers are the top five's, and say so.

FORM AND TABLES OVER EVERY LEAGUE. The frame the app's form lines and tables
read holds about 228,000 matches with every league in it; soccer_form scans
the frame it is given on every call (about 44 ms each, thousands of calls a
run). View cuts it once by club and by league-season and hands each call
only the rows it would have found - the same answer, bit for bit - and
gives each league's clubs badge codes unique within the league
(soccer_teams.codes_for: no "DUN v DUN").
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import soccer_backtest as bt
import soccer_data as sd
import soccer_feed as sfeed
import soccer_form as sf
import soccer_leagues as slg
import soccer_scores as ssc
import soccer_teams as st

APP = sd.SOCCER / "app"
APP_DATA = APP / "data"
SCHEMA = 1

# The short label on the league picker and every result row, for every league
# football-data.co.uk carries: the country, and the tier where the country
# has more than one division in the data (the five keep the labels they had).
LEAGUE_SHORT = {
    "E0": "ENG", "SP1": "ESP", "I1": "ITA", "D1": "GER", "F1": "FRA",
    "E1": "ENG2", "SP2": "ESP2", "I2": "ITA2", "D2": "GER2", "F2": "FRA2",
    "E2": "ENG3", "E3": "ENG4", "EC": "ENG5", "SC0": "SCO", "SC1": "SCO2", "SC2": "SCO3", "SC3": "SCO4",
    "N1": "NED", "B1": "BEL", "P1": "POR", "T1": "TUR", "G1": "GRE",
    "ARG": "ARG", "AUT": "AUT", "BRA": "BRA", "CHN": "CHN", "DNK": "DEN", "FIN": "FIN", "IRL": "IRL",
    "JPN": "JPN", "MEX": "MEX", "NOR": "NOR", "POL": "POL", "ROU": "ROU", "RUS": "RUS", "SWE": "SWE",
    "SWZ": "SUI", "USA": "USA",
}

# What a league's table is not, where its format is not one double round
# robin summed (soccer_form: the table is the regular season's - play-off
# clubs, post-season rounds and a split's rounds are left out). Data beside
# the table, shown with it; None where the summed table is the official one.
_SPLIT = ("The regular season's table: after the split it stays as it stood then, while the official standings "
          "go on within each group of the split")
TABLE_NOTES = {
    **{c: _SPLIT + "." for c in ("DNK", "FIN", "G1", "SC0", "SWZ")},
    **{c: _SPLIT + ", with every club's points halved at the split." for c in ("B1", "AUT", "ROU")},
    "ARG": ("The annual table (tabla anual): both halves of the season, every zone together; the knock-out "
            "rounds are left out."),
    "MEX": "The general table (tabla general) of both halves of the season; the liguilla is left out.",
    "USA": "The overall table, both conferences together (the Supporters' Shield's); MLS Cup is left out.",
}


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write(name, payload, out_dir=APP_DATA):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = dict(payload, schema=SCHEMA, generated=payload.get("generated") or now_stamp())
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(_clean(body), indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def now_stamp(now=None):
    t = now or datetime.now(timezone.utc)
    if getattr(t, "tzinfo", None) is not None:
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------- the leagues

def league_name(code):
    """A league's name as a form line or result row prints it: the ten keep
    theirs; another league whose name is shared ("Super League": China,
    Greece, Switzerland) carries its country."""
    meta = sd.ALL_LEAGUES.get(code)
    if meta is None:
        return str(code)
    if code in sd.LEAGUES:
        return meta.name
    shared = sum(1 for m in sd.ALL_LEAGUES.values() if m.name == meta.name) > 1
    return f"{meta.name} ({meta.country})" if shared else meta.name


def season_label_of(code, season):
    """'2026 season' on the calendar year (ARG BRA CHN FIN IRL NOR SWE USA),
    '2026-27 season' on August-May - each league on its own calendar."""
    if season is None:
        return None
    season = int(season)
    meta = sd.ALL_LEAGUES.get(code)
    if meta is not None and meta.calendar == "calendar-year":
        return f"{season} season"
    return f"{season}-{(season + 1) % 100:02d} season"


def feeds_of(code):
    """The live sources a league has, by name (soccer_feed.coverage)."""
    c = sfeed.coverage(code)
    return [name for key, name in (("espn", "ESPN"), ("fixturedownload", "fixturedownload.com")) if c[key]]


def league_lead(code):
    """A league's own lead line, from what experiments/leagues.py measured for
    it (soccer_leagues.SHIPPED): what its pick is and how it scored on its
    confirmation period beside the market and the model alone - never the
    top five's numbers."""
    s = slg.SHIPPED.get(code)
    if not s:
        return None
    c = s["measured"]["confirm"]
    key = f"{c['seasons'][0]}-{c['seasons'][1]}"
    how = ("the betting market's chances, recalibrated on earlier matches" if s["pick"] == "recalibrated" else
           "the betting market's chances with the bookmakers' cut removed, not recalibrated (recalibrating "
           "them scored worse here on the earlier seasons)")
    price = "closing" if s["measured"]["price"] == "closing" else "pre-match"
    return (f"The pick is {how}. Across {key} ({c['matches']:,} matches, {price} prices) it picked "
            f"{pct(c.get('acc_pick'))} of results, the betting favourite {pct(c.get('acc_market'))} and the model "
            f"alone {pct(c.get('acc_model'))}.")


def _measured(code):
    m = (slg.SHIPPED.get(code) or {}).get("measured") or {}
    return {k: m[k] for k in ("dev", "confirm") if k in m}


def league_info(code, season=None, model_through=None, results_through=None, in_app=True):
    """Everything the app shows about one league (the module's notes)."""
    meta = sd.ALL_LEAGUES[code]
    s = slg.SHIPPED.get(code) or {}
    dev, conf = bt.periods_for(code)
    return {
        "name": meta.name, "label": league_name(code), "country": meta.country, "tier": meta.tier,
        "short": LEAGUE_SHORT.get(code, code), "group": slg.group_of(code), "calendar": meta.calendar,
        "season": None if season is None else int(season), "season_label": season_label_of(code, season),
        "in_app": bool(in_app), "pick": s.get("pick"), "model_alone": s.get("model_alone"),
        "price_measured": (s.get("measured") or {}).get("price"),
        "periods": {"dev": list(dev), "confirm": list(conf)},
        "numbers": s.get("numbers"),
        "withheld": {src: list(slg.withheld(code, src)) for src in slg.SOURCES},
        "measured": _measured(code), "lead": league_lead(code),
        # never filtered, never replaced: the coverage note goes beside them
        "caveats": slg.caveats(code),
        "coverage": sfeed.coverage(code), "coverage_caveat": sfeed.coverage_caveat(code), "feeds": feeds_of(code),
        "table_note": TABLE_NOTES.get(code),
        "model_through": model_through, "results_through": results_through,
    }


def leagues(codes=None, seasons=None, through=None, results_through=None, in_app=None):
    """{code: league_info} for `codes` (default the five top leagues), each
    with its season (`seasons`), the date of football-data.co.uk's latest
    result in it (`through`: the last its ratings include) and the latest
    result shown (`results_through`, with the feeds' provisional ones).
    in_app: the leagues the app carries (default `codes`): a code outside
    it is here only because a pick of it is shown."""
    codes = list(sd.TOP) if codes is None else list(codes)
    seasons, through, results_through = seasons or {}, through or {}, results_through or {}
    app = set(codes if in_app is None else in_app)
    return {c: league_info(c, seasons.get(c), through.get(c), results_through.get(c), c in app) for c in codes}


def groups_payload(codes):
    """{group: {name, codes}} of the leagues in the app (soccer_leagues.GROUPS)."""
    return {g: {"name": slg.GROUP_NAMES[g], "codes": [c for c in members if c in set(codes)]}
            for g, members in slg.GROUPS.items() if any(c in set(codes) for c in members)}


def bands_payload():
    """Each group's band hit rates on its confirmation period
    (soccer_leagues.BANDS, rule d): the pick's ("pick") and the model's own
    picks' ("model"), with the period they were measured on - what a card's
    band is read against, by its league's group and its band_basis."""
    out = {}
    for g, b in slg.BANDS.items():
        lo, hi = bt.periods_for(slg.GROUPS[g][0])[1]
        out[g] = {"period": f"{lo}-{hi}", "pick": b.get("pick"), "model": b.get("model")}
    return out


def band_floors():
    return {name: floor for name, floor in ssc.BANDS}


def not_in_app(codes):
    """The measured leagues the app does not carry (RUS), with why."""
    return {c: {"name": sd.ALL_LEAGUES[c].name, "country": sd.ALL_LEAGUES[c].country, "caveats": slg.caveats(c)}
            for c in slg.SHIPPED if c not in set(codes)}


def pct(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:.{d}f}%"


# ------------------------------------------------------------- the caveats

def caveats(bt_summary, basis="blend", model_through=None, scores=None, leagues=None):
    """What the reader is told, with every number read from the backtest.

    basis "market": the pooling gave the model no weight, so the pick is
    the market's chances recalibrated on earlier seasons, and is called
    that - the model's own chances are shown beside it, labelled.
    model_through: the date of football-data.co.uk's latest result in the
    app's leagues, the last the model's ratings include. scores: the goals
    and scores numbers' measured table (soccer_scores.measure) - the top
    five's. leagues: the app's leagues (default the five): the pooled
    numbers here are the top five's and say so; every other league's own
    are in its leagues entry (league_info)."""
    codes = list(sd.TOP) if leagues is None else list(leagues)
    others = [c for c in codes if c not in sd.TOP]
    res = (bt_summary or {}).get("result", {})
    conf = (bt_summary or {}).get("periods", {}).get("confirm", [2016, 2025])
    key = f"{conf[0]}-{conf[1]}"
    blend = (res.get("blend") or {}).get(key, {})
    model = (res.get("model") or {}).get(key, {})
    value = (bt_summary or {}).get("value", {})
    raw = slg.raw_leagues([c for c in codes if c in slg.SHIPPED])
    where = " in the top five" if others else ""
    lead = (f"The pick is the betting market's chances, recalibrated on earlier seasons: across {key}{where} the "
            f"model alone picked {pct(model.get('accuracy'))} of results and the market "
            f"{pct(blend.get('market_accuracy'))}, and blending the model in added nothing the market did not already "
            "know. The model's own chances are shown beside each pick, labelled"
            if basis == "market" else
            f"The pick is the model blended with the betting market, which is hard to beat at soccer: "
            f"across {key}{where} the blend picked {pct(blend.get('accuracy'))} of results against the betting "
            f"favourite's {pct(blend.get('market_accuracy'))}")
    if others:
        lead += (". Every league's pick is its own market's price, measured on its own seasons"
                 + (f" - the raw price, with the bookmakers' cut removed, where recalibrating it scored worse "
                    f"({', '.join(raw)})" if raw else "")
                 + "; each league's own numbers and caveats are with it")
    out = {
        "model": (
            lead
            + (". Betting the model where it disagreed with the price did not make money. "
               if value.get("enabled_at") is None else ". ")
            + "Treat a pick as an opinion with a measured hit rate, not an edge."),
        "draws": (
            "A draw is rarely the single likeliest result, so it is rarely the pick - even though about "
            "a quarter of matches end level. Its chance is always shown."),
        "news": (
            "The model does not read team news - injuries, suspensions, rotation. The betting market "
            "does, which is part of why the pick leans on it."),
    }
    through = f", through {_day(model_through)}" if model_through else ""
    out["feeds"] = (
        "Fixtures football-data.co.uk has not posted yet, and results it has not posted yet, come from two "
        "free, unofficial feeds - ESPN's public scoreboard and fixturedownload.com - which give dates, kickoff "
        "times and scores, never a price"
        + (" (in the leagues they cover: each league says which, and seven have neither)" if others else "")
        + ". Until football-data.co.uk posts a match with its prices (Tuesday and "
        "Friday afternoons) its chances are the model's alone, and say so. A result from the feeds is "
        "provisional, and marked so, until football-data.co.uk posts its own; a provisional grade it "
        "disagrees with is corrected. Where the feeds have not confirmed a kickoff time, the pick closes at "
        "midnight UK time on match day. The model's ratings use only football-data.co.uk's results, with "
        f"their shots where the league's history has them{through}.")
    sc = scores_caveat(scores, scope="top five" if others else None)
    if sc:
        out["scores"] = sc
    out["rating_short"] = ("The goal difference a match each club would expect against an average club "
                           "of its league on a neutral pitch, by the model's ratings, strongest first.")
    out["rating"] = ("The model rates each club's attack and defence from its goals, shots on target and "
                     "shots - from its goals alone where its league's history has no shots, as that league "
                     "says - weighted toward recent matches; a promoted club carries its rating up with a "
                     "measured step. The rating is the model's own view - the pick on each match is the "
                     "market's chances, because the market forecasts better.")
    return out


def scores_caveat(measured, scope=None):
    """What the reader is told about both teams to score, over 1.5 / 3.5 and
    the likeliest scores: every number read from soccer_scores.measure (one
    group's: scope names it, e.g. "top five"). None without a measurement.
    A league shows only the numbers it shipped (soccer_leagues.withheld), so
    a group's pooled measure never claims more than that."""
    periods = (measured or {}).get("periods") or {}
    if not periods:
        return None
    key = max(periods)                                     # the confirming period
    r = periods[key]
    if not r.get("matches"):
        return None
    beat = all(p.get(t, {}).get("shown", {}).get("log_loss", 9) < p.get(t, {}).get("base", {}).get("log_loss", 0)
               for p in periods.values() if p.get("matches") for t in ("btts", "o15", "o35", "exact"))
    b, ex = r["btts"], r["exact"]["shown"]
    top1 = ex.get("top1_hit")
    short = (b["rate"] - b["shown"]["mean_p"]) * 100
    model_short = (b["rate"] - b["model"]["mean_p"]) * 100
    one_in = f", about 1 in {round(1 / top1)}" if top1 else ""
    names = " and ".join(sorted(periods))
    where = f" in the {scope}" if scope else ""
    pooled = f", pooled over the {scope}," if scope else ""
    return (
        "Both teams to score, over 1.5 and 3.5 goals and the likeliest scores are read off one score matrix per "
        "match: the one that best reproduces the market's home/draw/away chances (the pick's) and its over/under "
        "2.5 price. "
        + (f"Measured on {names}{pooled} each beat the leagues' own rates from earlier seasons. " if beat else
           f"Measured on {names}{where}; not every one beat the leagues' own rates from earlier seasons (Method). ")
        + ("A league shows only the numbers that beat its own rates in both of its periods; the others are "
           "withheld and shown as missing. " if scope else "")
        + f"Both teams to score runs low: across {key}{where} its chance averaged {pct(b['shown']['mean_p'])} where "
        f"both teams scored in {pct(b['rate'])} of matches, low in {b.get('deciles_low', '—')} of "
        f"{b.get('deciles', '—')} deciles - read it as about {short:.0f} points short. Exact scores are long shots: "
        f"the likeliest score happened in {pct(top1)} of matches{one_in}. Where the prices fit no score matrix "
        f"within {measured.get('fit_max_miss', 0) * 100:.0f} points the card says how far the closest misses, and "
        f"on which price. Where no price is posted, or only the home/draw/away price, these numbers are the "
        f"model's own, and say so - measured too, a little less accurate (both teams to score log loss "
        f"{b['model']['log_loss']:.4f} against {b['shown']['log_loss']:.4f}, the leagues' own rates "
        f"{b['base']['log_loss']:.4f}), and running {model_short:.0f} points low rather than {short:.0f}.")


def _day(iso):
    """'2026-10-21' -> 'Oct 21' (the app's own dates are the phone's)."""
    try:
        d = datetime.strptime(str(iso)[:10], "%Y-%m-%d")
    except ValueError:
        return str(iso)
    return f"{d:%b} {d.day}"


# ------------------------------------------------------------- form, tables, badges

class View:
    """The frame the app's form lines and tables read (soccer_schedule.shown),
    cut once by club and by league-season (the module's notes): sf.form and
    sf.table read only the rows of `team`, and of the league-season, so each
    is handed those rows alone - the answer is the one the whole frame gives.
    seasons {league: season}: the seasons whose clubs get badge codes unique
    within their league (chip)."""

    def __init__(self, matches, seasons=None):
        self.matches = matches
        self._home = matches.groupby("home", sort=False).indices
        self._away = matches.groupby("away", sort=False).indices
        self._ls = {(lg, int(s)): pos for (lg, s), pos in matches.groupby(["league", "season"], sort=False).indices.items()}
        self._clubs, self._positions, self._chips = {}, {}, {}
        self.seasons = dict(seasons or {})

    def club(self, team):
        if team not in self._clubs:
            pos = np.union1d(self._home.get(team, np.empty(0, int)), self._away.get(team, np.empty(0, int)))
            self._clubs[team] = self.matches.iloc[pos.astype(int)]
        return self._clubs[team]

    def league_season(self, league, season):
        pos = self._ls.get((league, int(season)))
        return self.matches.iloc[pos] if pos is not None else self.matches.iloc[:0]

    def positions(self, league, season, before):
        k = (league, int(season), str(before)[:10])
        if k not in self._positions:
            self._positions[k] = {r["team"]: r["position"] for r in sf.table(self.league_season(league, season),
                                                                             league, int(season), before)
                                  if r["played"] > 0}
        return self._positions[k]

    def chip(self, league, key):
        """The club's badge (soccer_teams.display): its code unique within its
        league's season on show (soccer_teams.codes_for), the same whatever
        order the cards come in."""
        if league not in self._chips:
            season = self.seasons.get(league)
            rows = self.league_season(league, season) if season is not None else self.matches.iloc[:0]
            self._chips[league] = st.codes_for(set(rows["home"]) | set(rows["away"]))
        chips = self._chips[league]
        if key not in chips:
            chips[key] = st.display(key, {d["code"] for d in chips.values()})
        return chips[key]


def club(key, league=None, view=None):
    return view.chip(league, key) if view is not None and league is not None else st.display(key)


def form_payload(matches, team, before, n=5):
    """A club's last n results before `before` (soccer_form.form), with the
    opponent's name and the league's (league_name). `matches`: a frame or a
    View."""
    rows = sf.form(matches.club(team) if isinstance(matches, View) else matches, team, before, n)
    for r in rows:
        r["opponent_name"] = st.display(r["opponent"])["name"]
        r["league_name"] = league_name(r["league"])
    return rows


def positions(matches, league, season, before):
    if isinstance(matches, View):
        return matches.positions(league, season, before)
    return {r["team"]: r["position"] for r in sf.table(matches, league, season, before)
            if r["played"] > 0}


def tables_payload(matches, season, ratings=None, generated=None, caveat=None, season_label=None, leagues=None,
                   seasons=None):
    """Each league's table as it stands (`leagues`, default the five top
    leagues, in that order), with each club's last five and the model's
    rating where it has one - on its own season: seasons {league: season}
    (a calendar-year league's is not the five's), else `season` for all.
    `matches` (a frame or a View) may carry provisional results
    (soccer_schedule.shown): each row counts its own, each league says how
    many it has and the date of its latest football-data.co.uk result - the
    last its ratings include - with its caveats (never filtered) and what
    its table is not (table_note). A league with no club to list is left
    out, as before."""
    view = matches if isinstance(matches, View) else None
    frame = view.matches if view is not None else matches
    codes = list(sd.TOP) if leagues is None else list(leagues)
    seasons = seasons or {}
    out = []
    for code in codes:
        s = seasons.get(code, season)
        if s is None:
            continue
        rows_of = view.league_season(code, s) if view is not None else frame
        rows = sf.table(rows_of, code, s)
        if not rows:
            continue
        for r in rows:
            r["club"] = club(r["team"], code, view)
            r["form"] = sf.form(view.club(r["team"]) if view is not None else frame, r["team"], "9999-12-31")
            r["rating"] = (ratings or {}).get(r["team"])
        done = rows_of[(rows_of["league"] == code) & (rows_of["season"] == s) & rows_of["result"].notna()]
        prov = done["provisional"].fillna(False).astype(bool) if "provisional" in done else \
            pd.Series(False, index=done.index)
        official = done.loc[~prov, "date"]
        meta = sd.ALL_LEAGUES[code]
        out.append({"code": code, "name": meta.name, "label": league_name(code), "country": meta.country,
                    "short": LEAGUE_SHORT.get(code, code), "group": slg.group_of(code), "season": int(s),
                    "season_label": season_label_of(code, s), "rows": rows, "provisional": int(prov.sum()),
                    "model_through": official.max() if len(official) else None,
                    "results_through": done["date"].max() if len(done) else None,
                    "caveats": slg.caveats(code), "table_note": TABLE_NOTES.get(code)})
    return {"season": season, "season_label": season_label, "leagues": out, "generated": generated,
            "caveats": caveat or {}}
