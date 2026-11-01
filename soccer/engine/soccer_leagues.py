"""Which leagues the app carries, and what it shows for each - as data.

experiments/leagues.py measured every league football-data.co.uk carries
(soccer_data.ALL_LEAGUES), walk-forward, and decided by rules declared
before any number was seen (its docstring has the protocol, every table and
the caveats). This module is what it decided, and the rules it decided by:
the experiment imports the rules below, and a test re-applies them to the
numbers stored here, so a decision can never drift from its measurement.

THE RULES (rule_* below):
  (a) in the app   a current season in the data (a result in the 45 days
                   before the check, or a fixture posted) - rule_current;
                   the pick is the market recalibrated if that was not worse
                   in log loss than the raw no-vig market on the league's
                   development period, else the raw no-vig market - rule_pick
  (b) model alone  a match with no price yet gets the model's own pick only
                   in a league where the model beat the league's base rate
                   (its home / draw / away frequencies over earlier seasons)
                   in log loss in BOTH periods; elsewhere "no pick until
                   priced" - rule_model_alone
  (c) numbers      both teams to score, over 1.5 / 2.5 / 3.5 and the exact
                   score each ship for a league only where the source the
                   card would read them from beat the league's base rate in
                   log loss in BOTH periods - the market-implied matrix
                   ("market", where the 1X2 and over/under are priced) and
                   the model's own ("model") judged apart - rule_numbers
  (d) bands        edges 0.40 / 0.50 / 0.60 as before; hit rates measured
                   per league group on the confirmation period - BANDS

HOW THE PIPELINE USES IT. smod.predict(frame, leagues=MODEL_LEAGUES,
england=ENGLAND_POOL) - the model over every league measured, so each
group's regression is the one measured (RUS's matches train the extra
leagues' even while RUS is not shown); smod.pick_leagues(frame, preds,
in_app(), PICK_WEIGHTS, raw=raw_leagues(), model_alone=model_alone());
ssc.card(..., withhold=withheld(code, source)); band_rate(code, band,
basis); caveats(code) on every card and on the Method tab.
"""

import soccer_data as sd

CHECKED_ON = "2026-10-30"           # the day rule (a)'s current-season check was run
RECENT_DAYS = 45
NUMBERS = ("btts", "o15", "o25", "o35", "exact")
SOURCES = ("market", "model")
GROUPS = {
    "top": list(sd.TOP),
    "europe": [c for c in sd.MAIN if c not in sd.TOP],
    "extra": list(sd.EXTRA),
}
GROUP_NAMES = {"top": "top five", "europe": "other European divisions", "extra": "extra leagues"}
MODEL_LEAGUES = list(sd.ALL_LEAGUES)      # the model runs over every league it was measured on


def group_of(code):
    """'top', 'europe' or 'extra'."""
    return next(g for g, codes in GROUPS.items() if code in codes)


# ----------------------------------------------------------- the rules

def rule_current(days_since_result, fixtures_posted, days=RECENT_DAYS):
    """(a): a current season in the data - a result within `days` of the
    check (days_since_result: None where the league has none), or any
    fixture posted for it."""
    return bool((days_since_result is not None and days_since_result <= days) or fixtures_posted > 0)


def season_check(matches, fixtures, code, today=CHECKED_ON, days=RECENT_DAYS):
    """(a)'s facts for one league: its newest result and how many days
    before `today` it was played, and the fixtures posted for it (on file -
    football-data's fixture files list matches not yet played - dated no
    earlier than `days` before `today`); "ok" is rule_current on them."""
    import pandas as pd
    res = matches[(matches["league"] == code) & matches["result"].notna()]
    last = res["date"].max() if len(res) else None
    since = (pd.Timestamp(today) - pd.Timestamp(last)).days if last else None
    since_day = (pd.Timestamp(today) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    posted = int(((fixtures["league"] == code) & (fixtures["date"] >= since_day)).sum()) if len(fixtures) else 0
    return {"ok": rule_current(since, posted, days), "last_result": last, "days_since": since,
            "fixtures_posted": posted, "season": int(res["season"].max()) if len(res) else None}


def rule_pick(dev_recalibrated, dev_raw):
    """(a): 'recalibrated' unless the recalibrated market's development log
    loss is worse (higher) than the raw no-vig market's on the same matches;
    then 'raw'. Nothing measured: 'raw'."""
    if dev_recalibrated is None or dev_raw is None:
        return "raw"
    return "recalibrated" if dev_recalibrated <= dev_raw else "raw"


def rule_model_alone(dev, confirm):
    """(b): True where the model's log loss beat the base rate's ({"model",
    "base"} per period, same matches) in BOTH periods."""
    return all(p.get("model") is not None and p.get("base") is not None and p["model"] < p["base"]
               for p in (dev, confirm))


def rule_numbers(dev, confirm):
    """(c): the NUMBERS a source ships for a league - each whose log loss beat
    the base rate's in BOTH periods. dev / confirm: {number: {"log_loss",
    "base"}} (a period with no matches: {} - nothing ships)."""
    def beats(p, k):
        x = p.get(k) or {}
        return x.get("log_loss") is not None and x.get("base") is not None and x["log_loss"] < x["base"]
    return [k for k in NUMBERS if beats(dev, k) and beats(confirm, k)]


# ----------------------------------------------------------- the decisions
# Filled from experiments/leagues.py's run (its main() prints this block).

ENGLAND_POOL = None
PICK_WEIGHTS = None
SHIPPED = {}
BANDS = {}
PRICE_GAP = {}


# ----------------------------------------------------------- reading them

def in_app(matches=None, fixtures=None, today=None):
    """The leagues the app carries, in soccer_data's order: as decided on
    CHECKED_ON - or, given the data and the day, every measured league whose
    current season rule (a) finds in that data now (a league's season ends,
    RUS's file resumes: its measured numbers and decisions stand)."""
    if matches is None:
        return [c for c in sd.ALL_LEAGUES if c in SHIPPED and SHIPPED[c]["in_app"]]
    return [c for c in sd.ALL_LEAGUES if c in SHIPPED
            and season_check(matches, fixtures, c, today or CHECKED_ON)["ok"]]


def raw_leagues():
    """Leagues whose pick is the raw no-vig market (rule a)."""
    return [c for c in in_app() if SHIPPED[c]["pick"] == "raw"]


def model_alone():
    """Leagues whose unpriced matches get the model's own pick (rule b)."""
    return [c for c in in_app() if SHIPPED[c]["model_alone"]]


def withheld(code, source):
    """The NUMBERS a card of `code` whose score matrix comes from `source`
    ("market" / "model") must not show (rule c) - every one for a league or
    a source that was not measured."""
    shown = (SHIPPED.get(code) or {}).get("numbers", {}).get(source)
    return tuple(k for k in NUMBERS if not shown or k not in shown)


def band_rate(code, band, basis="pick"):
    """The measured hit rate (and count) of `band` for `code`'s group on the
    confirmation period: basis "pick" for a market pick, "model" for a pick
    by the model alone. None where not measured."""
    g = BANDS.get(group_of(code)) or {}
    return (g.get(basis) or {}).get(band)


def caveats(code):
    """The caveats a league's cards and the Method tab carry (never deleted)."""
    return list((SHIPPED.get(code) or {}).get("caveats", []))
