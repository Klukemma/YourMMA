"""Build the JSON the phone app reads.

The engine is Python and runs in CI; the app is static files on a phone. The
only thing that crosses between them is this. Every payload here is built by a
pure function so it can be tested without running the twenty-minute pipeline,
and every one carries the caveats the numbers need - a probability shown on a
phone with no context is how a -1.8% ROI gets mistaken for an edge.

WHAT IS DELIBERATELY EXPORTED ALONGSIDE THE NUMBERS. Each payload names how it
was measured and what it is worth out of sample. The app is a place those
sentences are read, not a place they are dropped: the whole reason this project
found its leaks was that someone kept asking what a number actually meant.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3

# Sits beside every model probability shown in the app.
# The return figures in these two caveats are READ, not typed: they come from
# experiments/strategies.json, the file the Record tab shows. Typed by hand
# they had drifted - the card said -1.8% (one year's figure) where the
# confirm period's measured return is -0.2%, so two screens of one app
# disagreed. Missing file, missing figure: never a stale literal.
STRATEGIES = Path(__file__).resolve().parent / "experiments" / "strategies.json"


def _measured():
    try:
        return json.loads(STRATEGIES.read_text())
    except (OSError, ValueError):
        return {}


def _signed_pct(x):
    return f"{'+' if x >= 0 else '-'}{abs(x) * 100:.1f}%"


def model_caveat(measured=None):
    m = _measured() if measured is None else measured
    row = next((r for r in m.get("strategies", []) if r.get("label") == "MODEL"), None)
    text = "Walk-forward accuracy 61.7%, AUC 0.66."
    if row and row.get("roi") is not None and row.get("bets"):
        text += (f" Flat-stake ROI on the confirm period ({m.get('confirm_from', 2020)} onward, "
                 f"{row['bets']:,} priced bets) is {_signed_pct(row['roi'])}")
        if row.get("roi_low") is not None and row.get("roi_high") is not None:
            text += f", somewhere between {_signed_pct(row['roi_low'])} and {_signed_pct(row['roi_high'])}"
        text += "."
    return text + " Treat a pick as an opinion, not an edge."


def parlay_caveat(measured=None):
    m = _measured() if measured is None else measured
    row = next((r for r in m.get("parlays", []) if r.get("label") == "3 legs"), None)
    text = ("Combined probability multiplies the legs, which assumes the fights are "
            "independent; they are not quite.")
    if row and row.get("roi") is not None and row.get("parlays"):
        text += (f" Backtested on the confirm period, 3-leg parlays returned "
                 f"{_signed_pct(row['roi'])} over {row['parlays']:,} parlays - promising but about "
                 "1.5 standard deviations, and those parlays share legs, so this is NOT "
                 "an established edge.")
    return text


MODEL_CAVEAT = model_caveat()

SIMULATION_CAVEAT = (
    "20,000 simulations from both fighters' measured rates. As a winner "
    "picker this is weaker than the model (AUC 0.559 against 0.66), so the "
    "model's probability is the headline. Method and round are where the "
    "simulation is the only source, and there it finishes 55% of fights "
    "where the sport finishes 47% - read DEC as firmer and SUB as softer."
)

FLAG_CAVEAT = (
    "A second model, fitted only on bets that had already settled, guessing "
    "whether each pick is wrong. Its flags score about 0.70 where 0.5 is a "
    "coin toss - real, not decisive. Most of what it reads is the gap between "
    "this model and the market, so a fight nobody has priced cannot be "
    "scored and is shown as such rather than assumed safe."
)

INTEL_CAVEAT = (
    "Sourced news gathered before the bell - injuries, late replacements, a "
    "change of camp - about things the model cannot read from a record. NONE "
    "OF IT MOVES THE PREDICTION. The one soft factor we could label at scale, "
    "298 late replacements from Wikipedia, measured -3.5% against the model "
    "and -4.4% against the blend with both intervals spanning zero, needing "
    "roughly 1,800 labels to resolve. So this is shown beside the number, not "
    "folded into it: it tells you why a fight might not go the way the model "
    "thinks, and leaves the number something that was actually measured."
)

PARLAY_CAVEAT = parlay_caveat()


def _clean(value):
    """JSON has no NaN. A missing number must read as missing, never as 0."""
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return None
    # Rounded because this ships to a phone over mobile data and the sixth
    # decimal place of a strikes-per-minute rate is noise in the twelfth
    # significant figure of a float, not information. Six places keeps every
    # rate here well inside its own measurement error.
    return round(number, 6)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _method_probs(probs):
    """KO/TKO, Submission and Decision, or None if the model had no view."""
    if not probs:
        return None
    out = {k: _clean(v) for k, v in probs.items()}
    return out if any(v is not None for v in out.values()) else None


def _intel_payload(records):
    """Observations for one fight, or an empty list.

    `applies` is carried through rather than dropped so the app can never
    present a note as though it moved the prediction. If a weight is ever
    fitted, the flag changes here and the app can start saying so - until
    then every one of these is false and the app should say "not priced in".
    """
    out = []
    for record in records or []:
        out.append({
            "fighter": record.get("fighter"),
            "kind": record.get("kind"),
            "confidence": record.get("confidence"),
            "note": record.get("note"),
            "source": record.get("source"),
            "gathered": record.get("gathered"),
            "applies": bool(record.get("applies")),
            **({"replaced_opponent": record["replaced_opponent"]}
               if record.get("replaced_opponent") else {}),
        })
    return out


_RESULTS = ("W", "L", "D", "NC")


def _form_payload(form):
    """A fighter's pro record and last five (fighter_form.form), checked.

    None stays None - the app shows no record rather than an invented 0-0.
    A result outside W/L/D/NC, or a count that is not a whole number, means
    the producer is broken, and the whole record is dropped rather than
    shown half-right.
    """
    if not form:
        return None
    rec = form.get("record") or {}
    try:
        record = {k: int(rec[k]) for k in ("wins", "losses", "draws", "nc")}
    except (KeyError, TypeError, ValueError):
        return None
    last = []
    for bout in (form.get("last5") or [])[:5]:
        if bout.get("result") not in _RESULTS:
            return None
        last.append({"result": bout["result"],
                     "code": str(bout.get("code") or "??")[:3],
                     "date": bout.get("date"),
                     "opponent": bout.get("opponent"),
                     "event": bout.get("event"),
                     "method": bout.get("method")})
    return {"record": record, "last5": last,
            "source": form.get("source"),
            "verified": bool(form.get("verified")),
            "as_of": form.get("as_of")}


def card_payload(event_name, event_date, fights, skipped=(), parlays=None,
                 dataset_end=None, trained_on=None):
    """The card: one entry per fight, with the model and the simulation.

    `fights` entries carry whatever the pipeline has; anything absent comes out
    as null rather than as a confident zero. A fight the model refused belongs
    in `skipped` and keeps its reason, because "no data" is an answer the app
    has to be able to show.
    """
    out = []
    for fight in fights:
        simulation = fight.get("simulation")
        out.append({
            "number": fight.get("number"),
            "red": fight.get("red"),
            "blue": fight.get("blue"),
            "pick": fight.get("pick"),
            "win_prob": _clean(fight.get("win_prob")),
            # The model without the closing line, and the line itself. The
            # headline is blended and is what gets graded; these two are what
            # the edge is made of and are kept so a reader can see how much of
            # the prediction is the market.
            "win_prob_model": _clean(fight.get("win_prob_model")),
            "market_prob": _clean(fight.get("market_prob")),
            "confidence": _clean(fight.get("confidence")),
            "method": fight.get("method"),
            "method_prob": _clean(fight.get("method_prob")),
            # The whole distribution, not just its argmax. The top class on
            # its own is the least defensible thing this model produces:
            # walk-forward on the confirm period it is right 47.2% of the
            # time, where saying "Decision" every time is right 49.4%. What
            # the model is actually worth is the shape - it finds 53% of the
            # finishes at 58% precision, which a single word cannot carry.
            "method_probs": _method_probs(fight.get("method_probs")),
            # The binary, calibrated, plus what it was before the correction.
            # The app leads with the first and explains itself with the pair.
            "p_finish": _clean(fight.get("p_finish")),
            "p_finish_raw": _clean(fight.get("p_finish_raw")),
            "round": fight.get("round"),
            "recommendation": fight.get("recommendation"),
            "parlay_tier": fight.get("parlay_tier"),
            "odds": fight.get("odds"),
            "edge": _clean(fight.get("edge")),
            # The second layer. Absent, never 0.5, when it could not run: the
            # app must be able to tell a safe-looking pick from an unscored one.
            "p_fail": _clean(fight.get("p_fail")),
            "rounds_scheduled": fight.get("rounds_scheduled"),
            "title_fight": bool(fight.get("title_fight")),
            "simulation": _simulation_payload(simulation),
            # WHAT THE MODEL CANNOT SEE. Sourced observations gathered before
            # the bell - an injury, a late replacement, a new camp. They carry
            # `applies: false` because no weight is fitted for any kind, so
            # NONE of this has moved win_prob by a single point. It is shown
            # beside the number rather than folded into it, which is the
            # honest version of soft information: the reader is told why a
            # fight might not go the way the model thinks, and the number
            # stays something that was actually measured.
            "intel": _intel_payload(fight.get("intel")),
            # Each fighter's pro record and last five results, newest first
            # (fighter_form.py): for the reader, beside the name. Not a model
            # input, and only bouts dated before the card.
            "red_form": _form_payload(fight.get("red_form")),
            "blue_form": _form_payload(fight.get("blue_form")),
        })
    return {
        "schema": SCHEMA_VERSION,
        "generated": _now(),
        "event": {"name": event_name, "date": event_date},
        "dataset_end": dataset_end,
        "trained_on": trained_on,
        "fights": out,
        "skipped": [{"fight": s.get("fight"), "reason": s.get("reason"),
                     "red": s.get("red"), "blue": s.get("blue"),
                     "red_form": _form_payload(s.get("red_form")),
                     "blue_form": _form_payload(s.get("blue_form"))}
                    for s in skipped],
        "caveats": {"model": model_caveat(), "simulation": SIMULATION_CAVEAT,
                    "parlay": parlay_caveat(), "flag": FLAG_CAVEAT,
                    "intel": INTEL_CAVEAT},
        "parlays": parlays or [],
    }


def _simulation_payload(summary):
    if not summary:
        return None
    return {
        "red_win": _clean(summary.get("red_win")),
        "blue_win": _clean(summary.get("blue_win")),
        "draw": _clean(summary.get("draw")),
        "standard_error": _clean(summary.get("win_se")),
        "ko": _clean(summary.get("ko")),
        "sub": _clean(summary.get("sub")),
        "decision": _clean(summary.get("decision")),
        "finish_by_round": [_clean(p) for p in
                            summary.get("finish_by_round", [])],
        "decision_share": _clean(summary.get("decision_share")),
        "mean_seconds": _clean(summary.get("mean_seconds")),
        "median_seconds": _clean(summary.get("median_seconds")),
        # Which rates had to be assumed. A confident distribution resting on
        # league averages is the one thing a reader must be able to see.
        "assumed": sorted(set(summary.get("imputed_red", []))
                          | set(summary.get("imputed_blue", []))),
        "simulations": summary.get("simulations"),
    }


def american(p):
    """A probability as the American price that breaks even on it."""
    if p is None or not 0 < p < 1:
        return None
    return -int(round(100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def parlay_payload(candidates):
    """The suggested parlays, with the combined probability spelled out."""
    legs = [{"fight": c.get("matchup"), "pick": c.get("pick"),
             "prob": _clean(c.get("prob")), "tier": c.get("tier")}
            for c in candidates]
    out = []
    for size in (2, 3, 4):
        if len(legs) < size:
            continue
        chosen = legs[:size]
        combined = 1.0
        for leg in chosen:
            if leg["prob"] is None:
                combined = None
                break
            combined *= leg["prob"]
        out.append({
            "legs": chosen,
            "combined_prob": _clean(None if combined is None else round(combined, 6)),
            # What the bookmaker would have to pay for this to break even, as
            # an American price: a parlay more likely than not is a favourite
            # (-141 at 58%), not "+71", which no book would print.
            "fair_odds": american(combined),
        })
    return out


def fighters_payload(final_stats, columns=None):
    """Every fighter's current career state, for search and head-to-head.

    Keyed by fighter id with a name lookup beside it, because the app resolves
    a typed name to an id exactly once and then works in ids - two fighters
    genuinely share the name Bruno Silva in this dataset.

    COLUMNAR, because this is the one payload big enough to care. Repeating
    thirty-three key names across 2,617 fighters costs more bytes than the
    numbers do - it was 2.1 MB as objects and is about a third of that as
    parallel arrays. The app rebuilds the object for the one or two fighters
    it is actually showing.
    """
    frame = final_stats if columns is None else final_stats[list(columns)]
    names = list(frame.columns)
    ids = [str(i) for i in frame.index]
    values = [[_clean(v) for v in row] for row in frame.to_numpy()]
    return {
        "schema": SCHEMA_VERSION,
        "generated": _now(),
        "count": len(ids),
        "columns": names,
        "ids": ids,
        "values": values,
    }


def names_payload(name_by_id):
    """id -> display name, and the search index the app types against."""
    return {
        "schema": SCHEMA_VERSION,
        "generated": _now(),
        "names": {str(k): str(v) for k, v in name_by_id.items()},
    }


def by_event(history, limit=10):
    """Graded picks per card, newest first: [{date, event, graded, correct}].

    Counted by grade.py's own rules, so the cards add up to its unique-bout
    record: picks logged after their card never count, a bout is its most
    recent pick, and a bout logged twice (a card renamed and re-logged) is one
    bout - matched as grade.py matches, the same two fighters within its date
    tolerance (an Australian card is logged a day apart, sometimes across a
    month end). Keyed by the card's date, not its name.
    """
    from collections import Counter
    from grade import DATE_TOLERANCE, deduplicate
    from name_resolution import norm_name

    def day(pick):
        try:
            return datetime.strptime(str(pick.get("event_date"))[:10], "%Y-%m-%d")
        except ValueError:
            return None

    kept = []          # [(pair, day, pick)]
    for pick in deduplicate((history or {}).get("predictions", [])):
        if pick.get("status") not in ("correct", "incorrect") or day(pick) is None:
            continue
        pair = tuple(sorted((norm_name(pick.get("red_corner")), norm_name(pick.get("blue_corner")))))
        twin = next((i for i, (p, d, _) in enumerate(kept)
                     if p == pair and abs(d - day(pick)) <= DATE_TOLERANCE), None)
        if twin is None:
            kept.append((pair, day(pick), pick))
        elif str(pick.get("timestamp") or "") > str(kept[twin][2].get("timestamp") or ""):
            kept[twin] = (pair, day(pick), pick)
    cards = {}
    for _, _, pick in kept:
        card = cards.setdefault(str(pick.get("event_date") or "")[:10],
                                {"names": Counter(), "graded": 0, "correct": 0})
        card["names"][pick.get("event_name") or ""] += 1
        card["graded"] += 1
        card["correct"] += pick["status"] == "correct"
    rows = [{"date": date, "event": card["names"].most_common(1)[0][0],
             "graded": card["graded"], "correct": card["correct"]}
            for date, card in cards.items() if date]
    return sorted(rows, key=lambda r: r["date"], reverse=True)[:limit]


def upcoming(history, today=None):
    """Bouts picked and still to be fought: pending picks dated today or
    later, one per bout. "Pending" alone also counts picks whose fight never
    happened (a withdrawal, a renamed card), which will never be graded."""
    from name_resolution import norm_name
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bouts = {tuple(sorted((norm_name(p.get("red_corner")), norm_name(p.get("blue_corner")))))
             for p in (history or {}).get("predictions", [])
             if p.get("status") == "pending" and str(p.get("event_date") or "")[:10] >= today}
    return len(bouts)


def performance_payload(history, auc_by_year=None, backtest=None,
                        strategies=None):
    """The track record and what the strategies actually returned.

    The confirm-period figures are the ones that mean anything and the payload
    says so, because the tune-period +6.1% is exactly the number that would
    otherwise get screenshotted.
    """
    summary = (history or {}).get("summary", {})
    graded = summary.get("graded") or 0
    return {
        "schema": SCHEMA_VERSION,
        "generated": _now(),
        # TWO DIFFERENT DENOMINATORS, and conflating them put 59.3% on a tile
        # whose own sub-label read "128 of 204", which is 62.7%. The stored
        # accuracy is over UNIQUE BOUTS; correct/graded counts a bout once per
        # time it was predicted, so a card predicted three times counts thrice.
        # The unique figure is the honest one and the label now says so.
        "track_record": {
            "predictions": summary.get("total"),
            "graded": graded,
            "correct": summary.get("correct"),
            "pending": summary.get("pending"),
            "unique_bouts": summary.get("unique_bouts_graded"),
            "accuracy": _clean(summary.get("accuracy")),
            "accuracy_basis": "unique bouts",
            "by_event": by_event(history),
            "upcoming": upcoming(history),
        },
        "auc_by_year": auc_by_year or [],
        "roi_by_year": backtest or [],
        "strategies": strategies or [],
        "notes": {
            "which_number_counts":
                "The confirm period is 2020 onward. Anything before it was "
                "used to choose how the strategies work, so its +6.1% is not "
                "a forecast. The confirm number is.",
            "leak":
                "Figures published before 2026-09-18 came from a model that "
                "could see each fighter's final career record, which inflated "
                "historical accuracy to 78% and backtests to +16.2%. Those "
                "numbers were fiction and are not shown here.",
        },
    }


# Files big enough that pretty-printing costs real mobile data. They are
# rewritten wholesale on every sync anyway, so a readable diff buys nothing.
COMPACT = {"fighters", "names"}


def write_all(directory, payloads):
    """Write each payload to <directory>/<name>.json, creating the directory.

    Sorted keys and a trailing newline so a regenerated file that changed
    nothing produces an empty diff, which is what keeps the daily commit
    readable.

    allow_nan=False is the load-bearing argument: Python writes a bare NaN by
    default, which is not valid JSON, and JSON.parse throws on it. That is a
    blank screen on a phone rather than an error anyone can read.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in payloads.items():
        path = directory / f"{name}.json"
        if name in COMPACT:
            text = json.dumps(payload, sort_keys=True, allow_nan=False,
                              separators=(",", ":"))
        else:
            text = json.dumps(payload, indent=1, sort_keys=True,
                              allow_nan=False)
        path.write_text(text + "\n")
        written.append(path)
    return written


# Constants the app's JavaScript needs to reproduce the matchup advantages.
# EXPORTED RATHER THAN COPIED. A hand-transcribed constant in a second language
# is a silent divergence waiting to happen - the app would keep computing a
# striking advantage with last year's league baseline and nothing would say so.
# tests/test_app_export.py checks this list against the module itself.
MATCHUP_CONSTANTS = (
    "LEAGUE_SIG_LANDED_PER_MIN", "LEAGUE_SIG_ACC", "LEAGUE_SIG_ATT_PER_MIN",
    "LEAGUE_TD_ATT_PER_MIN", "LEAGUE_TD_ACC", "LEAGUE_CTRL_SEC_PER_MIN",
    "LEAGUE_SUB_ATT_PER_MIN", "LEAGUE_KD_PER_MIN",
    "LEAGUE_HEAD_ABSORBED_PER_MIN",
    "RATE_RATIO_FLOOR", "RATE_RATIO_CAP", "PROBABILITY_EPS",
    "K_STRIKE_VOLUME_MIN", "K_STRIKE_ABSORBED_MIN", "K_TD_RATE_MIN",
    "K_TD_CONCEDED_MIN", "K_CTRL_MIN", "K_CTRL_CONCEDED_MIN",
    "K_SUB_ATT_MIN", "K_SUB_CONCEDED_MIN",
    "K_SIG_ACC_ATT", "K_SIG_DEF_ATT", "K_TD_ACC_ATT", "K_TD_DEF_ATT",
    "STRIKE_ADV_SD", "TD_ADV_SD", "CTRL_ADV_SD", "SUB_ADV_SD",
    "GRAP_W_CTRL", "GRAP_W_SUB", "GRAP_W_TD",
    "SIZE_W_HEIGHT", "SIZE_W_REACH", "SIZE_W_APE",
    "HEIGHT_DIFF_SD_CM", "REACH_DIFF_SD_CM", "WEIGHT_DIFF_SD_KG",
    "HEIGHT_MIN_CM", "HEIGHT_MAX_CM", "REACH_MIN_CM", "REACH_MAX_CM",
)


def constants_payload(matchup_module):
    """Every constant the app needs, read straight off the module."""
    values = {}
    missing = []
    for name in MATCHUP_CONSTANTS:
        if not hasattr(matchup_module, name):
            missing.append(name)
            continue
        # NOT rounded. _clean trims a fighter's stats to six decimals to save
        # mobile data, which is far inside their measurement error - but a
        # CONSTANT is used as an exact divisor, and rounding
        # LEAGUE_SIG_ATT_PER_MIN (a derived 7.8730761...) moved the app's
        # striking advantage in the ninth decimal place against Python's.
        value = getattr(matchup_module, name)
        values[name] = None if value != value else float(value)
    if missing:
        raise AttributeError(
            "matchup.py is missing constants the app expects: "
            + ", ".join(missing))
    return {"schema": SCHEMA_VERSION, "generated": _now(),
            "matchup": values}
