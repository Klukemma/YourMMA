"""Match logged predictions against real results and report how the model did.

    python engine/grade.py            # report only, writes nothing
    python engine/grade.py --write    # also update statuses in the history file

Predictions are logged on every run of predict_card.py, so the same bout can
appear several times. Grading counts each bout once - its most recent
prediction - otherwise a card that happened to be re-run three times would
carry three times the weight in the accuracy figure.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_resolution import norm_name

ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = ENGINE_DIR / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
HISTORY = Path(os.environ.get("PREDICTIONS_LOG", DATA_DIR / "prediction_history.json"))

# How far a logged event date may sit from the recorded one and still match.
# Cards get rescheduled; names plus a nearby date is a safer key than the date.
DATE_TOLERANCE = timedelta(days=10)


def picks_may_be_logged(event_date, now=None, first_start=None):
    """Whether picks for a card dated `event_date` are made before the fight.

    With the card's first start time on file (the odds feed's commence_time),
    true only before it. Without one, true up to and including the event's own
    date (UTC): cards run in the evening and a fight-day run has the freshest
    line. After that the result may already be in the data and a pick is not a
    prediction - and grading keeps the newest pick for a bout, so a late one
    would replace the real one. An unreadable date is not loggable.
    """
    from datetime import date, datetime, timezone
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if first_start:
        try:
            start = datetime.fromisoformat(str(first_start).replace("Z", "+00:00"))
        except ValueError:
            start = None
        if start is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            return now < start
    try:
        day = date.fromisoformat(str(event_date)[:10])
    except ValueError:
        return False
    return now.astimezone(timezone.utc).date() <= day


def made_before_the_card(pred):
    """A logged pick that is a prediction: logged before its card's first
    start where the pick recorded one, else no later than the card's date -
    the cutoff the engine applied when it logged. A pick with no timestamp or
    no readable event date is kept: there is nothing to show it was late."""
    from datetime import date, datetime
    stamp, event = pred.get('timestamp'), pred.get('event_date')
    try:
        when = datetime.fromisoformat(str(stamp))
        date.fromisoformat(str(event)[:10])
    except (TypeError, ValueError):
        return True
    return picks_may_be_logged(event, now=when, first_start=pred.get('first_start'))


def categorise_method(method):
    """Dataset method text -> the three classes the model predicts.

    The newest archive rows (Sept-Dec 2025) carry the short labels - U-DEC,
    S-DEC, M-DEC - and reading only the word "decision" left 60 fights of
    the 3-year window with no method at all: graded on the winner, silently
    dropped from the method record, and a method leg on one could never
    settle."""
    if not isinstance(method, str):
        return None
    text = method.lower()
    if 'decision' in text or re.search(r'(^|[\s-])dec\b', text):
        return 'Decision'
    if 'sub' in text:
        return 'Submission'
    if 'ko' in text or 'tko' in text:
        return 'KO/TKO'
    return None            # Overturned, Could Not Continue, DQ


def load_results(csv_path):
    """{(norm_a, norm_b): [(date, winner, method)]} for both corner orders."""
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        red, blue = row.r_name, row.b_name
        if not isinstance(red, str) or not isinstance(blue, str):
            continue
        entry = (row.date, row.winner, row.method)
        index[(norm_name(red), norm_name(blue))].append(entry)
        index[(norm_name(blue), norm_name(red))].append(entry)
    return index


def find_result(prediction, index):
    """The recorded outcome for a prediction, or None."""
    key = (norm_name(prediction['red_corner']), norm_name(prediction['blue_corner']))
    candidates = index.get(key)
    if not candidates:
        return None
    predicted_date = pd.to_datetime(prediction['event_date'], errors='coerce')
    if pd.isna(predicted_date):
        return candidates[0]
    within = [c for c in candidates if abs(c[0] - predicted_date) <= DATE_TOLERANCE]
    if not within:
        return None
    return min(within, key=lambda c: abs(c[0] - predicted_date))


def deduplicate(predictions):
    """One prediction per logged bout - the most recent.

    This is a first pass only. It cannot catch a bout logged twice under
    different event dates, which happens when a card is re-logged after being
    renamed; deduplicate_graded does that once the real bout is known.
    """
    latest = {}
    for pred in predictions:
        # A pick logged after its card is not a prediction, and as the newest
        # it would replace the real one. The log holds such picks (re-runs of
        # old cards); they stay in the file and never count.
        if not made_before_the_card(pred):
            continue
        key = (norm_name(pred['red_corner']), norm_name(pred['blue_corner']),
               str(pred.get('event_date'))[:7])          # same bout, same month
        stamp = pred.get('timestamp') or ''
        if key not in latest or stamp > (latest[key].get('timestamp') or ''):
            latest[key] = pred
    return list(latest.values())


def deduplicate_graded(graded):
    """One row per real bout, keyed on the outcome it matched.

    The same fight was logged under both "UFC 325 - Australia" and
    "UFC 325 - Sydney, Australia" with different event dates, so keying on
    what was logged leaves duplicates. Keying on the matched bout - its actual
    date and the two fighters - cannot.
    """
    latest = {}
    for row in graded:
        pair = tuple(sorted((norm_name(row['red_corner']), norm_name(row['blue_corner']))))
        key = (str(row['actual_date'])[:10], pair)
        stamp = row.get('timestamp') or ''
        if key not in latest or stamp > (latest[key].get('timestamp') or ''):
            latest[key] = row
    return list(latest.values())


def grade(predictions, index):
    """Attach outcomes. Returns (graded, ungraded)."""
    graded, ungraded = [], []
    for pred in predictions:
        result = find_result(pred, index)
        if result is None:
            ungraded.append((pred, 'no matching bout in the dataset'))
            continue
        date, winner, method = result
        if not isinstance(winner, str) or not winner.strip():
            ungraded.append((pred, 'draw or no contest'))
            continue
        row = dict(pred)
        row['actual_winner'] = winner
        row['actual_method'] = categorise_method(method)
        row['actual_date'] = date
        row['correct'] = norm_name(winner) == norm_name(pred['predicted_winner'])
        graded.append(row)
    return graded, ungraded


def _brier(rows):
    return float(np.mean([(r['win_probability'] - (1.0 if r['correct'] else 0.0)) ** 2
                          for r in rows]))


def _log_loss(rows):
    eps = 1e-15
    return float(np.mean([
        -np.log(max(r['win_probability'] if r['correct'] else 1 - r['win_probability'], eps))
        for r in rows]))


def report(graded, ungraded, total_logged):
    n = len(graded)
    if n == 0:
        print("Nothing could be graded.")
        return
    correct = sum(r['correct'] for r in graded)
    acc = correct / n

    print("=" * 66)
    print("PREDICTION ACCURACY")
    print("=" * 66)
    print(f"  logged predictions : {total_logged}")
    print(f"  unique bouts       : {n + len(ungraded)}")
    print(f"  graded             : {n}")
    print(f"  correct            : {correct}")
    print(f"  ACCURACY           : {acc:.1%}")
    print(f"  Brier score        : {_brier(graded):.4f}")
    print(f"  Log loss           : {_log_loss(graded):.4f}")

    # A coin flip and always-take-the-favourite are the bars worth clearing.
    print(f"\n  vs coin flip       : {acc - 0.5:+.1%}")

    print("\n" + "-" * 66)
    print("CALIBRATION - does a stated 70% actually win 70% of the time?")
    print("-" * 66)
    print(f"  {'confidence band':<18}{'n':>5}{'predicted':>12}{'actual':>10}{'gap':>9}")
    bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bands:
        rows = [r for r in graded if lo <= r['win_probability'] < hi]
        if not rows:
            continue
        stated = float(np.mean([r['win_probability'] for r in rows]))
        actual = float(np.mean([r['correct'] for r in rows]))
        print(f"  {lo:.0%}-{hi if hi <= 1 else 1:.0%}{'':<11}{len(rows):>5}"
              f"{stated:>11.1%}{actual:>10.1%}{actual - stated:>+9.1%}")

    print("\n" + "-" * 66)
    print("METHOD")
    print("-" * 66)
    with_method = [r for r in graded if r['actual_method']]
    if with_method:
        hits = sum(r['method_predicted'] == r['actual_method'] for r in with_method)
        print(f"  method accuracy    : {hits}/{len(with_method)} = {hits/len(with_method):.1%}")
        for cls in ('KO/TKO', 'Submission', 'Decision'):
            rows = [r for r in with_method if r['method_predicted'] == cls]
            if rows:
                h = sum(r['actual_method'] == cls for r in rows)
                print(f"    predicted {cls:<12}{h:>4}/{len(rows):<4} = {h/len(rows):.1%}")

    print("\n" + "-" * 66)
    print("BY EVENT")
    print("-" * 66)
    by_event = defaultdict(list)
    for r in graded:
        by_event[r['event_name']].append(r)
    for event, rows in sorted(by_event.items(), key=lambda kv: -len(kv[1])):
        h = sum(r['correct'] for r in rows)
        print(f"  {h:>2}/{len(rows):<3} {h/len(rows):>6.0%}  {event}")

    if ungraded:
        print("\n" + "-" * 66)
        print(f"NOT GRADED ({len(ungraded)})")
        print("-" * 66)
        reasons = defaultdict(int)
        for _, reason in ungraded:
            reasons[reason] += 1
        for reason, count in reasons.items():
            print(f"  {count:>3}  {reason}")
        for pred, reason in ungraded[:10]:
            print(f"       {pred['event_date']}  {pred['red_corner']} vs {pred['blue_corner']}")


def write_back(graded, ungraded, path, index, now=None):
    """Record outcomes in the history file.

    EACH LOGGED PICK AGAINST ITS OWN BOUT. This used to key the outcome on the
    pair of names and stamp it, with the graded pick's `correct`, onto every
    log of that pair: 25 rows ended up contradicting their own pick (a second
    log picking the loser marked right), and a rematch logged months later
    would have been graded before the fight. Now every row is matched by the
    same date-tolerant rule grade() uses, judged by its own predicted winner,
    and left pending when no bout matches it.
    """
    history = json.loads(Path(path).read_text())
    for pred in history['predictions']:
        result = find_result(pred, index)
        if result is None:
            continue
        _, winner, _ = result
        if not isinstance(winner, str) or not winner.strip():
            continue
        correct = norm_name(winner) == norm_name(pred.get('predicted_winner'))
        pred['actual_winner'] = winner
        pred['correct'] = bool(correct)
        pred['status'] = 'correct' if correct else 'incorrect'
    graded_rows = [p for p in history['predictions'] if p.get('status') in ('correct', 'incorrect')]
    history['summary'] = {
        'total': len(history['predictions']),
        'graded': len(graded_rows),
        'correct': sum(p['correct'] for p in graded_rows),
        'pending': len(history['predictions']) - len(graded_rows),
        'unique_bouts_graded': len(graded),
        'accuracy': round(sum(r['correct'] for r in graded) / len(graded), 4) if graded else None,
    }
    report_parlays(grade_parlays(history, index, now=now), history)
    Path(path).write_text(json.dumps(history, indent=2))
    print(f"\nWrote outcomes to {path}")


# ------------------------------------------------------------------ parlays

def _corner(winner, red, blue):
    """'red' / 'blue' for the recorded winner, or None when the name is
    neither fighter's - a spelling the matcher cannot place is left open
    rather than guessed into a win or a loss."""
    w = norm_name(winner)
    if w == norm_name(red):
        return 'red'
    if w == norm_name(blue):
        return 'blue'
    return None


def _bout(red, blue, event_date):
    return {'red_corner': red, 'blue_corner': blue, 'event_date': event_date}


def is_no_ticket(row):
    """A logged run of a band that built no ticket (status "none"): no
    multiplier, or no legs. Read from what the row holds, not its status,
    because grading rewrites the status and a row with nothing to settle
    must never be settled."""
    return row.get('multiplier') is None or not row.get('legs')


def _day(value):
    """A logged or matched date as "YYYY-MM-DD", or None when unreadable."""
    when = pd.to_datetime(value, errors='coerce')
    return None if pd.isna(when) else str(when)[:10]


def _logged_cards(history, results):
    """{card date as logged: Counter of the result dates its bouts matched},
    over every logged bout - a pick or a parlay leg - whose result is on file.

    WHY A CARD WITH RESULTS IS NEEDED BEFORE A MISSING BOUT IS CANCELLED.
    Results arrive a card at a time (the UFCStats dataset and the provisional
    Wikipedia rows both add whole events), so before any of a card's results
    are in, a bout with no record means only "not yet". Being in is not
    enough on its own either - see _cancelled.

    WHICH REAL CARD A LOGGED DATE IS. One card was logged under two dates
    (UFC 325 as 2026-01-31 and as 2026-02-08), and the date-tolerant match
    finds the same bouts from both. The result dates its bouts matched say
    which card a logged date was."""
    from collections import Counter
    found = defaultdict(Counter)

    def add(day, red, blue):
        hit = find_result(_bout(red, blue, day), results)
        real = None if hit is None else _day(hit[0])
        if real is not None:
            found[day][real] += 1

    for p in history.get('predictions', []):
        add(str(p.get('event_date'))[:10], p.get('red_corner'), p.get('blue_corner'))
    for t in history.get('parlays', []):
        day = str(t.get('event_date'))[:10]
        for leg in t.get('legs') or []:
            add(day, leg.get('red'), leg.get('blue'))
    return found


def _fighter_dates(results):
    """{norm_name: [dates]}: every date each fighter has a result row on."""
    out = defaultdict(list)
    for (name, _), entries in results.items():
        out[name].extend(e[0] for e in entries)
    return out


def _instant(text):
    from datetime import datetime, timezone
    try:
        when = datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def card_is_complete(ticket, now):
    """Whether a card is old enough for its results to be whole: a day after
    its first bell, or two days after its date when no start was logged. An
    unreadable date never is."""
    from datetime import date, datetime, timezone
    start = _instant(ticket.get('first_start')) if ticket.get('first_start') else None
    if start is not None:
        return now >= start + timedelta(days=1)
    try:
        day = date.fromisoformat(str(ticket.get('event_date'))[:10])
    except ValueError:
        return False
    return now >= datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=2)


def _cancelled(leg, day, fighters):
    """Whether a leg whose bout has no result was a bout that did not happen:
    NEITHER fighter has any result row within DATE_TOLERANCE of the card.

    A missing match alone is not enough. "Zachary Reese vs Mickey Gall" was
    logged and the archive holds "Mickey Gall vs Zach Reese": the bout
    happened, the names did not match, and voiding it as cancelled would have
    paid a lost ticket. Where either fighter fought near the card, the leg
    stays open - shown as missing - until a person reconciles the names."""
    when = pd.to_datetime(day, errors='coerce')
    if pd.isna(when):
        return False
    for name in (leg.get('red'), leg.get('blue')):
        for seen in fighters.get(norm_name(name), ()):
            if not pd.isna(seen) and abs(seen - when) <= DATE_TOLERANCE:
                return False
    return True


def _newest(rows, key):
    """{key(row): index of its newest row by timestamp}, rows as (i, row)."""
    best = {}
    for i, t in rows:
        k = key(i, t)
        if k not in best or (str(t.get('timestamp') or '')
                             > str(best[k][1].get('timestamp') or '')):
            best[k] = (i, t)
    return {k: i for k, (i, _) in best.items()}


def _supersede(t, status='superseded'):
    t['status'] = status
    for stale in ('paid', 'profit'):
        t.pop(stale, None)


def grade_parlays(history, results, now=None):
    """Settle the logged parlay tickets in `history["parlays"]`, in place,
    against `results` (load_results' index). Returns the graded tickets.

    THE RUN IS THE UNIT. Every run of the card logs one row per band - a
    ticket, or a "none" row saying that run built nothing - and per card and
    profile the newest row logged before the card is the one that counts,
    which is the rule the picks follow and for the same reason: a re-run
    logs the card again, and a row logged after the first bell is not a
    prediction. A "none" row is never graded or counted, so a later run that
    built nothing supersedes an earlier ticket rather than letting it stand.
    The other rows stay in the file, marked "superseded" or "late".

    ONE CARD, ONE TICKET, WHATEVER DATE IT WAS LOGGED UNDER. After settling,
    the chosen rows are collapsed per profile on the real card their legs
    matched (the result date find_result returned; a row with no matched leg
    takes the card its logged date's other bouts matched, else its logged
    date), keeping the newest: UFC 325, logged as 2026-01-31 and as
    2026-02-08, counts once.

    Each leg is read off its bout's real result by the same date-tolerant
    match the picks use (find_result, on the full names). A draw or no
    contest voids the leg. A bout with no result is voided as cancelled only
    when its card has results in, neither fighter has a result within
    DATE_TOLERANCE of it (_cancelled), and the card is old enough to be
    complete (card_is_complete, at `now`); otherwise the leg stays open.
    The ticket is then settled by parlay_profiles.settle at the decimals
    that were logged, a void leg counting 1.0 - so a ticket is lost the
    moment one leg loses, and open ("pending") while a leg is still unknown
    and none has lost.

    Every chosen ticket is settled again from scratch on every run, never
    trusted from the last one: a provisional result replaced, or a bout that
    arrives late, then changes the grade instead of leaving a stale one.
    """
    from collections import Counter
    from datetime import datetime, timezone
    import parlay_profiles as pp
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    tickets = history.get('parlays') or []
    cards = _logged_cards(history, results)
    fighters = _fighter_dates(results)

    before = [(i, t) for i, t in enumerate(tickets) if made_before_the_card(t)]
    chosen = set(_newest(before, lambda i, t: (str(t.get('event_date'))[:10],
                                               t.get('profile'))).values())

    real = {}                  # index -> the real card date it settled on
    for i, t in enumerate(tickets):
        if i not in chosen:
            _supersede(t, 'superseded' if made_before_the_card(t) else 'late')
            continue
        day = str(t.get('event_date'))[:10]
        logged_as = cards[day].most_common(1)[0][0] if cards.get(day) else day
        if is_no_ticket(t):
            _supersede(t, 'none')
            real[i] = logged_as
            continue
        complete = card_is_complete(t, now)
        rebuilt, matched = [], Counter()
        for leg in t.get('legs', []):
            found = find_result(_bout(leg.get('red'), leg.get('blue'), day), results)
            why = None
            if found is None:
                gone = day in cards and complete and _cancelled(leg, day, fighters)
                result = {'winner': None, 'method': None} if gone else None
                why = 'cancelled' if gone else None
            else:
                when, winner, method = found
                if _day(when) is not None:
                    matched[_day(when)] += 1
                if not isinstance(winner, str) or not winner.strip():
                    result, why = {'winner': None, 'method': None}, 'draw or no contest'
                else:
                    corner = _corner(winner, leg.get('red'), leg.get('blue'))
                    result = (None if corner is None
                              else {'winner': corner, 'method': categorise_method(method)})
            rebuilt.append({'side': leg.get('side'), 'market': leg.get('market') or 'moneyline',
                            'decimal': float(leg['decimal']), 'result': result})
            leg['outcome'] = pp.leg_outcome(rebuilt[-1])
            if why and leg['outcome'] == 'void':
                leg['void_reason'] = why
            else:
                leg.pop('void_reason', None)
        settled = pp.settle({'legs': rebuilt})
        t['status'] = 'pending' if settled['status'] == 'open' else settled['status']
        t['paid'] = settled['paid']
        t['profit'] = settled['profit']
        real[i] = matched.most_common(1)[0][0] if matched else logged_as

    keep = set(_newest([(i, tickets[i]) for i in real],
                       lambda i, t: (real[i], t.get('profile'))).values())
    graded = []
    for i in sorted(real):
        t = tickets[i]
        if i not in keep:
            _supersede(t)
        elif not is_no_ticket(t):
            graded.append(t)
    return graded


def report_parlays(graded, history):
    """One line per profile: what its settled tickets returned, and how many
    cards' runs built no ticket."""
    if not history.get('parlays'):
        return
    print("\n" + "-" * 66)
    print("PARLAY TICKETS (one a card per profile, logged before the card)")
    print("-" * 66)
    by = defaultdict(list)
    for t in graded:
        by[t.get('profile')].append(t)
    none = defaultdict(int)
    for t in history.get('parlays') or []:
        if t.get('status') == 'none':
            none[t.get('profile')] += 1
    for profile in list(by) + [p for p in none if p not in by]:
        rows = by.get(profile, [])
        done = [t for t in rows if t['status'] in ('won', 'lost', 'void')]
        won = sum(t['status'] == 'won' for t in done)
        units = sum(t['profit'] for t in done)
        print(f"  {str(profile):<12}{len(done):>4} settled  {won:>3} won  "
              f"{units:+8.2f} units  {len(rows) - len(done):>3} pending  "
              f"{none.get(profile, 0):>3} no ticket")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='update the history file')
    args = ap.parse_args()

    history = json.loads(HISTORY.read_text())
    predictions = history['predictions']
    unique = deduplicate(predictions)
    index = load_results(UFC_CSV)
    graded, ungraded = grade(unique, index)
    before = len(graded)
    graded = deduplicate_graded(graded)
    if len(graded) != before:
        print(f"collapsed {before - len(graded)} bouts logged twice under "
              f"different event names\n")
    report(graded, ungraded, len(predictions))
    if args.write:
        # The parlay tickets are settled inside write_back, on the history it
        # writes, so one write carries the picks and the tickets together.
        write_back(graded, ungraded, HISTORY, index)
    else:
        import copy
        shown = copy.deepcopy(history)
        report_parlays(grade_parlays(shown, index), shown)


if __name__ == '__main__':
    main()
