"""football-data.co.uk's fixtures and results joined with the free feeds'
(soccer_feed.py): which matches are still to play, when each pick freezes,
which are postponed, and which results are provisional.

football-data.co.uk is the source of record. It posts the weekend's fixtures,
with the prices the pick is built from, on Friday afternoon and the midweek's
on Tuesday afternoon, and its results days late. The feeds (ESPN's public
scoreboard and fixturedownload.com) list the next rounds and the scores of the
round just played long before that; they carry no price the engine reads.
So, one rule each:

  LEAGUES. The ones the caller passes (soccer_predict: the app's,
      soccer_leagues.in_app()): the feeds' matches and football-data.co.uk's
      fixtures of any other league are left out - RUS has a feed and is out
      of the app, and a feed match of it would otherwise be picked by the
      model alone, unpriced. Told nothing, the feeds' matches of the five
      (sd.TOP) and every fixture given: the run before every league was
      measured, bit for bit.
  A MATCH IS ONE ROW. A pairing (league, season, home, away) is played once a
      league season - EXCEPT in the leagues whose format repeats one
      (soccer_feed.REPEATS: a split, a third round robin, play-offs). There
      a match is its pairing AND its day (key): a pairing played earlier in
      the season does not make a fixture "played", only a result of the
      pairing within soccer_feed.PAIR_DAYS of its date does (the same match,
      a day or two apart between the sources); and a feed match joins the
      football-data.co.uk fixture of its pairing nearest its date within
      PAIR_DAYS (closest first, each once: soccer_feed.legs' rule), or none.
      A lookup by the pairing alone there would give one meeting another's
      status, kickoff or result (find_key does it by date). Where
      football-data.co.uk has the fixture, its row is the
      match - its id and its prices; the feeds only add their kickoff time. A
      feed match football-data.co.uk has not posted is added with no price:
      the model's own chances, labelled (price_status "not_yet").
  KICKOFF, the one shown - the match's day, its id, when "kicked off"
      shows: football-data.co.uk's own time where it has the fixture (or
      the feeds' confirmed time, where that is earlier); for a match only
      the feeds list, ESPN's valid time, else fixturedownload's
      (soccer_feed.reconcile: fixturedownload's later rounds were measured
      stale, ESPN tracks a move).
  CLOSE (freeze_at). At the shown kickoff when that time is confirmed -
      football-data.co.uk's own, or ESPN and fixturedownload agreeing; at
      00:00 UK time on its day when it is not - and EARLIER, at any
      source's time before that, so no pick stays open past any source's
      kickoff. A source's time that has passed while ESPN still lists the
      match as scheduled later is not the kickoff and is left out. Where
      the sources' times differ, each is kept (kickoff_sources) for the card.
  POSTPONED or CANCELLED (per ESPN) on football-data.co.uk's date for it: not
      picked, shown as such (with the pick, where one was logged before -
      soccer_predict). A feed match ESPN says has started, finished or was
      abandoned is never picked, whatever its listed time.
  MOVED. A football-data.co.uk fixture whose time has passed with no result
      while ESPN lists the match as scheduled more than MOVED later was
      rescheduled and its fixtures file has not caught up: ESPN's date
      stands, with no price, until football-data.co.uk posts the new one.
      ESPN, because it is the feed that reports status: fixturedownload's
      "scheduled" only means no score posted yet, and its times were
      measured stale (soccer_feed). Where pairings repeat, only a move
      within PAIR_DAYS is seen as one: a feed match further away may be the
      pairing's next meeting, so it stands as a match of its own (no price),
      and the passed fixture is dropped like any other.
  PROVISIONAL RESULTS. A feed result (soccer_feed's accepted "final": ESPN
      final and fixturedownload not contradicting it) for a match
      football-data.co.uk has not posted. It grades a pick provisionally
      (soccer_log.grade), and is listed in the results and the tables,
      flagged, dated by ESPN's kickoff (the final's own time). It never
      reaches the model: the frame returned here for the model holds
      football-data.co.uk's results only. Feeds that contradict each other
      on a score give no result, and the card says why (disputes).
  NO DATE. A feed match with no kickoff time from either source (a round the
      league has not scheduled yet) has no id and nothing to freeze a pick
      at: shown as "date and time to be confirmed", never picked.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

import soccer_data as sd
import soccer_feed as sfeed

OFF = ("postponed", "cancelled")
# A feed kickoff this long after a passed one, for a match with no result:
# the match was moved, not merely listed at another time.
MOVED = pd.Timedelta(hours=3)
# A feed status that means the match is under way or over: never picked.
STARTED = ("live", "final", "abandoned", "disputed", "unconfirmed")
FEED_SOURCE = "ESPN + fixturedownload.com"
FD_SOURCE = "football-data.co.uk"


def pairing(league, season, home, away):
    return (league, int(season), home, away)


def repeats(league):
    """Whether the league plays a pairing more than once a season
    (soccer_feed.REPEATS): there a pairing is not a match."""
    return league in sfeed.REPEATS


def _iso(day):
    """'YYYY-MM-DD' of a date, a Timestamp or text; None (or NaN) stays None."""
    if day is None or (not isinstance(day, str) and pd.isna(day)):
        return None
    return day[:10] if isinstance(day, str) else pd.Timestamp(day).strftime("%Y-%m-%d")


def days_apart(a, b):
    """Whole days between two dates ('YYYY-MM-DD', dates or Timestamps)."""
    return abs((date.fromisoformat(_iso(a)) - date.fromisoformat(_iso(b))).days)


def key(league, season, home, away, day=None):
    """What a match is filed under in a Schedule's status, feed_kickoff,
    disputed and disputes (and its off and tbc entries' "key"): its pairing
    where a pairing is played once a season; its pairing and its UK date
    ('YYYY-MM-DD') where pairings repeat (REPEATS), since the pairing alone
    is two matches there. Look one up with find_key, which also finds the
    match a day or two away (the sources dating it apart)."""
    p = pairing(league, season, home, away)
    return p + (_iso(day),) if repeats(league) else p


def uk_day(kickoff):
    """The Europe/London date ('YYYY-MM-DD') of a kickoff (naive UTC, aware,
    or '...Z' text): the date football-data.co.uk, and so a match_id, gives it."""
    k = naive(kickoff)
    return None if k is None else k.tz_localize("UTC").tz_convert(sd.UK).strftime("%Y-%m-%d")


def key_of(p):
    """key() of a logged pick (league, season, home, away and its kickoff) or
    of this module's off / tbc entries (their date, or date_hint)."""
    day = p.get("date") or p.get("date_hint") or uk_day(p.get("kickoff"))
    return key(p["league"], p["season"], p["home"], p["away"], day)


def find_key(keys, k):
    """The key in `keys` (a Schedule's dict or set) of the match key() `k`
    names: k itself where it is there; where pairings repeat, else the key
    of k's pairing nearest k's day within soccer_feed.PAIR_DAYS (the same
    match dated a day or two apart - its pick logged on one source's date,
    the feeds now on another's); None if there is none. Never a meeting
    further away: in REPEATS that may be the pairing's next one."""
    if k in keys:
        return k
    if len(k) < 5 or k[4] is None:
        return None
    best = None
    for other in keys:
        if len(other) == 5 and other[:4] == k[:4] and other[4] is not None:
            gap = days_apart(other[4], k[4])
            if gap <= sfeed.PAIR_DAYS and (best is None or (gap, other[4]) < best[0]):
                best = ((gap, other[4]), other)
    return best[1] if best else None


def lookup(table, k, default=None):
    """table[find_key(table, k)], else `default`."""
    found = find_key(table, k)
    return table[found] if found is not None else default


def feed_day(m):
    """A reconciled feed match's UK date, or the day it is filed under where
    it has no time (its date_hint)."""
    return m.get("uk_date") or m.get("date_hint")


def feed_key(m):
    return key(m["league"], m["season"], m["home"], m["away"], feed_day(m))


def played_pairs(matches):
    """football-data.co.uk's results by pairing: {pairing: [their dates]}
    (is_played reads it)."""
    done = matches[matches["result"].notna()]
    out = defaultdict(list)
    for lg, season, home, away, day in zip(done["league"], done["season"].astype(int), done["home"], done["away"],
                                           done["date"]):
        out[(lg, season, home, away)].append(day)
    return dict(out)


def is_played(played, league, season, home, away, day=None):
    """Whether football-data.co.uk has this match's result: any of its
    pairing's this season where a pairing is played once; one within
    PAIR_DAYS of `day` where pairings repeat (played_pairs). A plain set of
    pairings is read the first way for every league."""
    k = pairing(league, season, home, away)
    if k not in played:
        return False
    if not repeats(league) or not isinstance(played, dict):
        return True
    return _iso(day) is not None and any(days_apart(d, day) <= sfeed.PAIR_DAYS for d in played[k])


def join_repeats(fx, feed):
    """Where pairings repeat: {football-data.co.uk fixture index: the feed
    match of its pairing nearest its date within PAIR_DAYS} - closest pairs
    first, each fixture and each feed match once (soccer_feed.legs' rule
    between the two feeds). A fixture with no feed match that near has none."""
    by = defaultdict(list)
    for j, m in enumerate(feed):
        if repeats(m["league"]) and feed_day(m):
            by[pairing(m["league"], m["season"], m["home"], m["away"])].append(j)
    gaps = []
    for i, lg, season, home, away, day in zip(fx.index, fx["league"], fx["season"], fx["home"], fx["away"],
                                              fx["date"]):
        if not repeats(lg) or _iso(day) is None:
            continue
        for j in by.get(pairing(lg, season, home, away), ()):
            gap = days_apart(day, feed_day(feed[j]))
            if gap <= sfeed.PAIR_DAYS:
                gaps.append((gap, i, j))
    used_i, used_j, out = set(), set(), {}
    for _, i, j in sorted(gaps):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out[i] = feed[j]
    return out


def naive(text):
    """'2026-10-31T15:00:00Z' (or a Timestamp) -> naive UTC Timestamp; None stays None."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return None
    t = pd.Timestamp(text)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def midnight_uk(kickoff):
    """00:00 Europe/London on the UK date of a (naive UTC) kickoff, as naive UTC:
    the start of the match's day on the clock football-data.co.uk dates it by."""
    k = naive(kickoff)
    day = k.tz_localize("UTC").tz_convert(sd.UK).date()
    start = datetime(day.year, day.month, day.day, tzinfo=sd.UK).astimezone(timezone.utc)
    return pd.Timestamp(start.replace(tzinfo=None))


def freeze_at(kickoff, confirmed):
    return naive(kickoff) if confirmed else midnight_uk(kickoff)


def closes(kickoff, confirmed, others=(), now=None, espn_later=False):
    """When a pick on the match closes (CLOSE in the notes): freeze_at of the
    shown kickoff, or any other source's time before it. With espn_later
    (ESPN lists the match as scheduled at a valid time still to come), a
    time that has already passed is not the kickoff and is left out."""
    t = freeze_at(kickoff, confirmed)
    for other in others:
        other = naive(other)
        if other is None or (espn_later and now is not None and other <= now):
            continue
        t = min(t, other)
    return t


def sources_of(kick_fd, f):
    """Each source's kickoff where they differ ({name: naive UTC}), else None."""
    times = {FD_SOURCE: kick_fd, "ESPN": naive((f or {}).get("kickoff_espn")),
             "fixturedownload.com": naive((f or {}).get("kickoff_fd"))}
    times = {k: v for k, v in times.items() if v is not None}
    return times if len(set(times.values())) > 1 else None


def espn_scheduled_later(f, now):
    """ESPN lists the match as scheduled at a valid time still to come."""
    k = naive((f or {}).get("kickoff_espn"))
    return bool(f) and f.get("status_espn") == "scheduled" and f.get("status") == "scheduled" \
        and k is not None and k > now


def feed_matches(path=sfeed.FEED, log=print):
    """(snapshot, reconciled rows) from the feed snapshot on file. A missing
    or unreadable snapshot is no feed at all - logged, never fatal: the run
    goes on with football-data.co.uk alone."""
    try:
        snap = sfeed.load(path)
        return snap, sfeed.reconciled(snap, log=lambda *_: None)
    except Exception as err:                   # noqa: BLE001 - logged; football-data.co.uk alone
        log(f"   feeds: snapshot unreadable ({err.__class__.__name__}: {err}); football-data.co.uk alone")
        return sfeed.empty(), []


@dataclass
class Schedule:
    frame: pd.DataFrame                         # the model's input: football-data results + matches to play
    info: dict = field(default_factory=dict)    # match_id -> how the match to play was put together
    off: list = field(default_factory=list)     # postponed, cancelled, abandoned (feed rows)
    tbc: list = field(default_factory=list)     # to play, no date or time from any source
    provisional: pd.DataFrame = None            # feed results football-data.co.uk has not posted
    # The four below are filed by key(): the pairing, and where pairings
    # repeat the pairing and its day (find_key looks one up). There a
    # football-data.co.uk fixture the feeds date a day or two apart is also
    # filed under its own day, so a pick logged from it finds its match.
    disputed: set = field(default_factory=set)  # keys whose feed scores contradict
    status: dict = field(default_factory=dict)  # key -> the feeds' status
    feed_kickoff: dict = field(default_factory=dict)  # key -> ESPN's kickoff (naive UTC) where it lists it as scheduled
    blocked: list = field(default_factory=list)  # football-data fixtures the feeds say are off or under way
    disputes: dict = field(default_factory=dict)  # key -> {feed: "h-a"} where the feeds' scores contradict


def _blank(league, season, day, kickoff, home, away):
    """A row in football-data.co.uk's columns with no price and no result."""
    row = dict.fromkeys(sd.COLUMNS)
    clock = kickoff.tz_localize("UTC").tz_convert(sd.UK).strftime("%H:%M")
    row.update(match_id=sd.match_id(league, pd.Timestamp(day).date(), home, away), league=league,
               season=int(season), date=day, time=clock, kickoff=kickoff, home=home, away=away)
    return row


def schedule(matches, fixtures, feed, now, ahead=pd.Timedelta(days=10), behind=pd.Timedelta(days=4), log=print,
             leagues=None):
    """football-data.co.uk's results and fixtures, and the feed's reconciled
    matches, as one Schedule (the rules are in this module's docstring).

    leagues: the leagues the run carries (soccer_predict: in_app()) - the
    feeds' matches and the fixtures of every other league are left out
    (`matches` never: the model learns from every league's results). None:
    the feeds' matches of the five (sd.TOP) and every fixture, as before
    every league was measured."""
    now = naive(now)
    played = played_pairs(matches)
    if leagues is None:
        feed = [m for m in feed if m["league"] in sd.TOP]
    else:
        keep_leagues = set(leagues)
        left = sum(1 for m in feed if m["league"] not in keep_leagues)
        feed = [m for m in feed if m["league"] in keep_leagues]
        fixtures = fixtures[fixtures["league"].isin(list(keep_leagues))]
        if left:
            log(f"   feeds: {left} match{'es' if left != 1 else ''} of leagues the app does not carry left out")
    by_pair = {}
    for m in feed:
        k = feed_key(m)
        if k in by_pair and repeats(m["league"]):
            # never seen: two of one pairing's matches on one day
            log(f"   {m['league']} {m['home']} v {m['away']}: the feeds list two matches on {k[4]}; "
                f"the later listed ({m.get('espn_id') or m.get('fd_id')}) is used")
        by_pair[k] = m
    out = Schedule(frame=None)
    out.status = {k: m["status"] for k, m in by_pair.items()}
    out.feed_kickoff = {k: naive(m["kickoff_espn"]) for k, m in by_pair.items()
                        if m.get("kickoff_espn") and m.get("status_espn") == "scheduled"}
    out.disputed = {k for k, m in by_pair.items() if m["status"] == "disputed"}
    out.disputes = {k: {"ESPN": "-".join(map(str, m.get("score_espn") or [])) or None,
                        "fixturedownload.com": "-".join(map(str, m.get("score_fd") or [])) or None}
                    for k, m in by_pair.items() if m["status"] == "disputed"}

    # football-data.co.uk's fixtures: its row is the match.
    fx = fixtures.drop(columns=["fetched_at"], errors="ignore").copy()
    fx = fx[~fx["match_id"].isin(matches["match_id"])]
    gone = [is_played(played, *k) for k in zip(fx["league"], fx["season"], fx["home"], fx["away"], fx["date"])]
    fx = fx[~pd.Series(gone, index=fx.index, dtype=bool)]
    # Where pairings repeat, a fixture is joined to the feed match nearest
    # its date (join_repeats); elsewhere to its pairing's. fd_pairs: the feed
    # matches football-data.co.uk's fixtures stand for.
    joined = join_repeats(fx, feed)
    fd_pairs = {pairing(*k) for k in zip(fx["league"], fx["season"], fx["home"], fx["away"]) if not repeats(k[0])}
    fd_pairs |= {feed_key(f) for f in joined.values()}
    for i, f in joined.items():
        mine, theirs = key(*fx.loc[i, ["league", "season", "home", "away", "date"]]), feed_key(f)
        if mine != theirs and mine not in by_pair:          # filed under the fixture's own day as well
            for table in (out.status, out.feed_kickoff, out.disputes):
                if theirs in table:
                    table[mine] = table[theirs]
            if theirs in out.disputed:
                out.disputed.add(mine)
    lo, hi = (now - behind).strftime("%Y-%m-%d"), (now + ahead).strftime("%Y-%m-%d")
    moved_from = {}

    def off(mid, f, kickoff, day):
        if day and lo <= day <= hi:
            out.off.append({"match_id": mid, "league": f["league"], "season": int(f["season"]), "home": f["home"],
                            "away": f["away"], "status": f["status"], "kickoff": kickoff, "date": day,
                            "source": FEED_SOURCE, "key": key(f["league"], f["season"], f["home"], f["away"], day)})

    keep = []
    for i, r in fx.iterrows():
        if repeats(r["league"]):
            f = joined.get(i)
            key_ = feed_key(f) if f else None
        else:
            key_ = pairing(r["league"], r["season"], r["home"], r["away"])
            f = by_pair.get(key_)
        kick_fd = None if pd.isna(r["kickoff"]) else pd.Timestamp(r["kickoff"])
        # The feeds' time counts for the shown kickoff only where they confirm it.
        feed_conf = naive(f["kickoff_utc"]) if f and f["time_confirmed"] else None
        shown_times = [t for t in (kick_fd, feed_conf) if t is not None]
        kickoff = min(shown_times) if shown_times else None
        # ESPN's postponement is for a date: football-data.co.uk listing the
        # pairing on another date is the match rescheduled, and stands.
        if f and (f["status"] == "abandoned"
                  or (f["status"] in OFF and r["date"] == (f["uk_date"] or f["date_hint"]))):
            out.blocked.append(r["match_id"])
            off(r["match_id"], f, kick_fd, r["date"])
            log(f"   {r['league']} {r['home']} v {r['away']} ({r['date']}): the feeds say {f['status']}; not picked")
            continue
        if f and f["status"] in STARTED and kickoff is not None and kickoff > now:
            out.blocked.append(r["match_id"])
            log(f"   {r['league']} {r['home']} v {r['away']}: football-data.co.uk lists it for "
                f"{kickoff:%Y-%m-%d %H:%M} UTC, the feeds say {f['status']}; not picked")
            continue
        espn_kick = naive((f or {}).get("kickoff_espn"))
        if (kick_fd is not None and kick_fd <= now and espn_scheduled_later(f, now)
                and espn_kick > kick_fd + MOVED):
            fd_pairs.discard(key_)
            moved_from[key_] = kick_fd
            log(f"   {r['league']} {r['home']} v {r['away']}: football-data.co.uk's {kick_fd:%Y-%m-%d %H:%M} UTC has "
                f"passed with no result, ESPN lists it for {espn_kick:%Y-%m-%d %H:%M}; moved - its date stands")
            continue
        if kickoff is None or kickoff <= now:
            continue
        close = closes(kickoff, True, ((f or {}).get("kickoff_espn"), (f or {}).get("kickoff_fd")), now,
                       espn_scheduled_later(f, now))
        both = sources_of(kick_fd, f)
        if both:
            log(f"   {r['league']} {r['home']} v {r['away']}: kickoff "
                + ", ".join(f"{k} {v:%Y-%m-%d %H:%M}" for k, v in both.items())
                + f" UTC; shown {kickoff:%Y-%m-%d %H:%M}, the pick closes {close:%Y-%m-%d %H:%M}")
        fx.at[i, "kickoff"] = kickoff
        keep.append(i)
        out.info[r["match_id"]] = {
            "source": FD_SOURCE, "time_confirmed": True, "freeze_at": close,
            "kickoff_fd": kick_fd, "kickoff_feed": naive(f["kickoff_utc"]) if f else None, "kickoff_sources": both,
            "feed_confirmed": bool(f and f["time_confirmed"]), "feed_id": (f or {}).get("match_id")}
    up = fx.loc[keep]

    # The feeds' matches football-data.co.uk has not posted: no price.
    extra = []
    for key_, f in by_pair.items():
        if key_ in fd_pairs or is_played(played, f["league"], f["season"], f["home"], f["away"], feed_day(f)):
            continue
        kick = naive(f["kickoff_utc"])
        if f["status"] in OFF + ("abandoned",):
            off(f["match_id"], f, naive(f["kickoff_espn"] or f["kickoff_fd"]), f["uk_date"] or f["date_hint"])
            continue
        if f["status"] != "scheduled":
            continue
        if kick is None:
            day = f["date_hint"]
            if day and now.strftime("%Y-%m-%d") <= day <= hi:
                out.tbc.append({"league": f["league"], "season": int(f["season"]), "home": f["home"],
                                "away": f["away"], "date_hint": day, "round": f["round"], "source": FEED_SOURCE,
                                "key": key_})
            continue
        if not (now < kick <= now + ahead):
            continue
        row = _blank(f["league"], f["season"], f["uk_date"], kick, f["home"], f["away"])
        if row["match_id"] != f["match_id"]:            # never: both are match_id(league, uk_date, ...)
            raise AssertionError(f"feed id {f['match_id']} != {row['match_id']}")
        extra.append(row)
        close = closes(kick, f["time_confirmed"], (f.get("kickoff_espn"), f.get("kickoff_fd")), now,
                       espn_scheduled_later(f, now))
        out.info[row["match_id"]] = {
            "source": FEED_SOURCE, "time_confirmed": bool(f["time_confirmed"]), "freeze_at": close,
            "kickoff_fd": None, "kickoff_feed": kick, "kickoff_sources": sources_of(moved_from.get(key_), f),
            "feed_confirmed": bool(f["time_confirmed"]), "feed_id": f["match_id"],
            "moved_from": moved_from.get(key_)}

    parts = [matches, up] + ([sd.frame(extra)] if extra else [])
    out.frame = sd.order(pd.concat([p for p in parts if len(p)], ignore_index=True))
    out.provisional = provisional_results(feed, played, now)
    return out


PROVISIONAL_COLUMNS = ["match_id", "league", "season", "date", "kickoff", "home", "away", "hg", "ag", "result",
                       "provisional", "source", "seen_at"]


def provisional_results(feed, played, now=None):
    """The feeds' accepted results for matches football-data.co.uk has not
    posted (`played`: played_pairs, or a set of pairings), in its columns,
    flagged provisional. A "final" whose own kickoff (ESPN's) is still to
    come is no result. Where pairings repeat, a result of the pairing on
    another day (an earlier meeting) does not make this one posted."""
    rows = []
    for f in feed:
        if f["status"] != "final" or f.get("result") is None or not f.get("uk_date") \
                or is_played(played, f["league"], f["season"], f["home"], f["away"], f["uk_date"]):
            continue
        if now is not None and f.get("kickoff_espn") and naive(f["kickoff_espn"]) > naive(now):
            continue
        sources = f.get("result_sources") or ["espn"]
        # dated by ESPN's time (the final's own), never fixturedownload's alone
        rows.append({"match_id": f["match_id"], "league": f["league"], "season": int(f["season"]),
                     "date": f["uk_date"], "kickoff": naive(f.get("kickoff_espn")), "home": f["home"], "away": f["away"],
                     "hg": int(f["hg"]), "ag": int(f["ag"]), "result": f["result"], "provisional": True,
                     "source": " + ".join("ESPN" if s == "espn" else "fixturedownload.com" for s in sources),
                     "seen_at": f.get("result_seen_at")})
    df = pd.DataFrame(rows, columns=PROVISIONAL_COLUMNS)
    for c in ("season", "hg", "ag"):
        df[c] = pd.to_numeric(df[c]).astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    return df


def shown(frame, provisional):
    """The frame the app's tables and form lines read: football-data.co.uk's
    results and the provisional ones, flagged (column `provisional`). Never
    the model's input."""
    base = frame.copy()
    base["provisional"] = False
    if provisional is None or not len(provisional):
        return base
    keys = set(zip(provisional["league"], provisional["season"].astype(int), provisional["home"], provisional["away"]))
    days = defaultdict(list)                    # where pairings repeat: pairing -> the provisional results' dates
    for k in zip(provisional["league"], provisional["season"].astype(int), provisional["home"], provisional["away"],
                 provisional["date"]):
        if repeats(k[0]):
            days[k[:4]].append(k[4])

    def over(league, season, home, away, day):
        k = (league, season, home, away)
        if not repeats(league):
            return k in keys
        return _iso(day) is not None and any(days_apart(d, day) <= sfeed.PAIR_DAYS for d in days.get(k, ()))
    # A row still to play for a match the feeds say is over: the result
    # replaces it - where pairings repeat, only the row of that day's match.
    todo = base["result"].isna()
    clash = todo & pd.Series([over(*k) for k in zip(base["league"], base["season"].astype(int), base["home"],
                                                    base["away"], base["date"])], index=base.index, dtype=bool)
    extra = provisional.drop(columns=["source", "seen_at"])
    out = pd.concat([base[~clash], extra], ignore_index=True)
    out["provisional"] = out["provisional"].fillna(False).astype(bool)
    return sd.order(out)
