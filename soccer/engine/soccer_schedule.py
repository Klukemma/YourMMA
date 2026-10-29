"""football-data.co.uk's fixtures and results joined with the free feeds'
(soccer_feed.py): which matches are still to play, when each pick freezes,
which are postponed, and which results are provisional.

football-data.co.uk is the source of record. It posts the weekend's fixtures,
with the prices the pick is built from, on Friday afternoon and the midweek's
on Tuesday afternoon, and its results days late. The feeds (ESPN's public
scoreboard and fixturedownload.com) list the next rounds and the scores of the
round just played long before that; they carry no price the engine reads.
So, one rule each:

  A MATCH IS ONE ROW. A pairing (league, season, home, away) is played once a
      league season. Where football-data.co.uk has the fixture, its row is the
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
      measured stale (soccer_feed).
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

from dataclasses import dataclass, field
from datetime import datetime, timezone

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
    disputed: set = field(default_factory=set)  # pairings whose feed scores contradict
    status: dict = field(default_factory=dict)  # pairing -> the feeds' status
    feed_kickoff: dict = field(default_factory=dict)  # pairing -> ESPN's kickoff (naive UTC) where it lists it as scheduled
    blocked: list = field(default_factory=list)  # football-data fixtures the feeds say are off or under way
    disputes: dict = field(default_factory=dict)  # pairing -> {feed: "h-a"} where the feeds' scores contradict


def _blank(league, season, day, kickoff, home, away):
    """A row in football-data.co.uk's columns with no price and no result."""
    row = dict.fromkeys(sd.COLUMNS)
    clock = kickoff.tz_localize("UTC").tz_convert(sd.UK).strftime("%H:%M")
    row.update(match_id=sd.match_id(league, pd.Timestamp(day).date(), home, away), league=league,
               season=int(season), date=day, time=clock, kickoff=kickoff, home=home, away=away)
    return row


def schedule(matches, fixtures, feed, now, ahead=pd.Timedelta(days=10), behind=pd.Timedelta(days=4), log=print):
    """football-data.co.uk's results and fixtures, and the feed's reconciled
    matches, as one Schedule (the rules are in this module's docstring)."""
    now = naive(now)
    done = matches[matches["result"].notna()]
    played = set(zip(done["league"], done["season"].astype(int), done["home"], done["away"]))
    feed = [m for m in feed if m["league"] in sd.TOP]
    by_pair = {pairing(m["league"], m["season"], m["home"], m["away"]): m for m in feed}
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
    keys = [pairing(*k) for k in zip(fx["league"], fx["season"], fx["home"], fx["away"])]
    fx = fx[pd.Series([k not in played for k in keys], index=fx.index, dtype=bool)]
    fd_pairs = {pairing(*k) for k in zip(fx["league"], fx["season"], fx["home"], fx["away"])}
    lo, hi = (now - behind).strftime("%Y-%m-%d"), (now + ahead).strftime("%Y-%m-%d")
    moved_from = {}

    def off(mid, f, kickoff, day):
        if day and lo <= day <= hi:
            out.off.append({"match_id": mid, "league": f["league"], "season": int(f["season"]), "home": f["home"],
                            "away": f["away"], "status": f["status"], "kickoff": kickoff, "date": day,
                            "source": FEED_SOURCE})

    keep = []
    for i, r in fx.iterrows():
        key = pairing(r["league"], r["season"], r["home"], r["away"])
        f = by_pair.get(key)
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
            fd_pairs.discard(key)
            moved_from[key] = kick_fd
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
    for key, f in by_pair.items():
        if key in fd_pairs or key in played:
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
                                "away": f["away"], "date_hint": day, "round": f["round"], "source": FEED_SOURCE})
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
            "kickoff_fd": None, "kickoff_feed": kick, "kickoff_sources": sources_of(moved_from.get(key), f),
            "feed_confirmed": bool(f["time_confirmed"]), "feed_id": f["match_id"],
            "moved_from": moved_from.get(key)}

    parts = [matches, up] + ([sd.frame(extra)] if extra else [])
    out.frame = sd.order(pd.concat([p for p in parts if len(p)], ignore_index=True))
    out.provisional = provisional_results(feed, played, now)
    return out


PROVISIONAL_COLUMNS = ["match_id", "league", "season", "date", "kickoff", "home", "away", "hg", "ag", "result",
                       "provisional", "source", "seen_at"]


def provisional_results(feed, played, now=None):
    """The feeds' accepted results for matches football-data.co.uk has not
    posted, in its columns, flagged provisional. A "final" whose own kickoff
    (ESPN's) is still to come is no result."""
    rows = []
    for f in feed:
        key = pairing(f["league"], f["season"], f["home"], f["away"])
        if f["status"] != "final" or f.get("result") is None or key in played or not f.get("uk_date"):
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
    # A row still to play for a match the feeds say is over: the result replaces it.
    todo = base["result"].isna()
    clash = todo & pd.Series([k in keys for k in zip(base["league"], base["season"].astype(int), base["home"],
                                                      base["away"])], index=base.index, dtype=bool)
    extra = provisional.drop(columns=["source", "seen_at"])
    out = pd.concat([base[~clash], extra], ignore_index=True)
    out["provisional"] = out["provisional"].fillna(False).astype(bool)
    return sd.order(out)
