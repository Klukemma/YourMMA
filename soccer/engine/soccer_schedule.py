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
  KICKOFF. The earlier of football-data.co.uk's and the feeds' times, so a pick
      is frozen no later than any source's kickoff.
  FREEZE. At kickoff when the time is confirmed - football-data.co.uk's own
      time, or ESPN and fixturedownload agreeing (soccer_feed.reconcile). A
      time the feeds do not confirm freezes the pick at 00:00 UK time on the
      match's UK date, so no pick is ever logged after an earlier kickoff than
      the one listed.
  POSTPONED or CANCELLED (per ESPN) on football-data.co.uk's date for it: not
      picked, shown as such. A feed match ESPN says has started, finished or
      was abandoned is never picked, whatever its listed time.
  PROVISIONAL RESULTS. A feed result (soccer_feed's accepted "final": ESPN
      final and fixturedownload not contradicting it) for a match
      football-data.co.uk has not posted. It grades a pick provisionally
      (soccer_log.grade), and is listed in the results and the tables,
      flagged. It never reaches the model: the frame returned here for the
      model holds football-data.co.uk's results only.
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
    blocked: list = field(default_factory=list)  # football-data fixtures the feeds say are off or under way


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
    out.disputed = {k for k, m in by_pair.items() if m["status"] == "disputed"}

    # football-data.co.uk's fixtures: its row is the match.
    fx = fixtures.drop(columns=["fetched_at"], errors="ignore").copy()
    fx = fx[~fx["match_id"].isin(matches["match_id"])]
    keys = [pairing(*k) for k in zip(fx["league"], fx["season"], fx["home"], fx["away"])]
    fx = fx[pd.Series([k not in played for k in keys], index=fx.index, dtype=bool)]
    fd_pairs = {pairing(*k) for k in zip(fx["league"], fx["season"], fx["home"], fx["away"])}
    lo, hi = (now - behind).strftime("%Y-%m-%d"), (now + ahead).strftime("%Y-%m-%d")

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
        kick_feed = naive(f["kickoff_utc"]) if f else None
        times = [t for t in (kick_fd, kick_feed) if t is not None]
        kickoff = min(times) if times else None
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
        if kickoff is None or kickoff <= now:
            continue
        confirmed = kick_fd is not None or bool(f and f["time_confirmed"])
        if kick_fd is not None and kick_feed is not None and kick_feed != kick_fd:
            log(f"   {r['league']} {r['home']} v {r['away']}: kickoff football-data.co.uk {kick_fd:%Y-%m-%d %H:%M}, "
                f"the feeds {kick_feed:%Y-%m-%d %H:%M} UTC; the earlier is kept")
        fx.at[i, "kickoff"] = kickoff
        keep.append(i)
        out.info[r["match_id"]] = {
            "source": FD_SOURCE, "time_confirmed": confirmed, "freeze_at": freeze_at(kickoff, confirmed),
            "kickoff_fd": kick_fd, "kickoff_feed": kick_feed if f else None,
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
        out.info[row["match_id"]] = {
            "source": FEED_SOURCE, "time_confirmed": bool(f["time_confirmed"]),
            "freeze_at": freeze_at(kick, f["time_confirmed"]), "kickoff_fd": None, "kickoff_feed": kick,
            "feed_confirmed": bool(f["time_confirmed"]), "feed_id": f["match_id"]}

    parts = [matches, up] + ([sd.frame(extra)] if extra else [])
    out.frame = sd.order(pd.concat([p for p in parts if len(p)], ignore_index=True))
    out.provisional = provisional_results(feed, played)
    return out


PROVISIONAL_COLUMNS = ["match_id", "league", "season", "date", "kickoff", "home", "away", "hg", "ag", "result",
                       "provisional", "source", "seen_at"]


def provisional_results(feed, played):
    """The feeds' accepted results for matches football-data.co.uk has not
    posted, in its columns, flagged provisional."""
    rows = []
    for f in feed:
        key = pairing(f["league"], f["season"], f["home"], f["away"])
        if f["status"] != "final" or f.get("result") is None or key in played or not f.get("uk_date"):
            continue
        sources = f.get("result_sources") or ["espn"]
        rows.append({"match_id": f["match_id"], "league": f["league"], "season": int(f["season"]),
                     "date": f["uk_date"], "kickoff": naive(f["kickoff_utc"]), "home": f["home"], "away": f["away"],
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
