"""A team's recent results, as the app shows them beside every match.

Shown, never a model input: the model reads every match through its own
ratings. "Form" is the last five league matches played on days BEFORE the
one asked about - in either tier, so a promoted team's run-in to promotion
shows in August - newest first.

A frame with a `provisional` column (soccer_schedule.shown: the free feeds'
results football-data.co.uk has not posted yet) marks those results: a form
entry says "provisional": true, and a table row counts them.

THE TABLE IS THE REGULAR SEASON'S. football-data.co.uk's files have no round
or stage column, and its extra leagues file their play-offs and cup rounds
under the top flight (soccer_data, PLAY-OFFS): promotion / relegation
play-offs against second-tier clubs, MLS Cup, Liga MX's liguilla, ARG's
knock-out rounds, and the championship and relegation rounds after a split
(B1, AUT, ROU, DNK, FIN, G1, SC0, SWZ; POL until 2019-20). Summed into one
table, those rows give standings no league publishes. So table() sums the
regular season only, which regular_season() tells from the results alone,
on the rows known before the day asked (a row's status never depends on a
later result):
  play-off clubs  a club with 1-4 matches while the league-season's median
                  club has 10 or more is from outside the league: its rows
                  are a play-off, and it is not listed.
  post-season     in ARG, MEX and USA (POST_SEASON), after a day on which
                  every club has played the same number of matches (5 or
                  more; one match apart allowed: a match postponed, or one
                  played early), a run of days on which at least a quarter of the
                  clubs (and two) never play, a club plays twice, and that
                  has lasted a week: MLS Cup, the liguilla, the knock-outs of
                  ARG's halves and of its Copa de la Liga (2020-2024). The
                  run ends when an idle club plays again or at a three-week
                  break (Liga MX's and ARG's second half).
  split           after the last day the league completed a round robin
                  (every pair met c times, every club played c(n-1)), once
                  every club has played twice more: the clubs fall into two
                  or more groups of three or more that never meet each other
                  and that are blocks of the table on that day (top six,
                  bottom six). Every row after that day is the split's.
What cannot be told apart, and so is summed:
  - the first rounds of a split, until every club has played twice in it
    (the groups only show then), and a split whose groups are not blocks of
    the points table (ties broken by head-to-head can put a club out of
    points order - then the split is not recognised);
  - a post-season's first week (the sign has not lasted yet), a knock-out
    phase entered before every club finished the regular phase (a
    postponed regular match played during the knock-outs), and any
    knock-out outside ARG, MEX and USA (J1's championship of 2015-2016,
    CHN's group stages of 2020-2021): elsewhere a quarter of the clubs
    idle for a week while others play twice is a cup week, so it is not
    looked for; inside them, a cup week of seven days or more in the
    regular phase would read as a post-season;
  - the zone phases of ARG's two halves (and of the Copa de la Liga) and
    MLS's unbalanced schedule are summed as the annual / overall table
    (Argentina's "tabla anual", MLS's Supporters' Shield), not per zone or
    conference; Liga MX's two halves likewise (its "tabla general");
  - a split's own standings: after it, this is the table of the regular
    season, frozen - not the official one, which goes on within each group
    and halves the points in B1, AUT and ROU.
The five top leagues have none of these: their tables are what they were,
bit for bit.
"""

from collections import Counter

import numpy as np
import pandas as pd

LAST = 5
# regular_season's thresholds (the notes): a play-off club has at most
# OUTSIDE matches beside a median of SEASONS_WORTH; a post-season leaves at
# least IDLE_SHARE of the clubs idle for POST_SEASON_DAYS or more, and ends
# at a break of BREAK_DAYS; a split's groups have SPLIT_GROUP clubs or more;
# a phase ends on a day every club has played EVEN_FROM or more.
OUTSIDE, SEASONS_WORTH, IDLE_SHARE, SPLIT_GROUP, EVEN_FROM = 4, 10, 1 / 3, 3, 5
POST_SEASON_DAYS, BREAK_DAYS = 7, 21
# The leagues whose season has knock-out rounds among its own clubs (MLS
# Cup, the liguilla, ARG's halves): only there is a post-season looked for.
# Elsewhere the same signs - a quarter of the clubs idle for a week while
# others play twice - are a cup week or a winter super cup (on the snapshot
# of 2026-11-10 they would have cut 5-16 rows from 29 seasons of the
# European divisions, E0 2018-19's 6-9 February among them).
POST_SEASON = ("ARG", "MEX", "USA")


def _flag(value):
    """A provisional flag: True only for a real true (NaN, None and NA are not)."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


def form(matches, team, before, n=LAST):
    """[{date, opponent, home, gf, ga, result}] newest first, result W/D/L
    from `team`'s side."""
    day = str(pd.Timestamp(before).date()) if not isinstance(before, str) else before[:10]
    mine = matches[((matches["home"] == team) | (matches["away"] == team))
                   & (matches["date"] < day) & matches["result"].isin(("H", "D", "A"))]
    out = []
    for _, m in mine.sort_values(["date", "kickoff"], na_position="first").tail(n).iloc[::-1].iterrows():
        home = m["home"] == team
        gf, ga = (int(m["hg"]), int(m["ag"])) if home else (int(m["ag"]), int(m["hg"]))
        item = {"date": m["date"], "opponent": m["away"] if home else m["home"], "home": bool(home),
                "gf": gf, "ga": ga, "league": m["league"], "result": "W" if gf > ga else "L" if gf < ga else "D"}
        if _flag(m.get("provisional")):
            item["provisional"] = True
        out.append(item)
    return out


def _outside(home, away):
    """The clubs from outside the league: 1 to OUTSIDE matches while the
    median club has SEASONS_WORTH or more (none otherwise)."""
    n = pd.Series(np.r_[home, away]).value_counts()
    if not len(n) or n.median() < SEASONS_WORTH:
        return set()
    return set(n.index[n <= OUTSIDE])


def _post_season(home, away, days, rows, keep):
    """Mark the post-season runs (the notes) not kept. rows: {day: positions}
    in date order; keep: the positions still regular."""
    stamp = [pd.Timestamp(d) for d in days]
    clubs = {c for d in days for i in rows[d] if keep[i] for c in (home[i], away[i])}
    n = len(clubs)
    idle_min = max(2, IDLE_SHARE * n)
    count = Counter()
    k = 0
    while k < len(days):
        for i in rows[days[k]]:
            if keep[i]:
                count[home[i]] += 1
                count[away[i]] += 1
        vals = [count[c] for c in clubs]
        low, high = (min(vals), max(vals)) if vals else (0, 0)
        if (low >= EVEN_FROM and high <= low + 1
                and min(sum(v > low for v in vals), sum(v < high for v in vals)) <= 2):
            played, twice, j = set(), Counter(), k + 1
            while j < len(days):
                on = [i for i in rows[days[j]] if keep[i]]
                today = {c for i in on for c in (home[i], away[i])}
                if len(clubs - played - today) < idle_min or (stamp[j] - stamp[j - 1]).days > BREAK_DAYS:
                    break
                played |= today
                for i in on:
                    twice[home[i]] += 1
                    twice[away[i]] += 1
                j += 1
            if (j > k + 1 and 3 * sum(v >= 2 for v in twice.values()) >= len(played)
                    and (stamp[j - 1] - stamp[k]).days >= POST_SEASON_DAYS):
                for d in days[k + 1:j]:
                    for i in rows[d]:
                        keep[i] = False
                k = j
                continue
        k += 1


def _split(home, away, hg, ag, days, rows, keep):
    """Mark a split's rows (the notes) not kept."""
    clubs = sorted({c for d in days for i in rows[d] if keep[i] for c in (home[i], away[i])})
    n = len(clubs)
    if n < 2 * SPLIT_GROUP:
        return
    at = {c: k for k, c in enumerate(clubs)}
    met = np.zeros((n, n), int)
    played = np.zeros(n, int)
    boundary = None
    for k, d in enumerate(days):
        for i in rows[d]:
            if keep[i]:
                a, b = at[home[i]], at[away[i]]
                met[a, b] += 1
                met[b, a] += 1
                played[a] += 1
                played[b] += 1
        c = int((met + np.eye(n, dtype=int) * 10 ** 6).min())
        if c >= 1 and (played == c * (n - 1)).all():
            boundary = k
    if boundary is None or boundary == len(days) - 1:
        return
    after = [i for d in days[boundary + 1:] for i in rows[d] if keep[i]]
    seen, probe = Counter(), []
    for d in days[boundary + 1:]:
        for i in rows[d]:
            if keep[i]:
                probe.append(i)
                seen[home[i]] += 1
                seen[away[i]] += 1
        if all(seen[c] >= 2 for c in clubs):
            break
    else:
        return                                   # not every club twice yet: not told
    parent = {c: c for c in clubs}

    def root(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for i in probe:
        parent[root(home[i])] = root(away[i])
    groups = {}
    for c in clubs:
        groups.setdefault(root(c), []).append(c)
    groups = list(groups.values())
    if len(groups) < 2 or min(len(g) for g in groups) < SPLIT_GROUP:
        return
    points = Counter()
    for d in days[:boundary + 1]:
        for i in rows[d]:
            if keep[i]:
                h, a = int(hg[i]), int(ag[i])
                points[home[i]] += 3 if h > a else 1 if h == a else 0
                points[away[i]] += 3 if a > h else 1 if h == a else 0
    groups.sort(key=lambda g: -max(points[c] for c in g))
    for upper, lower in zip(groups, groups[1:]):
        if min(points[c] for c in upper) < max(points[c] for c in lower):
            return                               # not blocks of the table: not a split
    for i in after:
        keep[i] = False


def regular_season(m):
    """For one league-season's results (any order), whether each row is of
    its regular season (the notes: play-off clubs, post-season runs and a
    split are not) - a boolean array in `m`'s order."""
    keep = np.ones(len(m), bool)
    if not len(m):
        return keep
    home, away = m["home"].astype(str).to_numpy(), m["away"].astype(str).to_numpy()
    out = _outside(home, away)
    if out:
        keep &= ~(np.isin(home, list(out)) | np.isin(away, list(out)))
    dates = m["date"].astype(str).to_numpy()
    rows = {}
    for i in np.argsort(dates, kind="stable"):
        rows.setdefault(dates[i], []).append(int(i))
    days = list(rows)
    if str(m["league"].iloc[0]) in POST_SEASON:
        _post_season(home, away, days, rows, keep)
    _split(home, away, m["hg"].to_numpy(float), m["ag"].to_numpy(float), days, rows, keep)
    return keep


def table(matches, league, season, before=None):
    """The league table of the regular season (regular_season: no play-off,
    post-season or split rows; a club from outside the league is not
    listed) from results on days before `before` (all of the season's
    results when None): played, won, drawn, lost, goals, points, ordered by
    points, goal difference, goals scored, then name."""
    day = None if before is None else str(pd.Timestamp(before).date())
    season_rows = matches[(matches["league"] == league) & (matches["season"] == season)]
    m = season_rows[season_rows["result"].isin(("H", "D", "A"))]
    if day is not None:
        m = m[m["date"] < day]
    m = m[regular_season(m)]
    listed = season_rows if day is None else season_rows[(season_rows["date"] < day) | season_rows["result"].isna()]
    outside = _outside(listed["home"].astype(str).to_numpy(), listed["away"].astype(str).to_numpy())
    teams = sorted((set(season_rows["home"]) | set(season_rows["away"])) - outside)
    rows = {t: {"team": t, "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "points": 0,
                "provisional": 0} for t in teams}
    flags = m["provisional"] if "provisional" in m else [False] * len(m)
    for h, a, hg, ag, prov in zip(m["home"], m["away"], m["hg"], m["ag"], flags):
        hg, ag = int(hg), int(ag)
        for team, f, g in ((h, hg, ag), (a, ag, hg)):
            r = rows[team]
            r["played"] += 1
            r["provisional"] += int(_flag(prov))
            r["gf"] += f
            r["ga"] += g
            if f > g:
                r["won"] += 1
                r["points"] += 3
            elif f == g:
                r["drawn"] += 1
                r["points"] += 1
            else:
                r["lost"] += 1
    out = sorted(rows.values(), key=lambda r: (-r["points"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]))
    for i, r in enumerate(out, 1):
        r["position"] = i
        r["gd"] = r["gf"] - r["ga"]
    return out
