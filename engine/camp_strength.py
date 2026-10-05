"""Camp power, measured as of each fight from the camp's own results.

A reputation is the wrong input. "American Top Team is elite" is a judgement
made today, with the benefit of every title its fighters have won since - so
applied to a 2014 fight it smuggles in results from 2015 to 2026. What can
be used is what the camp had DONE by the morning of the fight:

  members at date D   fighters whose infobox lists the gym with years that
                      cover D (fighter_profile: undated listings are what
                      the page says today and are never used for the past)
  strength at date D  from those members' UFC bouts BEFORE D, within
                      WINDOW_YEARS: bouts, wins (win rate shrunk toward .5),
                      and title-fight wins
  champion coach      a trainer who is himself in the archive with a UFC
                      title-fight win before D

    camp = CampIndex(profiles, archive)
    camp.at("Carl Brown", "2019-06-01")  ->  {"gyms": [...], "bouts": ...}
"""

from collections import defaultdict

import pandas as pd

from name_resolution import norm_name

WINDOW_YEARS = 3
PRIOR = 10          # bouts of shrinkage toward .500 for small camps


class CampIndex:
    def __init__(self, profiles, archive):
        """profiles: iterable of fighter_profile dicts with a "fighter" key.
        archive: frame with date, r_name, b_name, winner, title_fight."""
        self.spans = defaultdict(list)          # fighter -> [(gym, start, end)]
        self.members = defaultdict(list)        # gym -> [(fighter, start, end)]
        self.trainers = defaultdict(list)
        for p in profiles:
            who = norm_name(p["fighter"])
            for a in p.get("affiliations", []):
                if not a.get("dated"):
                    continue
                gym = norm_name(a["canonical"])
                span = (a["start"], a["end"] if a["end"] is not None else 9999)
                self.spans[who].append((gym, *span))
                self.members[gym].append((who, *span))
            self.trainers[who] = [norm_name(t["canonical"])
                                  for t in p.get("trainers", [])]
        frame = archive.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        self.results = defaultdict(list)        # fighter -> [(date, won, title)]
        self.title_wins = defaultdict(list)     # fighter -> [date]
        for row in frame.itertuples():
            title = bool(getattr(row, "title_fight", False)) and \
                str(getattr(row, "title_fight", "")).lower() not in ("0", "false")
            for me in (row.r_name, row.b_name):
                won = row.winner == me
                self.results[norm_name(me)].append((row.date, won, title))
                if won and title:
                    self.title_wins[norm_name(me)].append(row.date)

    def gyms(self, fighter, date):
        year = pd.Timestamp(date).year
        return sorted({g for g, s, e in self.spans.get(norm_name(fighter), [])
                       if s <= year <= e})

    def _strength(self, gym, date):
        date = pd.Timestamp(date)
        since = date - pd.DateOffset(years=WINDOW_YEARS)
        bouts = wins = titles = 0
        for member, s, e in self.members.get(gym, []):
            for when, won, title in self.results.get(member, []):
                if not (since <= when < date):       # strictly before D
                    continue
                if not (s <= when.year <= e):        # while a member
                    continue
                bouts += 1
                wins += won
                titles += won and title
        rate = (wins + PRIOR * 0.5) / (bouts + PRIOR)
        return {"bouts": bouts, "wins": wins, "rate": rate, "titles": titles}

    def at(self, fighter, date):
        """The fighter's camp as of the morning of `date`."""
        date = pd.Timestamp(date)
        gyms = self.gyms(fighter, date)
        best = {"bouts": 0, "wins": 0, "rate": 0.5, "titles": 0}
        for gym in gyms:
            s = self._strength(gym, date)
            if (s["titles"], s["rate"]) > (best["titles"], best["rate"]):
                best = s
        coach = any(d < date for t in self.trainers.get(norm_name(fighter), [])
                    for d in self.title_wins.get(t, []))
        return {"gyms": gyms, **best, "champion_coach": coach}
