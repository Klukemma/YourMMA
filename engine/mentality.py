"""Fighter mentality, read only from what a fighter had DONE before a fight.

Two sides, kept apart, and a third kept apart from both:

  IN THE FIGHT - what happens when it goes wrong.
      dropped          prior fights in which they were knocked down
      recovered        of those, fights they won or at least survived to
                       the final bell
      comebacks        of those, fights they won
      quit_losses      losses by tapping to strikes, retiring between
                       rounds, corner stoppage or verbal submission - the
                       plainest "gave up" signals a result records
  OUT OF THE FIGHT - hunger.
      rising           never held a UFC title, won their last 3+, rating up
                       over those fights, and active (last fight within 8
                       months): climbing, and still has it all to get
      champion_stage   number of UFC title fights won - the further along,
                       the more there is to be comfortable with
      pace_trend       strike and takedown attempts per minute in their last
                       two fights of 5+ minutes, against their own earlier
                       average: a drive that is fading, or not, measured in
                       the cage instead of in interviews
      layoff_days      time since their last fight
  CONFIDENCE - not the same thing as hunger, and never scored as it.
      dominant_streak  current win streak, and how many of it were finishes

WHY NOTHING HERE COMES FROM INTERVIEWS OR REPUTATION. "He wasn't hungry" is
said about fighters who have just lost, and "still hungry" about ones who
have just won - hindsight wearing the costume of insight, the same trap as
an injury revealed after a loss. Every number here is computed from fights
strictly before the one being predicted.

Coverage: knockdowns and pace come from the UFC archive (UFC fights only);
quit-type losses from event and results pages of every promotion, which
list every bout on a card whoever later became famous.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from name_resolution import canonical_winners, norm_name

# Tapping to STRIKES, retiring between rounds, the corner stopping it. The
# first version had no boundary after "knee", so "Submission (kneebar)"
# counted as quitting (68 of 269 rows), and any "verbal submission" did too,
# including a verbal tap to an armbar. Those are losses to a hold, not a
# fighter giving up.
QUIT = re.compile(
    r"submission \((?:punch|strike|elbow|knee|kick)(?:e?s)?(?: and \w+)?\)|"
    r"verbal submission \((?:punch|strike|elbow|knee|kick)(?:e?s)?\)|"
    r"\bretirement\b|corner stoppage", re.I)
# "Decision - Unanimous" and, in the newest archive rows, "U-DEC"/"S-DEC".
DECISION = re.compile(r"decision|\bdec\b|-dec", re.I)
RISING_STREAK = 3
ACTIVE_DAYS = 240
PRIOR = 2          # shrinkage for the recovery rate


class MentalityIndex:
    def __init__(self, archive, world=None):
        a = canonical_winners(archive)
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        a = a.dropna(subset=["date"]).sort_values("date", kind="stable")
        self.fights = defaultdict(list)
        for row in a.itertuples():
            # match_time_sec is the time elapsed IN THE FINAL ROUND, not
            # the fight: every decision stores 300. Total time is the full
            # rounds before it plus that. Read as a total, it kept almost
            # only decisions and made a 25-minute fight look like 5.
            rnd = getattr(row, "finish_round", np.nan)
            if pd.notna(row.match_time_sec) and pd.notna(rnd):
                minutes = ((int(rnd) - 1) * 300 + float(row.match_time_sec)) / 60.0
            else:
                minutes = np.nan
            title = str(getattr(row, "title_fight", 0)) in ("1", "True", "true")
            for me, kd_them, att, tds, mu in (
                    (row.r_name, row.b_kd, row.r_sig_str_atmpted,
                     getattr(row, "r_td_atmpted", 0), row.r_mu_pre),
                    (row.b_name, row.r_kd, row.b_sig_str_atmpted,
                     getattr(row, "b_td_atmpted", 0), row.b_mu_pre)):
                won = row.winner == me
                decided = row.winner in (row.r_name, row.b_name)
                finished = decided and not won and \
                    not DECISION.search(str(row.method))
                self.fights[norm_name(me)].append({
                    "date": row.date, "won": won, "lost": decided and not won,
                    "dropped": (kd_them or 0) > 0, "finished": finished,
                    "title": title,
                    # Output = strike AND takedown attempts per minute, in
                    # fights of 5+ minutes only. Strikes alone made a
                    # wrestler who takes his man down look like he had
                    # stopped trying (Makhachev read 0.24), and a 40-second
                    # finish says nothing about pace.
                    "pace": ((att + (tds if pd.notna(tds) else 0)) / minutes)
                    if pd.notna(att) and minutes == minutes and minutes >= 5
                    else np.nan,
                    "mu": mu,
                    "finish_win": won and not DECISION.search(str(row.method))})
        self.quits = defaultdict(list)
        if world is not None:
            w = world[~world["source"].astype(str).str.startswith("record:")]
            w = w[w["result"] == "win"]
            for row in w.itertuples():
                if QUIT.search(str(row.method)):
                    self.quits[norm_name(row.loser)].append(
                        pd.Timestamp(row.date))

    def at(self, fighter, date):
        """The fighter's mentality signals as of the morning of `date`."""
        date = pd.Timestamp(date)
        past = [f for f in self.fights.get(norm_name(fighter), [])
                if f["date"] < date]                        # strictly before
        dropped = [f for f in past if f["dropped"]]
        recovered = sum(1 for f in dropped if not f["finished"])
        streak, finishes = 0, 0
        for f in reversed(past):
            if not f["won"]:
                break
            streak += 1
            finishes += f["finish_win"]
        titles = sum(1 for f in past if f["title"] and f["won"])
        last = past[-1]["date"] if past else None
        layoff = (date - last).days if last is not None else None
        mus = [f["mu"] for f in past[-RISING_STREAK - 1:] if pd.notna(f["mu"])]
        rating_up = len(mus) >= 2 and mus[-1] > mus[0]
        paces = [f["pace"] for f in past if f["pace"] == f["pace"]]
        pace_trend = (np.mean(paces[-2:]) / np.mean(paces[:-2])
                      if len(paces) >= 4 and np.mean(paces[:-2]) > 0 else np.nan)
        return {
            "fights": len(past),
            "dropped": len(dropped), "recovered": recovered,
            "comebacks": sum(1 for f in dropped if f["won"]),
            "recovery_rate": (recovered + PRIOR * 0.5) / (len(dropped) + PRIOR),
            "quit_losses": sum(1 for d in self.quits.get(norm_name(fighter), [])
                               if d < date),
            "rising": titles == 0 and streak >= RISING_STREAK and rating_up
                      and layoff is not None and layoff <= ACTIVE_DAYS,
            "champion_stage": titles,
            "pace_trend": float(pace_trend) if pace_trend == pace_trend else None,
            "layoff_days": layoff,
            "dominant_streak": streak, "streak_finishes": finishes,
        }
