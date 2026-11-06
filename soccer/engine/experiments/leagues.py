"""Every league football-data.co.uk carries - measured walk-forward, and
what of each ships.

THE IDEA. The app predicted Europe's top five leagues. football-data.co.uk
carries 38 leagues with results and prices (soccer_data.ALL_LEAGUES): the
top five and their second tiers; twelve more European divisions (E2 E3 EC
SC0-SC3 N1 B1 P1 T1 G1) with pre-match AND closing prices, over/under and -
in about half their seasons - shots, since 2005-06; and sixteen "extra"
leagues (ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE SWZ USA)
since 2012 with CLOSING 1X2 prices only - no over/under, no shots, no
pre-match price in history (their live fixtures carry pre-match prices).
Each league is measured by the procedure that ships for the top five, and a
league, a number or a kind of pick reaches the app only where its own
measurement says so.

THE PROTOCOL (fixed before any number was seen; the lead's, with the
details this module had to settle written down before the first run).
  periods   the 22 European divisions DEV 2006-2015, CONFIRM 2016-2025
            (2005-06 warms up); the 16 extra leagues 2012 warm-up only, DEV
            2013-2018, CONFIRM 2019-2025 (soccer_backtest.periods_for).
            Season 2026 is live and never measured. Everything walks forward
            on DATES: ARG files one 2013/14 match on 29/01/2015, after
            season 2014's start. A league's first season on file is a
            warm-up (it has no earlier season for the base rate): only CHN,
            whose file starts in 2014, is affected.
  prices    the European divisions' pre-match average (as the top five);
            the extra leagues' closing average (all their history has),
            while their live pick will have the pre-match price. The gap
            between the two is measured on the European divisions'
            2019-2025, where both exist (log loss of the pre-match no-vig
            price against the closing one, same matches), and stated beside
            every extra league. Pinnacle's prices are never read (stale
            after 2025-07-23); the market averages are not affected.
  the model shots_ratings walk-forward extended to every league
            (shots_ratings.predict_leagues): one rating pool per country -
            England E0-EC, Scotland SC0-SC3, Spain/Italy/Germany/France top
            two, every one-league country its own; goals only where there
            are no shots (the proxy renormalises). Settled here, before any
            number: (i) rating passes by calendar - the top five's
            countries; every other August-May league; the leagues whose
            seasons are, or were through 2025, calendar years (ARG BRA CHN
            FIN IRL JPN NOR SWE USA) - because rate() turns its tier-step
            measurement over at a pass's first new season; (ii) one Poisson
            regression per league group, a dummy per league, slopes and
            stakes terms shared: the top five's exactly as shipped (trained
            on the top five only), the other 17 European divisions', the 16
            extra leagues'; (iii) the table's stakes for every league, read
            only while a club has played fewer than a double round robin
            (split seasons and play-offs read nothing at stake); (iv) a
            match filed under an earlier season than the club's current one
            (ARG's) never restarts the club as new. On the ten leagues with
            the top five predicted, every number is the shipped model's bit
            for bit. England's pool: E0-EC against E0-E1 as today, judged by
            the top five's DEV model log loss; if E0-EC is worse, E0-E1
            stays (E2, E3, EC then a pool of their own).
  the pick  the league's market price recalibrated, p ~ q^beta exp(c), with
            weights fitted only on matches dated before the league-season's
            first match (soccer_model.pick_leagues). ONE of three weight
            options is chosen on DEV - the mean DEV log loss over every DEV
            match of every league, each match once: per league (its own
            earlier matches; fewer than 1,000: the raw price), pooled per
            price type (every European division's pre-match prices; every
            extra league's closing prices), or the top five's weights for
            all (pick()'s season fits; for a league-season the latest whose
            matches all precede its first match). A match with no price:
            the model alone.
  ship (a)  a league is in the app if it has a current season in the data
            (a result in the 45 days to 2026-10-29, or a fixture posted on
            or after it) - and its pick is the recalibrated market if that
            is not worse in log loss than the raw no-vig market on its DEV,
            else the raw no-vig market. As run (soccer_leagues.season_check):
            the check on 2026-10-30, the day it ran; "posted" is a fixture on
            file dated no earlier than 45 days before (football-data's
            fixture files list only matches still to play, and the
            snapshot's were posted on Oct 27-29). It decides no league
            either way: every league but RUS has a result within 13 days,
            and RUS has neither.
       (b)  model-alone picks (no price yet) are shown for a league only if
            the model beats the league's base rate - its home / draw / away
            frequencies over its earlier seasons - on log loss in BOTH
            periods; otherwise such a match shows "no pick until priced".
       (c)  each score number - both teams to score, over 1.5, over 3.5, the
            exact score and, added here under the same rule (nothing ships
            unmeasured), over 2.5 - is shown for a league only if the source
            it would come from beats the league's base rate on log loss in
            BOTH periods. Two sources, judged apart: the market-implied
            matrix K (where the 1X2 and the over/under are priced: the
            European divisions; over 2.5 is then the over/under price
            itself) and the model's own matrix M (a match priced 1X2 only -
            every extra league's - or not priced at all).
       (d)  bands keep the edges 0.40 / 0.50 / 0.60; their hit rates, the
            pick's and the model's own picks', are measured per league
            group (top five; other European divisions; extra leagues - the
            leagues in the app) on CONFIRM and shipped with them.
  same rows every comparison of sources is on the same matches: a result,
            the league's measurement price, a model number and a base rate.
  point in time  the whole pipeline (model, pick, base rates, score
            matrices) re-run at one cut per period with every result and
            match statistic from the cut on scrambled and prices after the
            cut shuffled - nothing dated on or before the cut may move - and
            again with prices from the cut on shuffled too - nothing dated
            before it may move.

MEASURED (the run of 2026-10-30, main(); every table below is printed by it
and the results JSON is OUT; DEV-only numbers had been seen by an earlier,
interrupted pass of this same code - England's pool and the weight option
were chosen there, on DEV, and are what this run chose again).

  Runtime: the model over all 228,034 matches 77-82 s wall (about 230 s of
  CPU over 4 cores; the ten leagues alone: ~15 s). The runner's core over
  every league - model 82 s, pick 23 s, the score matrices of 213,123
  matches 23 s - about 2 minutes 10, against about 20 s for the ten today.
  On a 4-core sandbox shared with other jobs: wall times varied by up to 2x
  between runs (one model run took 184 s under load).

  England's pool (the model's log loss, same matches; E0-EC minus E0-E1):
                  DEV 2006-2015            CONFIRM 2016-2025
    top five      +0.00013 (se 0.00013)    +0.00025 (se 0.00014)
    E1            -0.00001 (se 0.00027)    -0.00031 (se 0.00031)
    E2, E3, EC    -0.00090 (se 0.00040)    -0.00179 (se 0.00039)
  The top five were worse on DEV with E0-EC: England's pool stays E0-E1
  (E2, E3, EC a pool of their own). The lower divisions themselves would
  have gained; the rule judges the top five only.

  The weight option (the pick's DEV log loss over all 103,140 DEV matches
  of the 38 leagues, each once; the raw no-vig market 1.00801):
    per league 1.00781   per price type 1.00735   the top five's 1.00746
  -> per price type (top five 0.98111 against 0.98133 with their own
  weights; other European divisions 1.01727 / 1.01742; the extra leagues
  alone 1.00302 / 1.00294 - they would have preferred the top five's, by
  0.00008, against the pooled margin of 0.00011 the choice was made on). So
  the top five's pick changes too: weights fitted on every European
  division's earlier matches, and the Bundesliga's pick the raw market
  (rule a).

  Per league: log loss of the pick as shipped / the raw no-vig market / the
  model alone / the base rate, accuracy of the pick (%); same matches per
  row; European divisions DEV 2006-2015, CONFIRM 2016-2025 on pre-match
  prices, extra leagues DEV 2013-2018, CONFIRM 2019-2025 on CLOSING prices.
  code pick   DEV: n   pick   market model  base   acc |CONF: n   pick   market model  base   acc
  E0   recal  3800 0.9676 0.9679 0.9704 1.0656 53.6 |  3800 0.9522 0.9520 0.9609 1.0663 55.6
  SP1  recal  3800 0.9514 0.9526 0.9587 1.0494 55.1 |  3800 0.9659 0.9673 0.9762 1.0676 53.7
  I1   recal  3799 0.9750 0.9770 0.9871 1.0627 53.2 |  3799 0.9466 0.9491 0.9586 1.0838 55.8
  D1   raw    3060 1.0006 1.0006 1.0058 1.0677 50.8 |  3060 0.9788 0.9788 0.9915 1.0702 52.5
  F1   recal  3800 1.0134 1.0137 1.0207 1.0701 49.9 |  3477 0.9822 0.9828 0.9943 1.0756 52.5
  E1   raw    5520 1.0485 1.0485 1.0534 1.0779 45.5 |  5520 1.0336 1.0336 1.0439 1.0759 47.5
  SP2  recal  4610 1.0372 1.0380 1.0488 1.0646 46.9 |  4599 1.0370 1.0382 1.0478 1.0678 47.0
  I2   recal  4610 1.0304 1.0314 1.0518 1.0732 46.4 |  3917 1.0452 1.0466 1.0598 1.0814 44.8
  D2   recal  3056 1.0343 1.0343 1.0461 1.0736 47.7 |  3060 1.0497 1.0492 1.0549 1.0776 46.2
  F2   recal  3790 1.0389 1.0401 1.0520 1.0621 46.1 |  3550 1.0441 1.0446 1.0552 1.0888 46.1
  E2   recal  5519 1.0394 1.0400 1.0483 1.0758 47.2 |  5367 1.0233 1.0247 1.0398 1.0746 49.2
  E3   raw    5519 1.0573 1.0573 1.0609 1.0842 45.4 |  5403 1.0491 1.0491 1.0583 1.0798 46.0
  EC   recal  5437 1.0255 1.0260 1.0335 1.0731 49.2 |  5284 1.0243 1.0248 1.0334 1.0773 48.2
  SC0  raw    2275 0.9844 0.9844 0.9940 1.0732 51.3 |  2231 0.9384 0.9384 0.9517 1.0690 55.2
  SC1  raw    1799 1.0207 1.0207 1.0276 1.0793 49.6 |  1712 1.0419 1.0419 1.0517 1.0885 45.6
  SC2  recal  1798 1.0014 1.0036 1.0163 1.0606 50.9 |  1689 1.0106 1.0113 1.0176 1.0720 50.7
  SC3  raw    1797 1.0268 1.0268 1.0299 1.0554 49.2 |  1687 1.0219 1.0219 1.0410 1.0757 48.5
  N1   recal  3060 0.9484 0.9489 0.9567 1.0516 55.8 |  2986 0.9347 0.9349 0.9440 1.0638 56.4
  B1   recal  2568 0.9791 0.9807 0.9879 1.0570 53.0 |  2805 0.9904 0.9910 1.0037 1.0732 52.0
  P1   recal  2528 0.9440 0.9478 0.9551 1.0732 54.1 |  3060 0.9165 0.9205 0.9348 1.0695 56.7
  T1   recal  3023 0.9976 0.9983 1.0095 1.0641 52.1 |  3365 0.9778 0.9807 0.9964 1.0672 53.1
  G1   recal  2502 0.9285 0.9325 0.9511 1.0471 55.4 |  2387 0.9303 0.9328 0.9530 1.0814 53.4
  ARG  recal  2435 1.0213 1.0225 1.0439 1.0732 48.5 |  3165 1.0538 1.0529 1.0684 1.0807 44.7
  AUT  raw    1097 0.9969 0.9969 1.0122 1.0627 52.4 |  1361 1.0026 1.0026 1.0202 1.0919 50.0
  BRA  recal  2279 0.9933 0.9946 1.0119 1.0361 52.1 |  2660 1.0001 1.0007 1.0191 1.0610 51.0
  CHN  recal   960 0.9676 0.9687 1.0024 1.0608 52.7 |  1602 0.8977 0.9021 0.9542 1.0685 57.9
  DNK  recal  1355 1.0052 1.0062 1.0246 1.0756 49.7 |  1399 1.0103 1.0111 1.0300 1.0780 51.4
  FIN  raw    1196 1.0156 1.0156 1.0324 1.0781 48.8 |  1150 0.9851 0.9851 1.0110 1.0787 54.0
  IRL  recal  1191 0.9329 0.9346 0.9597 1.0654 56.8 |  1189 0.9836 0.9832 1.0016 1.0774 51.6
  JPN  recal  1845 1.0286 1.0286 1.0393 1.0747 49.6 |  2372 1.0338 1.0344 1.0475 1.0828 47.2
  MEX  recal  2004 1.0365 1.0369 1.0492 1.0738 47.2 |  2317 1.0053 1.0067 1.0224 1.0683 50.4
  NOR  recal  1451 0.9982 0.9986 1.0122 1.0551 52.2 |  1689 0.9751 0.9761 0.9942 1.0604 52.6
  POL  raw    1776 1.0451 1.0451 1.0584 1.0757 46.3 |  2066 1.0343 1.0343 1.0494 1.0733 46.6
  ROU  recal  1694 1.0000 1.0008 1.0277 1.0821 50.3 |  2181 1.0009 1.0021 1.0277 1.0796 50.0
  RUS  recal  1464 0.9815 0.9820 1.0032 1.0774 53.6 |  1610 0.9848 0.9857 0.9992 1.0715 52.0
  SWE  recal  1452 0.9759 0.9770 1.0040 1.0721 53.0 |  1690 0.9838 0.9846 1.0086 1.0721 51.9
  SWZ  recal  1082 0.9957 0.9969 1.0125 1.0736 52.6 |  1412 1.0260 1.0256 1.0461 1.0771 48.9
  USA  recal  2189 0.9949 0.9968 1.0108 1.0228 52.5 |  3288 1.0230 1.0226 1.0405 1.0586 49.2
  ("raw": rule (a) - the recalibrated pick lost on DEV, by 0.00005 (SC0) to
  0.0022 (AUT); its own numbers are in OUT, "pick_recalibrated".)

  Per group (the leagues in the app, pooled; pick / market / model / base,
  accuracy of the pick):
    top five        DEV 18,259  0.98082 0.98163 0.98783 1.06290  52.6%
                    CONF 17,936 0.96426 0.96516 0.97536 1.07276  54.1%
    other European  DEV 59,411  1.01712 1.01797 1.02744 1.06943  48.9%
                    CONF 58,622 1.01076 1.01165 1.02329 1.07550  49.3%
    extra leagues   DEV 24,006  1.00418 1.00502 1.02318 1.06394  50.7%
                    CONF 29,541 1.00664 1.00714 1.02734 1.07275  49.9%
  (The top five's pick as shipped before this change: CONFIRM 0.9645,
  soccer/README.md; as decided here 0.96426.)

  Pre-match against closing (the European divisions 2019-2025, same 53,270
  matches, no-vig): pre-match 1.00256, closing 0.99905 - the pre-match price
  0.00351 worse (se 0.00031), accuracy 50.4% against 50.7% (top five 0.00243,
  other divisions 0.00385). The extra leagues were measured on the close and
  will pick on the pre-match price: expect about that much worse.

DECISIONS (soccer_leagues.SHIPPED holds them with these numbers; a test
re-applies the rules to them).
  (a) in the app: 37 leagues - every one but RUS (newest result 2026-08-02,
      89 days before the check, no fixture posted: football-data's file stops
      two rounds into 2026-27). The pick is the raw no-vig market in D1, E1,
      E3, SC0, SC1, SC3, AUT, FIN, POL; the recalibrated market elsewhere.
  (b) the model alone beat its league's base rate in both periods in all 38:
      a match with no price yet gets the model's own pick everywhere.
  (c) numbers withheld (did not beat the league's base rate in both periods):
      from the market's matrix - both teams to score in I1, SP2, I2, F2, B1,
      T1, G1; over 2.5 and 3.5 in SC2. From the model's own - both teams to
      score in SP2, I2, F2, SC1, T1, BRA, FIN, IRL, MEX, NOR; over 1.5 in F2, SWZ;
      over 2.5 in ARG, NOR; over 3.5 in ARG, DNK; and in POL all but the
      exact score. Every other number ships. (Serie A loses both teams to
      score where priced, which the app showed before: K's DEV log loss was
      0.0008 above the league rate.)
  (d) bands, CONFIRM hit rate (picks) - the pick's | the model's own:
                Strong (60%+)     Solid (50-60%)    Lean (40-50%)     Toss-up
      top five  71.2% (5,648)|71.4  54.6% (3,908)|55.3  45.1% (5,444)|45.6  37.2% (2,936)|36.8
      other Eur 70.7% (9,511)|73.7  54.8% (12,446)|56.6 44.7% (23,231)|45.1 37.0% (13,434)|37.0
      extra     68.9% (5,915)|67.1  53.9% (7,066)|53.5  44.4% (10,814)|44.4 35.9% (5,746)|37.0

POINT IN TIME. POINT_IN_TIME_RESULT

CAVEATS (never deleted; each league's own are in soccer_leagues.SHIPPED).
  - The extra leagues' history has closing prices only: their picks, their
    bands and rule (a)'s comparison were measured on the close, and they will
    pick on the pre-match price (0.0035 worse on the European divisions).
  - The extra leagues have no shots and no over/under: the model is goals
    alone there, and every goals and scores number is the model's own. The
    model is goals alone too wherever a European season has no shots: SP2,
    I2, F2, SC1-SC3, N1, B1, P1, T1 and G1 before 2017-18, D2 from 2007-08
    to 2016-17, EC since 2016-17 (all of its CONFIRM); E0-E3, SC0 and the top
    five have them throughout (D1 from 2006-07). Each league says so.
  - Several decisions sit far inside their noise: the recalibrated against
    the raw pick on DEV in D2 (1.03428 against 1.03431, se 0.00077), SC0
    (0.98441 against 0.98436, se 0.00119) and JPN (1.02857 / 1.02859, se
    0.00076); both teams to score from the model in FIN, MEX (DEV) and SWZ
    (CONFIRM) within 0.0001 of the base rate; over 3.5 in DNK (CONFIRM). The
    rules were fixed in advance and are applied as written.
  - On CONFIRM the recalibrated pick scored worse than the raw market in E0,
    D2, ARG, IRL, SWZ and USA (by 0.0002-0.0009); the rule decides on DEV
    only. Each league says so in its caveats.
  - The weight option pools every European division, so the top five's pick
    changed (weights fitted on 22 divisions, the Bundesliga's now raw). It is
    better on CONFIRM than the procedure it replaces (0.96426 against
    0.9645) - reported, never chosen on.
  - Base rates are each league's own earlier seasons (by date); CHN's first
    season (2014) only warms them up.
  - Seasons football-data labels oddly are rated as labelled: ARG's 2016
    holds two tournaments (Feb 2016 - Jun 2017), its 2013 runs to Jan 2015;
    JPN's 2026 half-season (Feb-Jun 2026) is not in the file, its 2026-27
    starts in August. The table's stakes read nothing once a club has played
    a double round robin (split seasons, play-offs).
  - Pinnacle's prices are never read (stale since 2025-07-23).
  - RUS is measured (its numbers are in SHIPPED) but not in the app: the data
    has no current season. soccer_leagues.in_app(matches, fixtures, today)
    re-applies rule (a) to the data of the day.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_backtest as bt  # noqa: E402
import soccer_data as sd  # noqa: E402
import soccer_leagues as sl  # noqa: E402
import soccer_model as smod  # noqa: E402
import soccer_scores as ssc  # noqa: E402
from experiments import shots_ratings as sr  # noqa: E402

OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/leagues.json")
TODAY = sl.CHECKED_ON
RECENT_DAYS = sl.RECENT_DAYS
ALL = list(sd.ALL_LEAGUES)
GROUPS = sl.GROUPS
GROUP_NAMES = sl.GROUP_NAMES
NUMBERS = sl.NUMBERS                       # btts, o15, o25, o35, exact
assert NUMBERS == ssc.NUMBERS and GROUPS["europe"] == sr.EUROPE_OTHER
CUTS = ("2014-04-12", "2021-10-16")        # one Saturday in each period, inside both kinds' periods
STATS = ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar")
PRICES = tuple(c for c in sd.SOURCES if c.startswith(("avg", "max", "ps", "b365")))
PIECE = ["p_h", "p_d", "p_a", "p_o25", "lam", "nu", "rho"]


group_of = sl.group_of


# ----------------------------------------------------------- (a) a current season

current_season = sl.season_check


# ----------------------------------------------------------- the pipeline

def numbers(frame, preds, pick, rows):
    """K (where shown_matrices gives the market's) and M for the given rows
    (a boolean mask): {source: {btts, o15, o25, o35, cells}} with NaN where
    the source does not apply, plus K's info. Row-local, computed in chunks."""
    idx = np.flatnonzero(rows)
    n = len(idx)
    out = {s: {k: np.full(n, np.nan) for k in ("btts", "o15", "o25", "o35")} for s in ("K", "M")}
    for s in out:
        out[s]["cells"] = np.full((n, ssc.N_CELLS), np.nan)
    src = np.full(n, None, dtype=object)
    flagged = np.zeros(n, bool)
    q_o = bt.no_vig_total(frame).to_numpy(float)
    p3 = pick[["p_h", "p_d", "p_a"]].to_numpy(float).copy()
    p3[pick["model_only"].to_numpy(bool)] = np.nan
    lam, nu, rho = (preds[c].to_numpy(float) for c in ("lam", "nu", "rho"))
    for a in range(0, n, 20000):
        j = idx[a:a + 20000]
        mats, info = ssc.shown_matrices(p3[j, 0], p3[j, 1], p3[j, 2], q_o[j], rho[j], lam[j], nu[j])
        mk = ssc.markets(np.nan_to_num(mats, nan=0.0))
        own = ssc.markets(ssc.dc_matrix(lam[j], nu[j], rho[j]))
        market = info["source"] == ssc.SOURCE_MARKET
        for k in ("btts", "o15", "o35"):
            out["K"][k][a:a + len(j)] = np.where(market, mk[k], np.nan)
            out["M"][k][a:a + len(j)] = own[k]
        out["K"]["o25"][a:a + len(j)] = np.where(market, q_o[j], np.nan)      # the card's over 2.5: the price
        out["M"]["o25"][a:a + len(j)] = preds["p_o25"].to_numpy(float)[j]
        out["K"]["cells"][a:a + len(j)] = np.where(market[:, None], mk["cells"], np.nan)
        out["M"]["cells"][a:a + len(j)] = own["cells"]
        src[a:a + len(j)] = info["source"]
        flagged[a:a + len(j)] = info["flagged"]
        # the whole shown matrix, for the point-in-time comparison
        if a == 0:
            shown = np.empty((n, mats.shape[1], mats.shape[2]))
        shown[a:a + len(j)] = mats
    return {"idx": idx, "K": out["K"], "M": out["M"], "source": src, "flagged": flagged,
            "shown": shown if n else np.empty((0, 11, 11))}


def pipeline(frame, england, option, raw=(), score_rows=None, preds=None):
    """Every number for every match of every league, as the app would build
    it: the model (predict_leagues, England's pool `england`), the pick
    (pick_leagues: weight `option`, the raw price for leagues in `raw`), the
    base rates and - for `score_rows` (a mask; default every row the model
    predicts) - the score matrices."""
    t0 = time.time()
    preds = sr.predict_leagues(frame, ALL, score_params=True, england=england) if preds is None else preds
    t_model = time.time() - t0
    pick, weights = smod.pick_leagues(frame, preds, ALL, option, raw=raw)
    base = ssc.baseline(frame, ALL)
    rows = preds["p_h"].notna().to_numpy() if score_rows is None else (score_rows & preds["p_h"].notna().to_numpy())
    nums = numbers(frame, preds, pick, rows)
    return {"preds": preds, "pick": pick, "weights": weights, "base": base, "numbers": nums,
            "runtime_model": t_model, "runtime": time.time() - t0}


# ----------------------------------------------------------- losses

def ll3(p, result):
    """Per-match three-way log loss (NaN where p or the result is missing)."""
    p = np.asarray(p, float)
    idx = pd.Series(result).map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ok = ~pd.isna(idx) & np.isfinite(p).all(axis=1)
    out = np.full(len(p), np.nan)
    i = idx[ok].astype(int)
    out[ok] = -np.log(np.clip(p[ok][np.arange(ok.sum()), i], 1e-12, 1))
    return out


def hit3(p, result):
    p = np.asarray(p, float)
    side = np.array(bt.RESULTS)[np.nan_to_num(p).argmax(axis=1)]
    return (side == np.asarray(result)).astype(float)


def bin_ll(p, y):
    return ssc._bin_loss(p, y)


def common(frame, q, preds, base, code, period):
    """The same rows for every source: league `code`, a season in `period`,
    a result, the league's measurement price, a model number and a base rate."""
    lo, hi = period
    return ((frame["league"] == code) & frame["season"].between(lo, hi) & frame["result"].isin(bt.RESULTS)
            & q.notna().all(axis=1) & preds["p_h"].notna() & base["p_h"].notna()).to_numpy()


def mean(x):
    x = np.asarray(x, float)
    return round(float(np.mean(x)), 5) if len(x) else None


# ----------------------------------------------------------- the steps

def england_choice(frame, q, preds_all, preds_two, base):
    """The top five's model log loss on the same matches, E0-EC against
    E0-E1 (the protocol: E0-EC only if not worse on DEV); England's other
    divisions beside it, for the record."""
    out = {}
    for name, codes in (("top five", sd.TOP), ("E1", ["E1"]), ("E2-EC", ["E2", "E3", "EC"])):
        out[name] = {}
        for key, per in (("dev", bt.DEV), ("confirm", bt.CONFIRM)):
            sel = np.zeros(len(frame), bool)
            for c in codes:
                sel |= common(frame, q, preds_all, base, c, per) & preds_two["p_h"].notna().to_numpy()
            res = frame["result"].to_numpy()[sel]
            a = ll3(preds_all[["p_h", "p_d", "p_a"]].to_numpy(float)[sel], res)
            b = ll3(preds_two[["p_h", "p_d", "p_a"]].to_numpy(float)[sel], res)
            d = a - b
            out[name][key] = {"matches": int(sel.sum()), "E0-EC": mean(a), "E0-E1": mean(b),
                              "diff": round(float(d.mean()), 6), "diff_se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 6)}
    choice = "all" if out["top five"]["dev"]["E0-EC"] <= out["top five"]["dev"]["E0-E1"] else "top-two"
    return choice, out


def weight_choice(frame, q, preds, base):
    """Each weight option's pick on every league's DEV rows (pooled, each
    match once), and per group; the lowest pooled DEV log loss is chosen.
    Returns (choice, table, {option: pick frame})."""
    picks, table = {}, {}
    dev = np.zeros(len(frame), bool)
    for code in ALL:
        dev |= common(frame, q, preds, base, code, bt.periods_for(code)[0])
    res = frame["result"].to_numpy()
    raw_ll = ll3(q.to_numpy(float), res)
    for opt in smod.WEIGHT_OPTIONS:
        t0 = time.time()
        picks[opt], _ = smod.pick_leagues(frame, preds, ALL, opt)
        ll = ll3(picks[opt][["p_h", "p_d", "p_a"]].to_numpy(float), res)
        row = {"dev_all": {"matches": int(dev.sum()), "log_loss": mean(ll[dev]), "raw": mean(raw_ll[dev])},
               "seconds": round(time.time() - t0, 1)}
        for g, codes in GROUPS.items():
            sel = dev & frame["league"].isin(codes).to_numpy()
            row[g] = {"matches": int(sel.sum()), "log_loss": mean(ll[sel]), "raw": mean(raw_ll[sel])}
        table[opt] = row
    choice = min(smod.WEIGHT_OPTIONS, key=lambda o: table[o]["dev_all"]["log_loss"])
    return choice, table, picks


def league_rows(frame, q, kind, preds, base, pick_opt, pick, decisions_season):
    """Per league and period: matches, log loss of the shipped pick, the
    chosen option's pick before rule (a), the raw no-vig market, the model,
    the base rate; the pick's accuracy (and the raw market's)."""
    res = frame["result"].to_numpy()
    srcs = {"pick": pick[["p_h", "p_d", "p_a"]].to_numpy(float),
            "pick_recalibrated": pick_opt[["p_h", "p_d", "p_a"]].to_numpy(float),
            "market": q.to_numpy(float), "model": preds[["p_h", "p_d", "p_a"]].to_numpy(float),
            "base": base[["p_h", "p_d", "p_a"]].to_numpy(float)}
    losses = {s: ll3(p, res) for s, p in srcs.items()}
    hits = {s: hit3(srcs[s], res) for s in ("pick", "market", "model")}
    out = {}
    for code in ALL:
        dev, conf = bt.periods_for(code)
        r = {"name": sd.ALL_LEAGUES[code].name, "country": sd.ALL_LEAGUES[code].country, "group": group_of(code),
             "price": "closing" if code in sd.EXTRA else "pre-match", "season_check": decisions_season[code]}
        for key, per in (("dev", dev), ("confirm", conf)):
            sel = common(frame, q, preds, base, code, per)
            r[key] = {"seasons": list(per), "matches": int(sel.sum())}
            if sel.any():
                r[key].update({s: mean(losses[s][sel]) for s in srcs})
                r[key].update({f"acc_{s}": round(float(hits[s][sel].mean()), 4) for s in hits})
                d = losses["model"][sel] - losses["base"][sel]
                r[key]["model_vs_base_se"] = round(float(d.std(ddof=1) / math.sqrt(len(d))), 5) if len(d) > 1 else None
                d = losses["pick_recalibrated"][sel] - losses["market"][sel]
                r[key]["recal_vs_market_se"] = round(float(d.std(ddof=1) / math.sqrt(len(d))), 5) if len(d) > 1 else None
        out[code] = r
    return out


def score_rows(frame, q, preds, base, nums):
    """Per league, period, source (K, M) and number: matches, the source's
    log loss and the base rate's on the same matches."""
    pos = {i: k for k, i in enumerate(nums["idx"])}
    take = np.array([pos.get(i, -1) for i in range(len(frame))])
    hg, ag = frame["hg"].to_numpy(float), frame["ag"].to_numpy(float)
    ys = {"btts": ((hg > 0) & (ag > 0)).astype(float), "o15": (hg + ag > 1.5).astype(float),
          "o25": (hg + ag > 2.5).astype(float), "o35": (hg + ag > 3.5).astype(float)}
    cat = ssc.score_index(np.nan_to_num(hg), np.nan_to_num(ag))
    bcells = base[[f"c{k}" for k in range(ssc.N_CELLS)]].to_numpy(float)
    out = {}
    for code in ALL:
        out[code] = {}
        for key, per in zip(("dev", "confirm"), bt.periods_for(code)):
            sel = common(frame, q, preds, base, code, per) & (take >= 0)
            out[code][key] = {}
            for s in ("K", "M"):
                k_rows = take[sel]
                has = np.isfinite(nums[s]["btts"][k_rows])
                rows = np.flatnonzero(sel)[has]
                kr = k_rows[has]
                rep = {"matches": int(len(rows))}
                if len(rows):
                    for t in ("btts", "o15", "o25", "o35"):
                        rep[t] = {"log_loss": mean(bin_ll(nums[s][t][kr], ys[t][rows])),
                                  "base": mean(bin_ll(base[t].to_numpy(float)[rows], ys[t][rows]))}
                    pe = nums[s]["cells"][kr, cat[rows]]
                    pb = bcells[rows, cat[rows]]
                    C = nums[s]["cells"][kr]
                    top1 = np.argsort(-C[:, :-1], axis=1, kind="stable")[:, 0]
                    rep["exact"] = {"log_loss": mean(-np.log(np.clip(pe, 1e-12, 1))),
                                    "base": mean(-np.log(np.clip(pb, 1e-12, 1))),
                                    "top1_hit": round(float((top1 == cat[rows]).mean()), 4)}
                    rep["btts"]["mean_p"] = round(float(np.mean(nums[s]["btts"][kr])), 4)
                    rep["btts"]["rate"] = round(float(ys["btts"][rows].mean()), 4)
                out[code][key][s] = rep
    return out


def decide(leagues, scores, season):
    """Rules (a)-(c) from the measured numbers, by soccer_leagues' rules."""
    out = {}
    for code in ALL:
        r = leagues[code]
        dev, conf = r["dev"], r["confirm"]
        shown = {}
        for s in ("K", "M"):
            if code in sd.EXTRA and s == "K":
                shown[s] = None                                    # no over/under price: no K
                continue
            shown[s] = sl.rule_numbers(scores[code]["dev"][s], scores[code]["confirm"][s])
        out[code] = {"in_app": season[code]["ok"],
                     "pick": sl.rule_pick(dev.get("pick_recalibrated"), dev.get("market")),
                     "model_only": sl.rule_model_alone(dev, conf),
                     "numbers": {"market": shown["K"], "model": shown["M"]}}
    return out


def bands(frame, q, preds, base, pick, decisions):
    """Rule (d): per league group (the leagues in the app), per period, the
    pick's band hit rates and the model's own picks' at the same edges."""
    res = frame["result"].to_numpy()
    pk = pick[["p_h", "p_d", "p_a"]].to_numpy(float)
    md = preds[["p_h", "p_d", "p_a"]].to_numpy(float)
    out = {}
    for g, codes in GROUPS.items():
        out[g] = {}
        for k, key in enumerate(("dev", "confirm")):
            sel = np.zeros(len(frame), bool)
            for c in codes:
                if decisions[c]["in_app"]:
                    sel |= common(frame, q, preds, base, c, bt.periods_for(c)[k])
            out[g][key] = {
                "pick": ssc._band_table(pk[sel].max(axis=1), hit3(pk[sel], res[sel]).astype(bool)),
                "model": ssc._band_table(md[sel].max(axis=1), hit3(md[sel], res[sel]).astype(bool))}
    return out


def price_gap(frame):
    """Pre-match against closing no-vig log loss where both exist: the
    European divisions, seasons 2019-2025, per group and per league."""
    pre, close = bt.no_vig(frame, "pre"), bt.no_vig(frame, "close")
    sel = (frame["league"].isin(sd.MAIN) & frame["season"].between(2019, 2025) & frame["result"].isin(bt.RESULTS)
           & pre.notna().all(axis=1) & close.notna().all(axis=1)).to_numpy()
    res = frame["result"].to_numpy()
    a, b = ll3(pre.to_numpy(float), res), ll3(close.to_numpy(float), res)
    out = {}
    for name, codes in [("all", sd.MAIN), ("top", sd.TOP), ("europe", sr.EUROPE_OTHER)] + [(c, [c]) for c in sd.MAIN]:
        s = sel & frame["league"].isin(codes).to_numpy()
        d = a[s] - b[s]
        out[name] = {"matches": int(s.sum()), "pre": mean(a[s]), "close": mean(b[s]),
                     "gap": round(float(d.mean()), 5), "gap_se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 5),
                     "acc_pre": round(float(hit3(pre.to_numpy(float)[s], res[s]).mean()), 4),
                     "acc_close": round(float(hit3(close.to_numpy(float)[s], res[s]).mean()), 4)}
    return out


def group_rows(frame, q, preds, base, pick, decisions):
    """Per league group (the leagues in the app), per period: the same
    columns as league_rows, pooled over the group's leagues (each match once)."""
    res = frame["result"].to_numpy()
    srcs = {"pick": pick[["p_h", "p_d", "p_a"]].to_numpy(float), "market": q.to_numpy(float),
            "model": preds[["p_h", "p_d", "p_a"]].to_numpy(float), "base": base[["p_h", "p_d", "p_a"]].to_numpy(float)}
    losses = {k: ll3(v, res) for k, v in srcs.items()}
    out = {}
    for g, codes in GROUPS.items():
        out[g] = {}
        for k, key in enumerate(("dev", "confirm")):
            sel = np.zeros(len(frame), bool)
            for c in codes:
                if decisions[c]["in_app"]:
                    sel |= common(frame, q, preds, base, c, bt.periods_for(c)[k])
            r = {"matches": int(sel.sum())}
            if sel.any():
                r.update({k2: mean(v[sel]) for k2, v in losses.items()})
                r["acc_pick"] = round(float(hit3(srcs["pick"][sel], res[sel]).mean()), 4)
                r["acc_market"] = round(float(hit3(srcs["market"][sel], res[sel]).mean()), 4)
                r["acc_model"] = round(float(hit3(srcs["model"][sel], res[sel]).mean()), 4)
            out[g][key] = r
    return out


NUMBER_NAMES = {"btts": "both teams to score", "o15": "over 1.5", "o25": "over 2.5", "o35": "over 3.5",
                "exact": "the exact score"}


def seasons_without_shots(frame):
    """{league: the seasons in its two periods where most matches have no
    shots on target} - the model rated those on goals alone."""
    out = {}
    has = frame["hst"].notna() & frame["ast"].notna()
    share = has.groupby([frame["league"], frame["season"]]).mean()
    for code in ALL:
        lo, hi = bt.periods_for(code)[0][0], bt.periods_for(code)[1][1]
        x = share.get(code)
        out[code] = [] if x is None else [int(s) for s, v in x.items() if lo <= s <= hi and v < 0.5]
    return out


def _spans(seasons):
    """[2006, 2007, 2008, 2012] -> '2006-07 to 2008-09, 2012-13'."""
    out, run = [], []
    for s in sorted(seasons):
        if run and s != run[-1] + 1:
            out.append(run)
            run = []
        run.append(s)
    if run:
        out.append(run)

    def lab(s):
        return f"{s}-{(s + 1) % 100:02d}"
    return ", ".join(lab(r[0]) if len(r) == 1 else f"{lab(r[0])} to {lab(r[-1])}" for r in out)


def caveats_for(code, league, decision, scores, gap, no_shots=()):
    """The caveats a league ships with (never deleted): what its measurement
    could not see, and every number it withholds and why."""
    out = []
    dev, conf = league["dev"], league["confirm"]
    pd_, pc = dev["seasons"], conf["seasons"]
    if code in sd.EXTRA:
        g = gap["all"]
        out.append(
            f"Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. "
            f"On the European divisions' 2019-2025, where both exist, the pre-match price scored {g['gap']:.4f} "
            f"worse in log loss than the closing one ({g['pre']:.4f} against {g['close']:.4f}, {g['matches']:,} "
            f"matches; accuracy {100 * g['acc_pre']:.1f}% against {100 * g['acc_close']:.1f}%): its live picks "
            f"should do about that much worse than measured, and its band rates were measured on closing-price picks.")
        out.append("No shots in its history: the model rates its clubs on goals alone.")
        out.append("No over/under price: every goals and scores number is the model's own.")
    elif no_shots:
        out.append(f"No shots in {_spans(no_shots)}: the model rated it on goals alone in "
                   f"{'that season' if len(no_shots) == 1 else 'those seasons'}.")
    if decision["pick"] == "raw":
        out.append(f"The recalibrated market lost to the raw market on {pd_[0]}-{pd_[1]} ({dev['pick_recalibrated']:.4f} "
                   f"against {dev['market']:.4f}), so its pick is the raw no-vig market.")
    elif conf.get("pick") is not None and conf["pick"] > conf["market"]:
        out.append(f"On {pc[0]}-{pc[1]} the recalibrated pick scored a little worse than the raw market "
                   f"({conf['pick']:.4f} against {conf['market']:.4f}).")
    if not decision["model_only"]:
        out.append("The model alone did not beat the league's base rates in both periods: no pick until a price is posted.")
    for source, label in (("market", "from the market's prices"), ("model", "from the model alone")):
        shown = decision["numbers"][source]
        if shown is None:
            continue
        held = [NUMBER_NAMES[k] for k in NUMBERS if k not in shown]
        if held:
            out.append(f"Not shown {label}: {', '.join(held)} - it did not beat the league's base rate in both "
                       f"{pd_[0]}-{pd_[1]} and {pc[0]}-{pc[1]}.")
    if code == "CHN":
        out.append("Its history starts in 2014: that season only warms the base rates up.")
    return out


def production_runtime(frame, england, option, raw, model_alone, in_app):
    """The runner's core over every league, timed as it would run: the model
    (soccer_model.predict), the pick (pick_leagues) and the score matrices
    (soccer_scores.walk) - wall and CPU seconds."""
    out = {}
    t, c = time.time(), time.process_time()
    preds = smod.predict(frame, leagues=sl.MODEL_LEAGUES, england=england)
    out["model"] = (round(time.time() - t, 1), round(time.process_time() - c, 1))
    t, c = time.time(), time.process_time()
    pick, _ = smod.pick_leagues(frame, preds, in_app, option, raw=raw, model_alone=model_alone)
    out["pick"] = (round(time.time() - t, 1), round(time.process_time() - c, 1))
    t, c = time.time(), time.process_time()
    walked = ssc.walk(frame, preds, pick, leagues=in_app)
    out["score_matrices"] = (round(time.time() - t, 1), round(time.process_time() - c, 1))
    out["matches"], out["matrices"] = int(len(frame)), int(len(walked[0]))
    return out


def shipped_block(out):
    """soccer_leagues' data block, as Python source, from a run's results."""
    def rnd(x):
        if isinstance(x, float):
            return round(x, 5)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    lines = ["# Generated by experiments/leagues.py (main) on the run of " + out["protocol"]["today"] + "; do not edit",
             "# by hand - a test re-applies the rules to the numbers stored here.",
             f"ENGLAND_POOL = {out['protocol']['england']!r}",
             f"PICK_WEIGHTS = {out['protocol']['weights']!r}",
             "SHIPPED = {"]
    for code in ALL:
        r, d = out["leagues"][code], out["decisions"][code]
        sc = out["scores"][code]

        def src(per, s):
            x = sc[per][s]
            return {k: {"log_loss": x[k]["log_loss"], "base": x[k]["base"]} for k in NUMBERS if k in x}
        meas = {"price": r["price"],
                "season_check": {k: r["season_check"][k] for k in ("last_result", "days_since", "fixtures_posted")},
                "dev": {k: r["dev"].get(k) for k in ("seasons", "matches", "pick", "pick_recalibrated", "market",
                                                     "model", "base", "acc_pick", "acc_market", "acc_model")},
                "confirm": {k: r["confirm"].get(k) for k in ("seasons", "matches", "pick", "pick_recalibrated",
                                                             "market", "model", "base", "acc_pick", "acc_market",
                                                             "acc_model")},
                "scores": {per: {"market": src(per, "K") if code not in sd.EXTRA else {}, "model": src(per, "M")}
                           for per in ("dev", "confirm")}}
        entry = {"name": r["name"], "country": r["country"], "group": r["group"], "in_app": d["in_app"],
                 "pick": d["pick"], "model_alone": d["model_only"],
                 "numbers": {"market": d["numbers"]["market"], "model": d["numbers"]["model"]},
                 "weights": out["latest_weights"][code], "caveats": r["caveats"], "measured": rnd(meas)}
        lines.append(f"    {code!r}: {{")
        for k, v in entry.items():
            if k == "measured":
                lines.append("        'measured': {")
                for k2, v2 in v.items():
                    if k2 == "scores":
                        lines.append("            'scores': {")
                        for per, srcs in v2.items():
                            lines.append(f"                {per!r}: {{")
                            for s_, nums in srcs.items():
                                lines.append(f"                    {s_!r}: {nums!r},")
                            lines.append("                },")
                        lines.append("            },")
                    else:
                        lines.append(f"            {k2!r}: {v2!r},")
                lines.append("        },")
            elif k == "caveats":
                lines.append("        'caveats': [")
                for c in v:
                    lines.append(f"            {c!r},")
                lines.append("        ],")
            else:
                lines.append(f"        {k!r}: {rnd(v)!r},")
        lines.append("    },")
    lines.append("}")
    bands = {g: {basis: {b: {"hit_rate": x["hit_rate"], "picks": x["picks"]}
                         for b, x in out["bands"][g]["confirm"][basis].items() if b != "all"}
                 for basis in ("pick", "model")} for g in GROUPS}
    lines.append("# Rule (d): each group's band hit rates on its confirmation period (the leagues in the app):")
    lines.append("# the pick's (\"pick\") and the model's own picks' at the same edges (\"model\").")
    lines.append("BANDS = {")
    for g, v in bands.items():
        lines.append(f"    {g!r}: {{")
        for basis, x in v.items():
            lines.append(f"        {basis!r}: {x!r},")
        lines.append("    },")
    lines.append("}")
    gap = {k: out["price_gap"][k] for k in ("all", "top", "europe")}
    lines.append("# Pre-match against closing no-vig log loss, the European divisions 2019-2025 (same matches).")
    lines.append(f"PRICE_GAP = {rnd(gap)!r}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------- point in time

def scramble(frame, cut, prices_from_cut=False, seed=1):
    """Every result and match statistic from `cut` on replaced by random
    ones (as soccer_backtest.check_point_in_time); every price of the
    matches AFTER the cut - or, prices_from_cut, from the cut on - shuffled
    among those matches."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    later = (out["date"] >= cut).to_numpy()
    for c in STATS:
        out.loc[later, c] = rng.integers(0, 6, int(later.sum()))
    hg, ag = out.loc[later, "hg"].astype(float), out.loc[later, "ag"].astype(float)
    out.loc[later, "result"] = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    move = (out["date"] >= cut).to_numpy() if prices_from_cut else (out["date"] > cut).to_numpy()
    perm = rng.permutation(np.flatnonzero(move))
    out.loc[move, list(PRICES)] = out.iloc[perm][list(PRICES)].to_numpy()
    return out


def _flat(run, rows):
    """Every number of a pipeline run for the given rows, one matrix."""
    nums = run["numbers"]
    pos = {i: k for k, i in enumerate(nums["idx"])}
    k = np.array([pos.get(i, -1) for i in np.flatnonzero(rows)])
    have = k >= 0
    parts = [run["preds"][PIECE].to_numpy(float)[rows], run["pick"][["p_h", "p_d", "p_a"]].to_numpy(float)[rows],
             run["base"].to_numpy(float)[rows]]
    shown = np.full((int(rows.sum()), 121), np.nan)
    shown[have] = nums["shown"][k[have]].reshape(int(have.sum()), -1)
    parts.append(shown)
    return np.column_stack(parts)


def point_in_time(frame, cut, england, option, raw, reference=None):
    """The whole pipeline re-run with the future scrambled (module notes):
    nothing on or before the cut moves with results from the cut on and
    prices after it scrambled; nothing before the cut with prices from the
    cut on scrambled as well. Score matrices are compared for the rows up to
    the cut (a matrix is row-local: soccer_scores' notes). reference: the
    model's predictions on the unscrambled frame, when already computed by
    the same call."""
    upto = (frame["date"] <= cut).to_numpy()
    before = (frame["date"] < cut).to_numpy()
    ref = pipeline(frame, england, option, raw, score_rows=upto, preds=reference)
    a = _flat(ref, upto)
    b = _flat(pipeline(scramble(frame, cut), england, option, raw, score_rows=upto), upto)
    c_run = pipeline(scramble(frame, cut, prices_from_cut=True), england, option, raw, score_rows=before)
    c = _flat(c_run, before)
    a_before = _flat(ref, before)
    on_day = (frame["date"] == cut).to_numpy()
    return {"cut": cut, "rows_on_or_before": int(upto.sum()), "rows_on_cut_day": int(on_day.sum()),
            "leagues_on_cut_day": sorted(set(frame.loc[on_day, "league"])),
            "numbers_compared": int(np.isfinite(a).sum()),
            "unmoved_on_or_before_cut": bool(np.array_equal(a, b, equal_nan=True)),
            "unmoved_before_cut_prices_from_cut_scrambled": bool(np.array_equal(a_before, c, equal_nan=True))}


# ----------------------------------------------------------- the run

def run(pit=True, log=print, frame=None, fixtures=None, cuts=CUTS, preds_all=None, preds_two=None, timing=True):
    """The whole measurement (module notes). frame / fixtures: what to
    measure on (default every league's snapshot; the tests and a DEV-only
    debugging pass give a slice). preds_all / preds_two: the model's two
    England variants on `frame`, where already computed (a re-run)."""
    t_all = time.time()
    frame = sd.load_matches(leagues="all") if frame is None else frame
    fixtures = sd.load_fixtures(leagues="all") if fixtures is None else fixtures
    log(f"{len(frame):,} matches in {frame['league'].nunique()} leagues, {len(fixtures)} fixtures on file")
    q, kind = smod.market_price(frame)

    t0 = time.time()
    if preds_all is None:
        preds_all = sr.predict_leagues(frame, ALL, score_params=True, england="all")
    t_model = time.time() - t0
    log(f"model, every league, England E0-EC: {t_model:.1f}s")
    if preds_two is None:
        preds_two = sr.predict_leagues(frame, ALL, score_params=True, england="top-two")
    log("model, England E0-E1 as before: done")
    base = ssc.baseline(frame, ALL)
    england, eng_table = england_choice(frame, q, preds_all, preds_two, base)
    preds = preds_all if england == "all" else preds_two
    log(f"England's pool: {'E0-EC' if england == 'all' else 'E0-E1'} (top five DEV model log loss "
        f"E0-EC {eng_table['top five']['dev']['E0-EC']} vs E0-E1 {eng_table['top five']['dev']['E0-E1']})")

    option, w_table, picks = weight_choice(frame, q, preds, base)
    log(f"weights: {option}")
    season = {c: current_season(frame, fixtures, c) for c in ALL}
    first = league_rows(frame, q, kind, preds, base, picks[option], picks[option], season)
    raw = [c for c in ALL if sl.rule_pick(first[c]["dev"].get("pick_recalibrated"), first[c]["dev"].get("market")) == "raw"]
    pick, weights = smod.pick_leagues(frame, preds, ALL, option, raw=raw)
    leagues = league_rows(frame, q, kind, preds, base, picks[option], pick, season)
    t0 = time.time()
    nums = numbers(frame, preds, pick, preds["p_h"].notna().to_numpy())
    t_scores = time.time() - t0
    scores = score_rows(frame, q, preds, base, nums)
    decisions = decide(leagues, scores, season)
    assert sorted(raw) == sorted(c for c in ALL if decisions[c]["pick"] == "raw")
    band = bands(frame, q, preds, base, pick, decisions)
    gap = price_gap(frame)
    no_shots = seasons_without_shots(frame)
    for c in ALL:
        leagues[c]["caveats"] = caveats_for(c, leagues[c], decisions[c], scores[c], gap, no_shots[c])
    groups = group_rows(frame, q, preds, base, pick, decisions)
    in_app = [c for c in ALL if decisions[c]["in_app"]]
    production = None
    if timing:
        production = production_runtime(frame, england, option, raw, [c for c in in_app if decisions[c]["model_only"]],
                                        in_app)
        log(f"the runner's core over every league in the app: {production}")
    latest = {}
    for c in ALL:
        ws = sorted((s, w) for (lg, s), w in weights.items() if lg == c and w is not None)
        latest[c] = {"season": ws[-1][0], "w": [round(v, 4) for v in ws[-1][1]]} if ws else None
    out = {"protocol": {"england": england, "weights": option, "cuts": list(CUTS), "today": TODAY,
                        "recent_days": RECENT_DAYS},
           "runtime": {"model_every_league_s": round(t_model, 1), "score_matrices_s": round(t_scores, 1),
                       "matches": int(len(frame)), "production": production},
           "england": eng_table, "weight_options": w_table, "leagues": leagues, "groups": groups, "scores": scores,
           "decisions": decisions, "bands": band, "price_gap": gap, "latest_weights": latest}
    if pit:
        out["point_in_time"] = []
        for cut in cuts:
            t0 = time.time()
            r = point_in_time(frame, cut, england, option, raw, reference=preds)
            r["seconds"] = round(time.time() - t0, 1)
            out["point_in_time"].append(r)
            log(f"point in time {cut}: {r}")
    out["runtime"]["experiment_s"] = round(time.time() - t_all, 1)
    return out


# ----------------------------------------------------------- printing

def _f(x, d=4):
    return "   -  " if x is None else f"{x:.{d}f}"


def tables(out):
    lines = []
    p = out["protocol"]
    lines.append(f"Runtime: the model over {out['runtime']['matches']:,} matches {out['runtime']['model_every_league_s']}s; "
                 f"score matrices {out['runtime']['score_matrices_s']}s")
    if out["runtime"].get("production"):
        lines.append(f"The runner's core, every league in the app (wall s, CPU s): {out['runtime']['production']}")
    lines.append("\nENGLAND'S POOL (model log loss, same matches; diff = E0-EC minus E0-E1)")
    for name, r in out["england"].items():
        for key in ("dev", "confirm"):
            x = r[key]
            lines.append(f"  {name:>9} {key:>7}: {x['matches']:>6}  E0-EC {_f(x['E0-EC'], 5)}  E0-E1 {_f(x['E0-E1'], 5)}"
                         f"  diff {x['diff']:+.5f} (se {x['diff_se']:.5f})")
    lines.append(f"  -> {'E0-EC' if p['england'] == 'all' else 'E0-E1'}")
    lines.append("\nWEIGHT OPTIONS (DEV log loss of the pick; raw = the no-vig price, same matches)")
    for opt, r in out["weight_options"].items():
        lines.append(f"  {opt:>10}: all {r['dev_all']['log_loss']:.5f} (raw {r['dev_all']['raw']:.5f}, "
                     f"{r['dev_all']['matches']})  " + "  ".join(
                         f"{g} {r[g]['log_loss']:.5f}" for g in GROUPS))
    lines.append(f"  -> {p['weights']}")
    lines.append("\nPER LEAGUE: log loss of the pick as shipped (recal = recalibrated before rule a) / raw market / "
                 "model / base rate, accuracy of the pick")
    lines.append("  code  in  pick  |    DEV n   pick  recal  market  model   base   acc |  CONF n   pick  recal  market"
                 "  model   base   acc | model-only")
    for code, r in out["leagues"].items():
        d = out["decisions"][code]
        row = f"  {code:<4}  {'Y' if d['in_app'] else 'n'}  {d['pick'][:5]:<5} |"
        for key in ("dev", "confirm"):
            x = r[key]
            row += (f" {x['matches']:>6} {_f(x.get('pick'))} {_f(x.get('pick_recalibrated'))} {_f(x.get('market'))} "
                    f"{_f(x.get('model'))} {_f(x.get('base'))} {_f(x.get('acc_pick'), 3)} |")
        row += f" {'yes' if d['model_only'] else 'no'}"
        lines.append(row)
    lines.append("\nPER GROUP (the leagues in the app, pooled): log loss pick / market / model / base, accuracy pick / market / model")
    for g, r in out["groups"].items():
        for key in ("dev", "confirm"):
            x = r[key]
            lines.append(f"  {GROUP_NAMES[g]:>24} {key:>7}: {x['matches']:>6}  {_f(x.get('pick'), 5)} {_f(x.get('market'), 5)} "
                         f"{_f(x.get('model'), 5)} {_f(x.get('base'), 5)}  {_f(x.get('acc_pick'), 3)} "
                         f"{_f(x.get('acc_market'), 3)} {_f(x.get('acc_model'), 3)}")
    lines.append("\nIN THE APP (rule a): " + ", ".join(c for c, d in out["decisions"].items() if d["in_app"]))
    for c, d in out["decisions"].items():
        if not d["in_app"]:
            sc = out["leagues"][c]["season_check"]
            lines.append(f"  NOT: {c} - newest result {sc['last_result']} ({sc['days_since']} days before {out['protocol']['today']}), "
                         f"{sc['fixtures_posted']} fixtures posted")
    lines.append("\nSCORE NUMBERS (rule c), log loss minus the base rate's, DEV / CONFIRM; * = shown")
    for code in out["leagues"]:
        for s, key in (("K", "market"), ("M", "model")):
            shown = out["decisions"][code]["numbers"][key]
            if shown is None:
                continue
            dv, cf = out["scores"][code]["dev"][s], out["scores"][code]["confirm"][s]
            if not dv.get("matches") or not cf.get("matches"):
                lines.append(f"  {code:<4} {s}: no matches")
                continue
            lines.append(f"  {code:<4} {s} {dv['matches']:>5}/{cf['matches']:>5}: " + "  ".join(
                f"{t}{'*' if t in shown else ' '} {dv[t]['log_loss'] - dv[t]['base']:+.4f}/{cf[t]['log_loss'] - cf[t]['base']:+.4f}"
                for t in NUMBERS))
    lines.append("\nBANDS (rule d), CONFIRM: hit rate (picks) - the pick | the model's own")
    for g, r in out["bands"].items():
        c = r["confirm"]
        lines.append(f"  {GROUP_NAMES[g]:>24}: " + "  ".join(
            f"{b} {_f(c['pick'][b]['hit_rate'], 3)} ({c['pick'][b]['picks']}) | {_f(c['model'][b]['hit_rate'], 3)}"
            for b, _ in ssc.BANDS))
    lines.append("\nPRE-MATCH vs CLOSING (European divisions 2019-2025, same matches)")
    for k in ("all", "top", "europe"):
        g = out["price_gap"][k]
        if not g["matches"]:
            continue
        lines.append(f"  {k:>6}: {g['matches']} matches  pre {g['pre']:.5f}  close {g['close']:.5f}  "
                     f"gap {g['gap']:+.5f} (se {g['gap_se']:.5f})  accuracy {g['acc_pre']:.3f} vs {g['acc_close']:.3f}")
    for r in out.get("point_in_time", []):
        lines.append(f"\nPOINT IN TIME {r['cut']}: {r}")
    return "\n".join(lines)


def main():
    out = run()
    print(tables(out))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    block = OUT.with_name("soccer_leagues_block.py")
    block.write_text(shipped_block(out))
    print(f"\nwrote {OUT} and soccer_leagues' data block {block}")


def rerender(path=OUT):
    """Re-write the caveats and soccer_leagues' data block from a saved run
    (`python experiments/leagues.py --render`): for a change of wording only
    - no number is recomputed."""
    out = json.loads(path.read_text())
    no_shots = seasons_without_shots(sd.load_matches(leagues="all"))
    for c in ALL:
        out["leagues"][c]["caveats"] = caveats_for(c, out["leagues"][c], out["decisions"][c], out["scores"][c],
                                                   out["price_gap"], no_shots[c])
    path.write_text(json.dumps(out, indent=1))
    block = path.with_name("soccer_leagues_block.py")
    block.write_text(shipped_block(out))
    print(f"rewrote {path} and {block}")


if __name__ == "__main__":
    rerender() if "--render" in sys.argv[1:] else main()
