"""The simulation layer: run a matchup thousands of times and report how it ends.

The statistical model answers one question - who wins - as a single number. A
simulation answers a different set: how often each fighter wins, by what route,
in which round, and how long the fight lasts. Those are the questions a model
trained on a binary outcome cannot answer at all, and they are what this module
exists to add.

WHAT IT IS WORTH, MEASURED RATHER THAN ASSUMED. As a predictor of the WINNER
the simulator is weaker than the model it sits beside: standalone AUC 0.559
over 2,052 fights from 2019 with both corners carrying three or more prior
bouts, against the model's 0.66. By year: 0.542, 0.530, 0.558, 0.578, 0.593,
0.527, 0.564, 0.607. Real, stable, and clearly worse.

So the model's probability stays the headline and this is reported beside it,
never instead of it. Where the simulation is the only source is the method and
round distribution, which nothing else here produces from fighter rates.

WHERE IT IS GOOD IS THE PART IT IS THE ONLY SOURCE FOR. Over the same 2,052
fights, aggregated method and duration against what actually happened:

                  simulated    actual
    KO/TKO           32.2%     30.7%
    submission       19.8%     16.3%
    decision         45.1%     51.8%
    mean duration   11.72 m   11.93 m

Duration lands within 2%. The bias that is there runs one way and is worth
stating: it finishes 55% of fights where the sport finishes 47%, over-calling
submissions by about three points and decisions short by seven. Read a
"Decision" call as firmer than the number says and a submission call as softer.

It is also the honest reading of a disagreement: when the model likes a fighter
the simulation does not, the two are using different information - ratings,
experience and momentum on one side, pure rate statistics on the other - and
that is worth seeing rather than averaging away.
"""

import numpy as np

import matchup_inputs as mi
import simulate as sm

# A fighter with nothing measured would otherwise be simulated entirely from
# league defaults, producing a confident-looking 50/50 between two average
# fighters wearing the names asked about. resolve_rates does report every
# imputed field, but a caller reading only the probability would never see it,
# and "no data" is the answer the whole pipeline is supposed to give.
#
# More than half the rates assumed means the distribution describes the league
# rather than the fighter, so it is refused instead of reported.
MAX_IMPUTED_SHARE = 0.5
MIN_PRIOR_BOUTS = 1

# 20,000 puts the standard error on a win probability near 0.35 of a percentage
# point, which is finer than the number is displayed to and costs about a tenth
# of a second a fight. The million-simulation runs in simulate.py's docstring
# are for calibration work, not for a card.
CARD_SIMULATIONS = 20000


def _prefixed(stats, corner):
    """A fighter snapshot re-keyed the way career_stats emits, for one corner."""
    row = {f"{corner}_{key}": value for key, value in stats.items()}
    row[f"{corner}_name"] = stats.get("name", f"{corner}-corner")
    return row


def simulate_matchup(red_stats, blue_stats, *, rounds=3, n_sims=CARD_SIMULATIONS,
                     seed=0, defaults=None):
    """Simulate one matchup from two fighter snapshots.

    Returns None when the fighters cannot be simulated at all, which a caller
    must render as "no simulation" rather than as an even fight. Raising would
    lose a whole card to one unmeasured debutant; a silent 50/50 would state a
    conclusion nobody computed.
    """
    row = {}
    row.update(_prefixed(red_stats, "r"))
    row.update(_prefixed(blue_stats, "b"))
    row = mi.shrunk_career_row(mi.shrunk_career_row(row, "r"), "b")

    for stats in (red_stats, blue_stats):
        bouts = stats.get("cd_bouts")
        try:
            bouts = float(bouts)
        except (TypeError, ValueError):
            return None
        if not bouts >= MIN_PRIOR_BOUTS:
            return None

    if defaults is None:
        defaults = sm.league_rates()
    try:
        red = sm.rates_from_career_stats(row, "r")
        blue = sm.rates_from_career_stats(row, "b")
        distribution = sm.simulate_fight(red, blue, rounds=rounds,
                                         n_sims=n_sims, defaults=defaults,
                                         rng=np.random.default_rng(seed))
    except (ValueError, KeyError, TypeError):
        return None

    total = len(sm._REQUIRED_RATE_FIELDS)
    for imputed in (distribution.imputed_a, distribution.imputed_b):
        if len(imputed) / total > MAX_IMPUTED_SHARE:
            return None
    return distribution


def summarise(distribution, red_name, blue_name):
    """The simulation as plain values, ready for a table or a phone screen."""
    if distribution is None:
        return None
    d = distribution
    by_method = {
        "KO/TKO": d.ko_prob_a + d.ko_prob_b,
        "Submission": d.sub_prob_a + d.sub_prob_b,
        "Decision": d.dec_prob_a + d.dec_prob_b,
    }
    method = max(by_method, key=by_method.get)
    # finish_round_prob has length rounds+1 and INDEX 0 MEANS "went to
    # decision", not "round 1". Reading it as round 1 printed a 70% opening
    # round on a fight the same line called a 66% decision.
    decision_share = float(d.finish_round_prob[0])
    rounds = [float(p) for p in d.finish_round_prob[1:]]
    return {
        "red": red_name,
        "blue": blue_name,
        "red_win": d.win_prob_a,
        "blue_win": d.win_prob_b,
        "draw": d.draw_prob,
        "win_se": d.win_prob_se_a,
        "method": method,
        "method_prob": by_method[method],
        "ko": by_method["KO/TKO"],
        "sub": by_method["Submission"],
        "decision": by_method["Decision"],
        # Indexed from round 1, so finish_by_round[0] is round 1. These sum to
        # the probability the fight is FINISHED, not to 1.
        "finish_by_round": rounds,
        "decision_share": decision_share,
        "mean_seconds": d.mean_duration_sec,
        "median_seconds": d.median_duration_sec,
        # Which of each fighter's rates had to be assumed rather than measured.
        # A confident-looking distribution built on league averages is the one
        # thing a reader must be able to see.
        "imputed_red": sorted(d.imputed_a),
        "imputed_blue": sorted(d.imputed_b),
        "simulations": d.n_sims,
    }


def format_line(summary):
    """One line per fight, for the card printout."""
    if summary is None:
        return "    no simulation - not enough measured history"
    winner = (summary["red"] if summary["red_win"] >= summary["blue_win"]
              else summary["blue"])
    share = max(summary["red_win"], summary["blue_win"])
    minutes, seconds = divmod(int(summary["mean_seconds"]), 60)
    return (f"    {winner} {share*100:.1f}% "
            f"(+-{summary['win_se']*100:.1f})  "
            f"KO {summary['ko']*100:.0f}%  "
            f"SUB {summary['sub']*100:.0f}%  "
            f"DEC {summary['decision']*100:.0f}%  "
            f"avg {minutes}:{seconds:02d}")
