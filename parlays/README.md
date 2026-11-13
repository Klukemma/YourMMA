# YourMulti: same-day parlays

https://klukemma.github.io/YourMMA/parlays/app/ - the fourth page beside the UFC, NFL and soccer desks.

One ticket a day, every leg played that day (the US Eastern calendar day of each start):
among the day's 12 likeliest picks from the three apps - each at least a 50% chance and
priced - the combination of 1 to 6 legs paying 3-5x with the highest chance that every leg
lands. The rule is in `engine/sameday.py`'s docstring and was fixed before it was measured.

**Measured** (`engine/experiments/sameday_backtest.py` -> `data/sameday_backtest.json`): every
day from 2016-08-01 to 2026-08-08, point in time, at the prices on file, beside a market-only
control. 1,245 tickets, 420 won (33.7% against the picks' 31.8%), +4.5% a ticket
(-3.7% to +12.6%); last 3 years +5.1% (-9.9% to +19.8%); the control -0.8%. The typical ticket
pays about 3x with two or three legs. No established edge: the market is hard to beat.

**The underdog ticket** (v1, fixed in `engine/sameday.py` before any of its results): a second
ticket a day, for variety and a bigger payout - never the better bet. Each event's market
underdog (the side the no-vig price rates under 50%; in soccer the less likely team, never the
draw) priced 2.00-4.00 is a candidate; a seeded lottery draws one, in proportion to its chance
(close to an even draw), and the likeliest picks from the day's other events fill it to 3-5x -
almost always one favourite, so a ticket is a dog plus one leg at about 3.8x.

**Measured** the same way on the same days (`data/sameday_backtest.json`, under `dog`): at the
live draw (the lottery with its seed, the salt `sameday-dog-v1`) 1,487 tickets, 397 won (26.7%
against the prices' 24.4%), -1.6% a ticket (-9.9% to +7.0%); over the 1,000 simulation draws
-9.9% (-14.7% to -4.7%); by the prices -8.5%. The live draw drew well: it ranks above 992 of the
1,000, and one draw's luck is not the rule's. Beside the likeliest ticket, on the 1,132 days both
exist, it won 6.1 points less often; holding both returned +4.0% a unit at the live draw, -1.9%
over the draws (-6.7% to +3.2%) and -10.2% by the prices. No edge: a dog leg carries the heavier
margin, and the market is hard to beat.

Both records leave out the UFC fights that ended without a winner (92 of 5,050 in the window):
the UFC history keeps decided fights only, so they are never legs there, while live such a leg
voids. The json's caveats count them, with every other caveat; the figures in this README are
that file's, and `engine/tests/test_dog_measure.py` holds them to it.

**Live** (`engine/build_sameday.py`, run by `.github/workflows/parlays.yml` after any sport's
runner and on a schedule): reads `app/data/card.json`, `nfl/app/data/week.json` and
`soccer/app/data/matches.json`; never uses a leg whose start has passed; rebuilds a day's
ticket at the latest prices until its first leg starts, then locks it; logs every ticket to
`data/history.json` and grades it from each sport's own results.

Rebuild the history and the backtest (offline, about three minutes):

    python parlays/engine/legs_history.py
    python parlays/engine/experiments/sameday_backtest.py

`legs_history.py` builds each sport in its own process (the UFC and soccer engines both have a
package named `experiments`) and writes the pick side (`data/legs_*.csv.gz`) and every event's
market underdog (`data/legs_dogs_*.csv.gz`). The soccer legs come from this checkout's
`soccer/engine` - the shipped pick the live legs take; `--soccer-engine DIR` builds them from
another one (main's, to compare). The backtest checks that its 1,000-salt draw reproduces
`sameday.build_dog` at the live salt, and stops if it does not.

For entertainment and analysis - not betting advice.
