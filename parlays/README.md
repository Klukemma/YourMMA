# YourMulti: same-day parlays

https://klukemma.github.io/YourMMA/parlays/app/ - the fourth page beside the UFC, NFL and soccer desks.

One ticket a day, every leg played that day (the US Eastern calendar day of each start):
among the day's 12 likeliest picks from the three apps - each at least a 50% chance and
priced - the combination of 1 to 6 legs paying 3-5x with the highest chance that every leg
lands. The rule is in `engine/sameday.py`'s docstring and was fixed before it was measured.

**Measured** (`engine/experiments/sameday_backtest.py` -> `data/sameday_backtest.json`): every
day from 2016-08-01 to 2026-08-08, point in time, at the prices on file, beside a market-only
control. 1,245 tickets, 429 won (34.5% against the picks' 31.7%), +6.5% a ticket
(-1.5% to +14.4%); last 3 years +3.8% (-11% to +18%); the control -0.8%. The typical ticket
pays about 3x with two or three legs. No established edge: the market is hard to beat.

**Live** (`engine/build_sameday.py`, run by `.github/workflows/parlays.yml` after any sport's
runner and on a schedule): reads `app/data/card.json`, `nfl/app/data/week.json` and
`soccer/app/data/matches.json`; never uses a leg whose start has passed; rebuilds a day's
ticket at the latest prices until its first leg starts, then locks it; logs every ticket to
`data/history.json` and grades it from each sport's own results.

Rebuild the history and the backtest (offline, about a minute):

    git archive origin/main soccer | tar -x -C /tmp/main_soccer
    python parlays/engine/legs_history.py --soccer-engine /tmp/main_soccer/soccer/engine
    python parlays/engine/experiments/sameday_backtest.py

For entertainment and analysis - not betting advice.
