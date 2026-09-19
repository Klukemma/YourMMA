# The app

A static page. It reads JSON the engine writes into `app/data/` and runs
entirely in the browser, which is what lets it work from a phone with nothing
deployed but files.

    app/
      index.html     the page
      app.js         rendering, tab by tab
      matchup.js     a checked port of engine/matchup.py
      data/*.json    written by the engine, never edited by hand

## Where the data comes from

| file | written by | holds |
|---|---|---|
| `card.json` | `engine/predict_card.py` | the current card: model pick, odds, simulation, parlays |
| `fighters.json` | `engine/build_app_data.py` | every fighter's career state, columnar |
| `names.json` | `engine/build_app_data.py` | id to display name, for the search box |
| `performance.json` | `engine/build_app_data.py` | track record, the three strategies, AUC and ROI by year |
| `constants.json` | `engine/build_app_data.py` | the league baselines and weights `matchup.js` needs |

`build_app_data.py` runs in seconds and needs only the dataset. `card.json` is
written at the end of a full `predict_card.py` run, because it needs the
trained model.

    python3 engine/build_app_data.py

CI refreshes the first four on every sync.

## Why matchup.js is not trusted

The Compare tab computes advantages in the browser so any two fighters can be
matched without a server. That means one formula exists in two languages, and
two implementations diverge the moment either is edited - silently, because a
browser shows a slightly wrong number with no complaint.

So `engine/tests/test_app_js.py` executes `matchup.js` under node against
advantages computed in Python, on real fighters including thin records, and
fails at the ninth decimal place. It also checks that every constant the
JavaScript reads is one the engine actually exports: a constant nobody writes
is `undefined` in JavaScript, which turns the arithmetic into `NaN` without an
error.

Nothing in `matchup.js` hardcodes a league baseline. They all come from
`constants.json`, written straight off `matchup.py`, so a recalibration cannot
leave the app computing against last year's numbers.

## What the app will not do

- It will not show a probability the engine did not produce. A missing number
  renders as a dash, never as zero and never as 50%.
- It will not simulate a fighter it cannot measure. More than half a fighter's
  rates assumed means no simulation, because a confident 50/50 between two
  league-average fighters wearing real names is worse than saying nothing.
- Every screen carries what its numbers are worth out of sample. The best
  strategy returned +0.3% over 2,339 bets and the interval spans zero; that
  sentence ships with the page rather than living in a commit message.

## Serving it

Any static host. Locally:

    python3 -m http.server 8000
    # then open http://localhost:8000/app/

On GitHub Pages, enable Pages for the repository (Settings -> Pages, deploy
from a branch, root) and the app is at `/app/`.
