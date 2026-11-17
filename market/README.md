# Butterfly: market research engine

An event-to-effect research engine for stocks. It gathers prices, macro data, FX,
company filings and world news. It maps what depends on what, traces the ripple an
event sends through that map, and **writes every prediction down before its outcome
exists** so it can be graded honestly afterwards.

Research comes first and trading later. Nothing here places an order. Real money
waits until the ledger shows an edge after costs (see *Gates*).

> Development home. This folder lives in a public repository: never put keys or
> account details here. `market/.env` is gitignored.

## Quick start

```bash
cd market
pip install -r requirements.txt
python -m pytest tests -q

python -m butterfly fetch all              # prices, macro, fx, news, filings
python -m butterfly backtest --cash DTB3   # trend switch vs buy-and-hold, 1% spread charged
python -m butterfly ripple event:opec_cut --tickers-only
python -m butterfly predict --ticker XOM --direction 1 --horizon 21 --confidence 0.6 \
    --event "OPEC+ cut announced" --chain "cut tightens supply" "oil rises" "XOM upstream revenue rises"
python -m butterfly verify                 # the ledger was never edited
python -m butterfly grade                  # score every closed prediction
```

## What's in it

| Module | Job |
|---|---|
| `sources/prices.py` | Daily dividend-adjusted prices from Tiingo (free key), with Yahoo / Stooq fallbacks that mostly refuse scripts now |
| `sources/macro.py` | FRED: yields, T-bills, oil, gas, dollar, VIX, credit spreads, breakevens. ECB FX rates via Frankfurter |
| `sources/sec.py` | EDGAR: each company's 10-K / 10-Q / 8-K with exact acceptance times; filing text; Risk Factors and Business sections |
| `sources/news.py` | GDELT world news for standing themes (shipping, energy, rates, trade, chips, weather, FX), de-duplicated and time-stamped |
| `graph.py` | The dependency map, plus `ripple()`: signs and weights multiplied along every path from an event |
| `ledger.py` | Append-only, hash-chained prediction log |
| `grader.py` | Scores predictions against the benchmark, with baselines and calibration |
| `backtest.py` | Monthly trend switch vs buy-and-hold, with the spread charged on every trade |

## The rules that keep it honest

1. **No backdating.** The ledger stamps `created_at` itself, and a caller can't
   supply one. Each record carries the hash of the one before it, so editing or
   deleting an old call is detected (`verify`). Git history adds an outside
   timestamp.
2. **No evidence from the future.** Evidence fetched after the prediction is refused.
3. **No same-day entry.** A prediction enters at the close of the first trading
   day *after* the day it was made (New York time).
4. **Accuracy isn't edge.** A call is graded on return relative to SPY, and the
   whole set must beat two free baselines on the same windows: *always long* and
   *21-day momentum*. Long calls are also scored as real xStocks trades, net of
   the 1% spread each way. The MMA engine taught this lesson: 57–63% hit rates
   that still lost money.
5. **AI models know the past.** Any model reasoning about old news already knows
   what happened next, so historical "tests" of the reasoning are contaminated.
   The ledger only counts predictions made live, going forward.
6. **Every map edge names its source.** The seed map (`data/graph.json`) comes
   from general knowledge, and every edge is `verified: false`. An edge becomes
   verified once it is confirmed against a filing, and its `source` then becomes
   that filing's URL. `ripple --verified-only` uses confirmed links only.

## Gates

| Stage | Pass to move on |
|---|---|
| Backtest | Trend switch beats buy-and-hold after the spread in most 5-year blocks and most windows (6–12 months), not one lucky setting. Otherwise: buy and hold |
| Ledger | 150+ graded live predictions; mean signed abnormal return beats both baselines; long calls positive after 2% round-trip; confidence calibrated |
| Paper | 2–3 months of a pretend $200 account tracking the backtest / ledger |
| Live | Real money only after all of the above |

## Where it runs

| Source | Dev container | GitHub Actions | Needs |
|---|---|---|---|
| FRED macro, ECB FX | yes | yes | nothing |
| Tiingo prices | yes, once the key is saved as a network secret | yes | `TIINGO_API_KEY` (free account) |
| SEC EDGAR | yes | yes | `SEC_USER_AGENT` = your name and a contact email |
| GDELT news | no (refused) | yes | nothing |

`.github/workflows/market.yml` runs the whole loop on GitHub Actions on every push
to the dev branch (and, once merged to the default branch, every weekday after
the US close), then commits the data back. Put both keys in `market/.env`
locally and as repository secrets for Actions.

## Next steps

- Verify the seed edges against 10-K Risk Factors / Business sections, and grow
  the map from filings: suppliers, customers, geographic revenue, commodity and
  FX exposure.
- Daily reasoning pass: new 8-Ks + news clusters -> events -> ripple -> AI
  reasoning -> ledger entries, run as scheduled sessions.
- Paper portfolio from ledger calls above a confidence threshold.
