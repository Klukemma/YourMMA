"""Offline tests: synthetic prices, temp files, no network."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from butterfly import backtest as bt
from butterfly import grader
from butterfly.graph import Graph
from butterfly.ledger import Ledger, LedgerError
from butterfly.sources import macro, news, prices, sec


def _px(n=2000, seed=0, drift=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


# ---------- backtest ----------

def test_buy_and_hold_reproduces_the_asset_minus_one_spread():
    px = _px()
    d = bt.run(px, bt.always_in, spread=0.01)
    start = d.index[0]
    expected = (px.iloc[-1] / px.loc[start]) * (1 - 0.01)
    assert d["equity"].iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert bt.metrics(d)["trades"] == 1


def test_zero_spread_buy_and_hold_is_exactly_the_asset():
    px = _px(seed=3)
    d = bt.run(px, bt.always_in, spread=0.0)
    assert d["equity"].iloc[-1] == pytest.approx(px.iloc[-1] / px.loc[d.index[0]], rel=1e-12)


def test_every_switch_pays_the_spread():
    px = _px(seed=1, drift=0.0, vol=0.02)
    free = bt.run(px, bt.trend_signal(10), spread=0.0)
    paid = bt.run(px, bt.trend_signal(10), spread=0.01)
    switches = int((paid["pos"].diff().abs() > 0).sum()) + int(paid["pos"].iloc[0] != 0)
    assert paid["cost"].sum() == pytest.approx(0.01 * switches)
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_signal_cannot_see_the_future():
    """Changing prices after a month end must not change the position held
    up to (and including) the day after that month end's fill."""
    px = _px(seed=2)
    a = bt.run(px, bt.trend_signal(6), spread=0.01)
    cut = px.index[1200]
    shocked = px.copy()
    shocked.loc[shocked.index > cut] *= 3.0
    b = bt.run(shocked, bt.trend_signal(6), spread=0.01)
    assert (a.loc[:cut, "pos"] == b.loc[:cut, "pos"]).all()


def test_compare_starts_every_strategy_on_the_same_day():
    t = bt.compare(_px(seed=4), months_list=(6, 12), spread=0.01)
    assert t["start"].nunique() == 1


# ---------- ledger ----------

def _add(led, **kw):
    base = dict(ticker="XOM", direction=1, horizon_days=5, confidence=0.6,
                event="OPEC+ cut", chain=["cut", "oil up", "XOM revenue up"])
    base.update(kw)
    return led.add(**base)


def test_ledger_chain_and_tamper_detection(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    _add(led)
    _add(led, ticker="CVX")
    assert led.verify() == []
    lines = led.path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["confidence"] = 0.9                       # quietly "improve" an old call
    lines[0] = json.dumps(rec)
    led.path.write_text("\n".join(lines) + "\n")
    assert any("changed" in p for p in led.verify())
    with pytest.raises(LedgerError):
        _add(led, ticker="BA")


def test_ledger_refuses_deleting_a_line(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    for t in ("XOM", "CVX", "BA"):
        _add(led, ticker=t)
    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert led.verify()


def test_ledger_refuses_evidence_from_the_future(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    with pytest.raises(LedgerError, match="after the prediction"):
        _add(led, evidence=[{"url": "https://x", "fetched": later}])


@pytest.mark.parametrize("kw", [dict(direction=0), dict(horizon_days=7), dict(confidence=0.4),
                                dict(confidence=1.0), dict(ticker="SPY"), dict(chain=[])])
def test_ledger_rejects_bad_predictions(tmp_path, kw):
    with pytest.raises(LedgerError):
        _add(Ledger(tmp_path / "l.jsonl"), **kw)


# ---------- grader ----------

def _prices():
    idx = pd.bdate_range("2026-01-05", periods=40)
    spy = pd.Series(100.0, index=idx)
    xom = pd.Series(np.linspace(100, 139, 40), index=idx)      # +1 a day
    return pd.DataFrame({"SPY": spy, "XOM": xom})


def test_entry_is_the_close_after_the_prediction_day():
    days = _prices().index
    # Monday 2026-01-05 15:00 New York (20:00 UTC): entry must be Tuesday
    i = grader.entry_index(days, "2026-01-05T20:00:00+00:00")
    assert days[i] == pd.Timestamp("2026-01-06")
    # 2026-01-06 02:00 UTC is still Monday evening in New York: also Tuesday
    i = grader.entry_index(days, "2026-01-06T02:00:00+00:00")
    assert days[i] == pd.Timestamp("2026-01-06")


def test_grade_scores_direction_against_the_benchmark():
    px = _prices()
    pred = {"id": "P1", "ticker": "XOM", "benchmark": "SPY", "direction": 1, "horizon_days": 5,
            "confidence": 0.7, "created_at": "2026-01-05T20:00:00+00:00"}
    g = grader.grade_one(pred, px, spread=0.01)
    assert g["entry_date"] == "2026-01-06" and g["exit_date"] == "2026-01-13"
    assert g["r_stock"] == pytest.approx(106 / 101 - 1)
    assert g["hit"] and g["signed"] > 0
    assert g["long_net"] == pytest.approx(106 / 101 - 1 - 0.02)
    assert g["brier"] == pytest.approx(0.09)
    short = grader.grade_one({**pred, "direction": -1}, px)
    assert not short["hit"] and short["long_net"] is None


def test_open_window_is_pending_not_graded():
    px = _prices()
    pred = {"id": "P1", "ticker": "XOM", "direction": 1, "horizon_days": 63,
            "confidence": 0.6, "created_at": "2026-01-05T20:00:00+00:00"}
    g, pending = grader.grade([pred], px)
    assert g.empty and len(pending) == 1


def test_summary_compares_with_baselines():
    px = _prices()
    preds = [{"id": f"P{k}", "ticker": "XOM", "direction": 1, "horizon_days": 5, "confidence": 0.6,
              "created_at": f"2026-02-{k:02d}T15:00:00+00:00"} for k in range(2, 13)]
    g, _ = grader.grade(preds, px)
    s = grader.summarize(g)
    assert s["n"] == len(g) > 0
    # always-long equals our all-long calls, so it does not "beat" the baseline
    assert s["mean_signed_abnormal"] == pytest.approx(s["baseline_always_long"])
    assert s["beats_baselines"] is False


# ---------- graph ----------

def _graph():
    g = Graph()
    for i, t in [("e", "event"), ("oil", "commodity"), ("XOM", "company"), ("fuel", "commodity"),
                 ("air", "sector"), ("BA", "company")]:
        g.nodes[i] = {"type": t, "name": i}
    g.add_edge("e", "oil", 1, 0.8, relation="", source="t", verified=True)
    g.add_edge("oil", "XOM", 1, 0.5, relation="", source="t", verified=True)
    g.add_edge("oil", "fuel", 1, 1.0, relation="", source="t")
    g.add_edge("fuel", "air", -1, 0.5, relation="", source="t")
    g.add_edge("air", "BA", 1, 0.5, relation="", source="t")
    return g


def test_ripple_multiplies_signs_and_weights():
    r = {x["node"]: x for x in _graph().ripple("e", shock=1, max_hops=5)}
    assert r["XOM"]["net_impact"] == pytest.approx(0.4)
    assert r["air"]["net_impact"] == pytest.approx(-0.4)
    assert r["BA"]["net_impact"] == pytest.approx(-0.2)
    neg = {x["node"]: x for x in _graph().ripple("e", shock=-1, max_hops=5)}
    assert neg["XOM"]["net_impact"] == pytest.approx(-0.4)


def test_ripple_respects_hops_strength_and_verification():
    g = _graph()
    assert "BA" not in {x["node"] for x in g.ripple("e", max_hops=3)}
    assert "BA" not in {x["node"] for x in g.ripple("e", max_hops=5, min_strength=0.3)}
    assert {x["node"] for x in g.ripple("e", max_hops=5, verified_only=True)} == {"oil", "XOM"}


def test_ripple_handles_cycles():
    g = _graph()
    g.add_edge("XOM", "oil", 1, 0.5, relation="", source="t")
    g.ripple("e", max_hops=6)     # must terminate


def test_graph_refuses_edges_without_a_source():
    g = _graph()
    with pytest.raises(ValueError, match="source"):
        g.add_edge("XOM", "BA", 1, 0.5, relation="", source="")


def test_seed_graph_is_valid():
    from butterfly import config
    g = Graph.load(config.ROOT / "data" / "graph.json")
    tickers = {n["ticker"] for n in g.nodes.values() if n.get("ticker")}
    assert tickers <= set(config.UNIVERSE)
    assert g.ripple("event:panama_drought")


# ---------- parsers ----------

def test_parse_yahoo_chart():
    payload = {"chart": {"result": [{"meta": {"gmtoffset": -18000},
                                     "timestamp": [1767623400, 1767709800],
                                     "indicators": {"quote": [{"close": [10.0, None]}],
                                                    "adjclose": [{"adjclose": [9.5, None]}]}}]}}
    df = prices.parse_yahoo_chart(payload)
    assert list(df["date"]) == ["2026-01-05"] and df["adj_close"].iloc[0] == 9.5


def test_parse_tiingo():
    df = prices.parse_tiingo([{"date": "2026-01-05T00:00:00.000Z", "close": 10.0, "adjClose": 9.4},
                              {"date": "2026-01-06T00:00:00.000Z", "close": None, "adjClose": None}])
    assert list(df["date"]) == ["2026-01-05"] and df["adj_close"].iloc[0] == 9.4
    assert df["source"].iloc[0] == "tiingo"


def test_parse_fred_drops_missing():
    df = macro.parse_fred_csv("observation_date,DGS10\n2026-01-02,4.1\n2026-01-05,.\n", "DGS10")
    assert len(df) == 1 and df["value"].iloc[0] == 4.1


def test_parse_sec_recent_filings():
    payload = {"cik": "320193", "filings": {"recent": {
        "form": ["8-K", "10-K", "4"], "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002", "x"],
        "filingDate": ["2026-01-30", "2025-10-31", "2026-01-01"],
        "acceptanceDateTime": ["2026-01-30T16:31:02.000Z", "2025-10-31T06:01:00.000Z", ""],
        "primaryDocument": ["a8k.htm", "a10k.htm", "f4.xml"], "items": ["2.02,9.01", "", ""]}}}
    rows = sec.parse_recent_filings(payload, {"8-K", "10-K"})
    assert [r["form"] for r in rows] == ["8-K", "10-K"]
    assert rows[0]["items"] == ["2.02", "9.01"] and rows[0]["item_names"][0] == "results of operations"
    assert rows[0]["url"].endswith("/320193/000032019326000001/a8k.htm")


def test_risk_factor_section_skips_table_of_contents():
    text = ("Item 1A. Risk Factors 12\nItem 1B. Unresolved 20\n" +
            "Item 1A. Risk Factors\nOur supply chain depends on Asia. " * 1 + "x" * 500 +
            "\nItem 1B. Unresolved Staff Comments")
    s = sec.risk_factors(text)
    assert "supply chain" in s and len(s) > 500


def test_news_append_dedupes(tmp_path, monkeypatch):
    from butterfly import config
    monkeypatch.setattr(config, "NEWS_DIR", tmp_path)
    arts = news.parse_gdelt({"articles": [{"url": "u1", "title": "a", "seendate": "20261117T101500Z"},
                                          {"url": "u1", "title": "a"}, {"url": "u2"}]}, "energy", "t")
    assert arts[0]["seen"] == "2026-11-17T10:15:00+00:00"
    assert news.append(arts) == 2 and news.append(arts) == 0
