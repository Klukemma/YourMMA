"""Run the app's JavaScript against advantages computed in Python.

The app reimplements matchup.py in a second language so a phone can compare
any two fighters without a server. Two implementations of one formula diverge
the moment either is edited, and a browser gives no sign - it just shows a
slightly wrong number, confidently.

So the JavaScript is not trusted. It is executed here, on real fighters, and
compared against the Python it mirrors.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import app_export as ax
import career_stats as cs
import matchup as mu
import matchup_inputs as mi

APP = ENGINE.parent / "app"
UFC_CSV = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"

# Every advantage the app draws, named the same on both sides.
ADVANTAGES = {
    "strikingAdvantage": mu.striking_advantage,
    "takedownAdvantage": mu.takedown_advantage,
    "controlAdvantage": mu.control_advantage,
    "submissionAdvantage": mu.submission_advantage,
    "grapplingAdvantage": mu.grappling_advantage,
}
SIZE = {
    "sizeAdvantage": mu.size_advantage,
    "massAdvantage": mu.mass_advantage,
    "heightAdvantage": mu.height_advantage_cm,
    "reachAdvantage": mu.reach_advantage_cm,
}


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    return node


@pytest.fixture(scope="module")
def pairs():
    """Real fighters, including thin records, so the NaN paths are exercised."""
    if not UFC_CSV.exists():
        pytest.skip("dataset not present")
    df = pd.read_csv(UFC_CSV, low_memory=False).sort_values("date")
    df = df.reset_index(drop=True)
    final = cs.final_stats(df)

    body = {}
    for corner in cs.CORNERS:
        columns = [f"{corner}_id", f"{corner}_height", f"{corner}_reach",
                   f"{corner}_weight"]
        for row in df[columns].itertuples(index=False):
            if row[0] == row[0]:
                body[row[0]] = {"height_cm": row[1], "reach_cm": row[2],
                                "weight_kg": row[3]}

    chosen = list(final.index[:40]) + list(final[final["cd_bouts"] >= 10]
                                           .index[:40])
    out = []
    for i in range(0, len(chosen) - 1, 2):
        pair = []
        for fighter_id in (chosen[i], chosen[i + 1]):
            stats = {k: (None if v != v else float(v))
                     for k, v in final.loc[fighter_id].items()}
            stats.update({k: (None if v != v else float(v))
                          for k, v in body.get(fighter_id, {}).items()})
            pair.append(stats)
        out.append(pair)
    return out


def _python_expected(pair):
    """What matchup.py says, through the same adapter the pipeline uses."""
    row = {}
    for corner, stats in zip(("r", "b"), pair):
        row.update({f"{corner}_{k}": v for k, v in stats.items()})
    frame = pd.DataFrame([row])
    careers = frame

    forms = [mi.form_frame(careers, corner).iloc[0] for corner in ("r", "b")]
    red, blue = (mu.Form(**f.to_dict()) for f in forms)
    bodies = [mu.Physique(
        height_cm=mu.plausible_height_cm(stats.get("height_cm")),
        reach_cm=mu.plausible_reach_cm(stats.get("reach_cm")),
        weight_kg=stats.get("weight_kg")) for stats in pair]

    values = {name: fn(red, blue) for name, fn in ADVANTAGES.items()}
    values.update({name: fn(bodies[0], bodies[1]) for name, fn in SIZE.items()})
    return values


def test_the_javascript_reproduces_every_advantage_python_computes(pairs):
    """The whole point of the file. A drift in either language fails here."""
    node = _node()
    constants = ax.constants_payload(mu)["matchup"]

    script = APP / "matchup.js"
    assert script.exists(), script
    runner = f"""
      import {{ makeMatchup, makeForm }} from {json.dumps(str(script))};
      const chunks = [];
      for await (const chunk of process.stdin) chunks.push(chunk);
      const input = JSON.parse(Buffer.concat(chunks).toString());
      const M = makeMatchup(input.constants);
      const form = makeForm(input.constants);
      const names = ["strikingAdvantage","takedownAdvantage","controlAdvantage",
                     "submissionAdvantage","grapplingAdvantage","sizeAdvantage",
                     "massAdvantage","heightAdvantage","reachAdvantage"];
      const out = input.pairs.map(([a, b]) => {{
        const red = form(a), blue = form(b);
        const row = {{}};
        for (const n of names) {{
          const v = M[n](red, blue);
          row[n] = Number.isFinite(v) ? v : null;
        }}
        return row;
      }});
      process.stdout.write(JSON.stringify(out));
    """
    payload = json.dumps({
        "constants": constants,
        "pairs": [[{k: v for k, v in stats.items()} for stats in pair]
                  for pair in pairs],
    })
    result = subprocess.run([node, "--input-type=module", "-e", runner],
                            input=payload, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr[:2000]
    actual = json.loads(result.stdout)
    assert len(actual) == len(pairs)

    compared = 0
    for pair, got in zip(pairs, actual):
        expected = _python_expected(pair)
        for name, value in expected.items():
            theirs = got[name]
            if value != value:  # NaN in Python must be null in JavaScript
                assert theirs is None, f"{name}: python NaN, js {theirs}"
                continue
            assert theirs is not None, f"{name}: python {value}, js null"
            assert abs(theirs - value) < 1e-9, (name, value, theirs)
            compared += 1
    # Guard against a test that passes because everything was NaN.
    assert compared > 100, f"only {compared} real numbers compared"


def test_every_constant_the_javascript_reads_is_exported(pairs):
    """A constant the app reads but the engine never writes is `undefined` in
    JavaScript, which turns every arithmetic result into NaN silently."""
    exported = set(ax.constants_payload(mu)["matchup"])
    source = (APP / "matchup.js").read_text()
    # C.NAME and C["NAME"] and the string tables that index into C.
    used = set(re.findall(r'C\.([A-Z][A-Z0-9_]{3,})', source))
    used |= set(re.findall(r'"([A-Z][A-Z0-9_]{3,})"', source))
    missing = used - exported
    assert not missing, f"matchup.js reads constants nobody exports: {missing}"
