"""Learn odds-api.io's response shape from a runner, since the docs are blocked.

The key belongs to odds-api.io, not the-odds-api.com, and the two return
completely different JSON. Writing an adapter against a guess is how the
surname matcher ended up pricing the wrong fighter, so the shape is read off a
real response first.

Prints STRUCTURE, never values beyond what is needed to recognise a field: a
bookmaker's name is structure, a fighter's odds are not the point here, and an
API key is never printed at all.

    ODDS_API_KEY=... python3 engine/describe_odds_api.py
"""

import json
import os
import sys

BASE = "https://api.odds-api.io/v3"
MAX_DEPTH = 4


def shape(value, depth=0):
    """A value described by its structure rather than its contents."""
    if depth >= MAX_DEPTH:
        return "..."
    if isinstance(value, dict):
        return {k: shape(v, depth + 1) for k, v in list(value.items())[:25]}
    if isinstance(value, list):
        if not value:
            return []
        return [shape(value[0], depth + 1), f"...{len(value)} items"]
    if isinstance(value, str):
        # Short strings are usually identifiers worth seeing; long ones are not.
        return value if len(value) <= 40 else f"<str len {len(value)}>"
    return type(value).__name__


def get(session, path, key, **params):
    url = f"{BASE}/{path}"
    try:
        response = session.get(url, params={"apiKey": key, **params},
                               timeout=25)
    except Exception as error:                      # noqa: BLE001
        return None, str(error).replace(key, "<redacted>")[:200]
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    try:
        return response.json(), None
    except json.JSONDecodeError:
        return None, f"not JSON: {response.text[:200]}"


def main():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ODDS_API_KEY is not set")
        return 2

    import requests
    session = requests.Session()

    print("=" * 70)
    print("ODDS-API.IO SCHEMA")
    print("=" * 70)

    sports, error = get(session, "sports", key)
    if error:
        print(f"  /sports failed - {error}")
        return 1

    names = []
    if isinstance(sports, list):
        for entry in sports:
            names.append(entry if isinstance(entry, str)
                         else (entry.get("key") or entry.get("id")
                               or entry.get("name")))
    elif isinstance(sports, dict):
        names = list(sports)
    print(f"\n  /sports returned {len(names)} entries")
    print(f"  shape: {json.dumps(shape(sports), indent=1)[:600]}")

    mma = [n for n in names if n and any(
        word in str(n).lower() for word in ("mma", "ufc", "martial", "fight"))]
    print(f"\n  MMA-looking sports: {mma}")

    for sport in (mma or [])[:3]:
        for path in ("events", "odds"):
            payload, error = get(session, path, key, sport=sport)
            print(f"\n  /{path}?sport={sport}")
            if error:
                print(f"    {error}")
                continue
            count = len(payload) if isinstance(payload, list) else "n/a"
            print(f"    entries: {count}")
            print(f"    shape: {json.dumps(shape(payload), indent=1)[:1800]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
