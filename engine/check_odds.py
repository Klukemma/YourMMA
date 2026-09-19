"""Verify the odds key in thirty seconds instead of a twenty-minute run.

Adding ODDS_API_KEY and finding out whether it works should not require
predicting a whole card. This makes one request, reports what came back, and
says how much quota is left.

    ODDS_API_KEY=... python3 engine/check_odds.py

It is also the only place the quota is ever read. The Odds API returns the
remaining balance in a response header on every call, and nothing in the engine
was looking at it, so the first sign of a spent quota would have been a card
that quietly had no prices - the same shape of silence as the key that was
never passed to the runner.
"""

import json
import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

ENDPOINT = ("https://api.the-odds-api.com/v4/sports/"
            "mma_mixed_martial_arts/odds")


def redact(text, key):
    """Take the key out of anything about to be printed.

    requests puts the full URL in its exception text, query string included, so
    a connection failure prints the key in clear. GitHub masks registered
    secrets in its own logs, but that does not cover a terminal, an uploaded
    sync.log, or anyone running this locally - and a key is not worth trusting
    to one layer of masking.
    """
    text = str(text)
    if key:
        text = text.replace(key, "<redacted>")
    return text

# The same shape predict_card asks for, and the cheapest one available: a call
# costs markets x regions credits, so one market in one region is 1 credit.
# The free tier's 500 credits are therefore 500 real calls, not 83.
PARAMS = {"regions": "us", "markets": "h2h", "oddsFormat": "american"}


def describe(status, headers, payload):
    """Turn one API response into something worth printing."""
    remaining = headers.get("x-requests-remaining")
    used = headers.get("x-requests-used")
    out = {"status": status, "remaining": remaining, "used": used,
           "events": 0, "books": 0, "fighters": []}

    if status == 401:
        out["problem"] = ("The key was rejected. Check it was copied whole, "
                          "and that it is the key rather than the account id.")
        return out
    if status == 429:
        out["problem"] = ("Quota is spent for this period. It resets on the "
                          "first of the month on the free tier.")
        return out
    if status != 200:
        out["problem"] = f"The API returned {status}."
        return out

    events = payload if isinstance(payload, list) else []
    out["events"] = len(events)
    books = set()
    for event in events:
        for book in event.get("bookmakers", []):
            books.add(book.get("title") or book.get("key"))
        home, away = event.get("home_team"), event.get("away_team")
        if home and away:
            out["fighters"].append(f"{home} vs {away}")
    out["books"] = len(books)
    if not events:
        out["problem"] = ("The key works, but no MMA events are priced right "
                          "now. Books usually post a card a week or so out, "
                          "so this is normal between events rather than a "
                          "fault.")
    return out


def report(result):
    print("=" * 66)
    print("ODDS API CHECK")
    print("=" * 66)
    print(f"  HTTP status      {result['status']}")
    if result.get("remaining") is not None:
        print(f"  credits left     {result['remaining']}")
        print(f"  credits used     {result['used']}")
    print(f"  events priced    {result['events']}")
    print(f"  bookmakers       {result['books']}")
    if result.get("problem"):
        print(f"\n  {result['problem']}")
    for line in result["fighters"][:12]:
        print(f"    {line}")
    if len(result["fighters"]) > 12:
        print(f"    ... and {len(result['fighters']) - 12} more")
    print("=" * 66)
    return result["status"] == 200


def main():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ODDS_API_KEY is not set.")
        print("  Free key: https://the-odds-api.com/  (500 credits a month;")
        print("  this engine asks for one market in one region, so a credit")
        print("  is one call.)")
        print("  Locally:  export ODDS_API_KEY='...'")
        print("  In CI:    Settings -> Secrets and variables -> Actions")
        return 2

    import requests
    try:
        response = requests.get(ENDPOINT, params={**PARAMS, "apiKey": key},
                                timeout=20)
    except Exception as error:                      # noqa: BLE001
        print(f"Could not reach the API: {redact(error, key)}")
        print("A sandbox may block this host; a CI runner will not.")
        return 3

    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = []
    return 0 if report(describe(response.status_code, response.headers,
                                payload)) else 1


if __name__ == "__main__":
    sys.exit(main())
