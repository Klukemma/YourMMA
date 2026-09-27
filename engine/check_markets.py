"""Which betting markets does this key's plan actually expose?

The method question has no market anchor. The failure model works on the
winner question because three of its five features are market-derived - the
de-vigged price, the gap between the model and that price, and whether the
pick is an underdog. Strip those and it scores 0.605 where confidence alone
scores 0.609, which is to say worse than nothing.

If "fight goes the distance" is available on this plan, the same construction
becomes possible for the method question. If it is not, a method-side
wrongness layer has no second opinion to work from and should not be built.

That is a yes-or-no question and this answers it for ONE event's worth of
credits. It asks the events endpoint (free) for a single event id, then asks
that event for every market the docs list, and reports which ones come back.

    ODDS_API_KEY=... python3 engine/check_markets.py

Prints market names and bookmaker counts. Never prints the key, and never
prints a price - the question here is what exists, not what it costs.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4"
SPORT = "mma_mixed_martial_arts"

# h2h is the one already in use and is asked for as a control: if it comes
# back and the others do not, the others are genuinely absent rather than the
# request being wrong.
MARKETS = [
    ("h2h", "moneyline - the one already used, asked as a control"),
    ("totals", "over/under rounds - the distance question in another shape"),
    ("fight_result_method", "method of victory"),
    ("method_of_victory", "method of victory, alternate spelling"),
    ("fight_goes_the_distance", "goes the distance - the one that matters"),
    ("fight_to_go_distance", "goes the distance, alternate spelling"),
    ("outrights", "event futures - asked only to see how the plan answers a\n     market it certainly does not carry for a single bout"),
]


def redact(text, key):
    """A key must never reach a log, a terminal or a commit."""
    if not key:
        return text
    return text.replace(key, "<redacted>").replace(
        urllib.parse.quote(key), "<redacted>")


def call(path, params, key):
    url = f"{BASE}{path}?" + urllib.parse.urlencode({**params, "apiKey": key})
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, json.loads(response.read().decode()), \
                response.headers.get("x-requests-remaining")
    except urllib.error.HTTPError as err:
        body = err.read().decode()[:200]
        return err.code, redact(body, key), err.headers.get(
            "x-requests-remaining")
    except Exception as err:                       # noqa: BLE001
        return None, redact(str(err), key), None


def main():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        sys.exit("ODDS_API_KEY is not set. Nothing was requested.")

    print("=" * 70)
    print("WHICH MARKETS DOES THIS PLAN EXPOSE?")
    print("=" * 70)

    # The events list is free on this API and gives an id to ask about.
    status, events, remaining = call(f"/sports/{SPORT}/events", {}, key)
    print(f"\n  events endpoint: {status}   credits left: {remaining}")
    if status != 200 or not events:
        print(f"  {events}")
        sys.exit(1)

    event = events[0]
    print(f"  asking about: {event.get('home_team')} vs "
          f"{event.get('away_team')}  ({event.get('commence_time')})")

    print(f"\n  {'market':<26}{'status':>8}   what came back")
    print("  " + "-" * 66)
    available = []
    for market, why in MARKETS:
        status, body, remaining = call(
            f"/sports/{SPORT}/events/{event['id']}/odds",
            {"regions": "us", "markets": market, "oddsFormat": "american"},
            key)
        if status == 200 and isinstance(body, dict):
            books = body.get("bookmakers") or []
            names = sorted({m["key"] for b in books for m in b.get("markets", [])})
            if names:
                available.append(market)
                print(f"  {market:<26}{status:>8}   {len(books)} bookmakers, "
                      f"markets {names}")
            else:
                print(f"  {market:<26}{status:>8}   accepted, no bookmaker "
                      f"is pricing it")
        else:
            detail = body if isinstance(body, str) else json.dumps(body)[:90]
            print(f"  {market:<26}{status:>8}   {detail}")

    print(f"\n  credits left: {remaining}")
    print("\n" + "=" * 70)
    distance = [m for m in available
                if "distance" in m or m == "totals"]
    if distance:
        print(f"  A DISTANCE MARKET EXISTS: {distance}")
        print("  The method question can have a market anchor, so the")
        print("  wrongness layer can be built for it the way it was for the")
        print("  winner question.")
    else:
        print("  NO DISTANCE MARKET on this plan.")
        print("  A method-side wrongness layer would have no second opinion")
        print("  to read, and the winner-side one scored worse than plain")
        print("  confidence without its market features. Do not build it on")
        print("  this evidence.")
    print("=" * 70)


if __name__ == "__main__":
    main()
