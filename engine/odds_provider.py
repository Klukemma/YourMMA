"""Which odds service to call, and how, as data rather than as code.

The engine was written against the-odds-api.com with its endpoint, its
parameter names and its response shape spread through fetch_mma_odds. When the
key turned out not to belong to that provider, switching meant editing the
fetch, the parser and the tests - so the natural move was to guess which
provider and rewrite for it, which is exactly the move that produced a matcher
pricing the wrong fighter.

A provider is a handful of facts: where to call, how the key travels, what the
sport is called, and where the fighters and prices sit in the reply. Holding
them as data means adding one is a dict, not a rewrite, and ODDS_PROVIDER picks
between them at run time.

WHAT IS DELIBERATELY NOT HERE. No provider is guessed at. Each entry below was
either read off a real response (the-odds-api.com, from the working
integration) or is marked unverified, and an unverified provider says so out
loud when it is used, because a field path taken from documentation nobody
could reach is a hypothesis, not a fact.
"""

import os

# Each provider names, in order:
#   base        the URL to call for upcoming events with prices
#   auth        ("query", <param>) | ("header", <name>) | ("bearer", <header>)
#   sport       what this service calls MMA, and the parameter that carries it
#   extras      any other query parameters the endpoint needs
#   paths       where to find the pieces in one event of the response
#   verified    True only when the shape was read off a real reply
PROVIDERS = {
    "the-odds-api": {
        "base": "https://api.the-odds-api.com/v4/sports/"
                "mma_mixed_martial_arts/odds",
        "auth": ("query", "apiKey"),
        "sport": None,
        "extras": {"regions": "us", "markets": "h2h",
                   "oddsFormat": "american"},
        "paths": {
            "home": "home_team", "away": "away_team",
            "start": "commence_time", "books": "bookmakers",
            "book_name": "title", "markets": "markets",
            "market_key": "key", "market_h2h": "h2h",
            "outcomes": "outcomes", "outcome_name": "name",
            "outcome_price": "price",
        },
        "quota_header": "x-requests-remaining",
        "verified": True,
    },
    "odds-api-io": {
        "base": "https://api.odds-api.io/v3/odds",
        "auth": ("query", "apiKey"),
        "sport": ("sport", "mma"),
        "extras": {},
        # UNVERIFIED. /v3/sports is public and answered 200, but every
        # authenticated endpoint refused the key, so no real reply was ever
        # seen and these paths are a guess until one is.
        "paths": {},
        "quota_header": None,
        "verified": False,
    },
}

DEFAULT = "the-odds-api"


def name():
    """The selected provider's name."""
    return os.environ.get("ODDS_PROVIDER", DEFAULT).strip() or DEFAULT


def selected():
    """The selected provider's settings, or a clear error naming the choices."""
    chosen = name()
    if chosen not in PROVIDERS:
        raise KeyError(
            f"ODDS_PROVIDER={chosen!r} is not one of: "
            + ", ".join(sorted(PROVIDERS)))
    return PROVIDERS[chosen]


def request_for(key, provider=None):
    """(url, params, headers) for one upcoming-odds call."""
    provider = provider or selected()
    params = dict(provider["extras"])
    headers = {}

    style, field = provider["auth"]
    if style == "query":
        params[field] = key
    elif style == "bearer":
        headers[field] = f"Bearer {key}"
    else:
        headers[field] = key

    if provider["sport"]:
        sport_param, sport_value = provider["sport"]
        params[sport_param] = sport_value

    return provider["base"], params, headers


def warn_if_unverified(provider=None, echo=print):
    """Say so when a provider's response shape has never been seen.

    An unverified provider will parse to nothing and show a card with no
    prices, which looks exactly like a quiet day at the bookmakers. Saying it
    out loud is the difference between a known gap and a mystery.
    """
    provider = provider or selected()
    if provider.get("verified"):
        return False
    echo(f"  [Odds] Provider {name()!r} has an UNVERIFIED response shape. "
         "No real reply from it has ever been read, so the field paths are a "
         "guess and prices may silently come back empty.")
    return True
