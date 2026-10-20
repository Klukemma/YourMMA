"""Player news for the week's games, from public NFL news feeds.

The runner (GitHub's machines, which have the open internet) reads a few
public RSS feeds each run, keeps every item with the moment it was fetched,
and the weekly run attaches to each game the items that name one of its two
teams or one of their players. The app shows the headline, the source and
the time, linked to the article. Nothing here moves a number: news is SHOWN,
NOT APPLIED, as the MMA app's fight-week news is.

POINT IN TIME. An item carries `published` (the feed's own date) and
`gathered` (when this code first saw it). A game shows only items that were
both published AND gathered before its kickoff. The gathered time is our own
clock and the only one that can be trusted: one feed's dates run hours
behind (RotoWire stamped "won't return Sunday" - an in-game injury - before
the 1pm kickoff it happened in), and a match report must never appear as
pre-game news.

MATCHING. A player is matched by his full name as the roster prints it
(two words at least, whole words only - "Allen" alone would match half the
league); a team by its full name or its nickname ("Chiefs"). Items naming a
player outrank items naming only a team.
"""

import html
import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import nfl_data as nd

CACHE = nd.DATA / "news.json"
SCHEMA = 1
KEEP_DAYS = 10
PER_GAME = 8
FEEDS = {
    # ESPN's feed answered the runner with an empty body; Pro Football
    # Rumors is injury, roster and transaction news, the player news wanted.
    "Pro Football Rumors": "https://www.profootballrumors.com/feed",
    "CBS Sports": "https://www.cbssports.com/rss/headlines/nfl/",
    "ProFootballTalk": "https://profootballtalk.nbcsports.com/feed/",
    "Yahoo Sports": "https://sports.yahoo.com/nfl/rss.xml",
    "RotoWire": "https://www.rotowire.com/rss/news.php?sport=NFL",
}
USER_AGENT = "YourNFL/1.0 (+https://github.com/Klukemma/YourMMA)"
# Not news: sportsbook promotions and TV listings name every team playing.
NOT_NEWS = re.compile(
    r"promo code|bonus code|bonus bets|sign-?up bonus|sportsbook promo|kalshi|"
    r"how to watch|live stream|tv channel|where to watch|best bets|player props?\b|parlay|"
    r"betting promos|open thread|live chat|live updates|discussion|odds, predictions|expert picks|"
    r"odds, spreads|odds, picks|picks, predictions",
    re.IGNORECASE)


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(text):
    if not text:
        return None
    text = text.strip()
    try:
        return parsedate_to_datetime(text).astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except ValueError:
        return None


def _text(node):
    """Element text with markup and entities removed, whitespace collapsed."""
    if node is None or node.text is None:
        return ""
    raw = re.sub(r"<[^>]+>", " ", html.unescape(node.text))
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def parse_feed(xml_bytes, source):
    """Items of an RSS 2.0 or Atom feed: [{title, link, summary, published,
    source}]. Items without a title or a link are dropped."""
    root = ElementTree.fromstring(xml_bytes)
    atom = "{http://www.w3.org/2005/Atom}"
    items = []
    for item in root.iter("item"):
        items.append({"title": _text(item.find("title")), "link": _text(item.find("link")),
                      "summary": _text(item.find("description")),
                      "published": _parse_date(_text(item.find("pubDate")))})
    for entry in root.iter(f"{atom}entry"):
        link = entry.find(f"{atom}link")
        items.append({"title": _text(entry.find(f"{atom}title")),
                      "link": link.get("href", "") if link is not None else "",
                      "summary": _text(entry.find(f"{atom}summary")) or _text(entry.find(f"{atom}content")),
                      "published": _parse_date(_text(entry.find(f"{atom}published"))
                                               or _text(entry.find(f"{atom}updated")))})
    out = []
    for it in items:
        if it["title"] and it["link"].startswith(("http://", "https://")):
            it["source"] = source
            it["summary"] = it["summary"][:320]
            out.append(it)
    return out


def empty():
    return {"schema": SCHEMA, "items": {}, "fetches": []}


def load(path=CACHE):
    path = Path(path)
    if not path.exists():
        return empty()
    cache = json.loads(path.read_text())
    if cache.get("schema") != SCHEMA:
        raise ValueError(f"news cache is schema {cache.get('schema')}, expected {SCHEMA}")
    return cache


def save(cache, path=CACHE):
    Path(path).write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")


def merge(cache, items, now=None):
    """Add items not seen before (by link), stamping when they were gathered;
    drop items older than KEEP_DAYS. Returns how many were new."""
    now = now or _now()
    new = 0
    for it in items:
        if it["link"] in cache["items"]:
            continue
        published = it["published"] or now
        cache["items"][it["link"]] = {
            "title": it["title"], "link": it["link"], "summary": it["summary"],
            "source": it["source"], "published": _stamp(published), "gathered": _stamp(now),
            "dated_by_feed": it["published"] is not None,
        }
        new += 1
    cutoff = now - timedelta(days=KEEP_DAYS)
    cache["items"] = {k: v for k, v in cache["items"].items()
                      if _parse_date(v["published"]) >= cutoff}
    return new


def refresh(cache, feeds=FEEDS, log=print, now=None):
    """Read every feed; one that fails is logged and skipped."""
    import requests
    now = now or _now()
    for source, url in feeds.items():
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            items = parse_feed(r.content, source)
            new = merge(cache, items, now)
            status = r.status_code
            log(f"   [news] {source}: {len(items)} items, {new} new")
        except Exception as err:
            items, status = [], f"failed: {str(err)[:100]}"
            log(f"   [news] {source}: {status}")
        cache["fetches"].append({"at": _stamp(now), "source": source, "status": status, "items": len(items)})
    cache["fetches"] = cache["fetches"][-200:]


def _pattern(names):
    names = sorted({n for n in names if n and len(n.split()) >= 2 or n in _NICKNAMES}, key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)


_NICKNAMES = {name.split()[-1] for name, *_ in nd.TEAMS.values()}


def for_game(cache, teams, players, kickoff, limit=PER_GAME):
    """Items naming either team or one of `players` (full names), published
    before `kickoff` (a UTC timestamp), newest first, player items first.
    Each item gains `teams` and `players` - what it matched."""
    kickoff = kickoff if kickoff.tzinfo else kickoff.replace(tzinfo=timezone.utc)
    team_names = {}
    for code in teams:
        full = nd.TEAMS[code][0]
        team_names[full.lower()] = code
        team_names[full.split()[-1].lower()] = code
    team_re = _pattern(list({nd.TEAMS[c][0] for c in teams} | {nd.TEAMS[c][0].split()[-1] for c in teams}))
    player_re = _pattern(players)
    matched = []
    for it in cache["items"].values():
        published, gathered = _parse_date(it["published"]), _parse_date(it.get("gathered"))
        if published is None or gathered is None or published >= kickoff or gathered >= kickoff:
            continue
        text = f"{it['title']} {it['summary']}"
        if NOT_NEWS.search(it["title"]):
            continue
        hit_players = sorted({m.group(0) for m in player_re.finditer(text)}) if player_re else []
        hit_teams = sorted({team_names[m.group(0).lower()] for m in team_re.finditer(text)}) if team_re else []
        if hit_players or hit_teams:
            matched.append(dict(it, players=hit_players, teams=hit_teams))
    matched.sort(key=lambda it: (not it["players"], -_parse_date(it["published"]).timestamp()))
    # One story, once: Yahoo syndicates ProFootballTalk and others word for word.
    seen, unique = set(), []
    for it in matched:
        key = re.sub(r"[^a-z0-9]+", " ", it["title"].lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique[:limit]
