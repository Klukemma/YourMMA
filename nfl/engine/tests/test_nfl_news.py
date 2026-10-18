"""Player news: feeds parsed, matched to a game's teams and players, and
never shown for a game if published after its kickoff."""

from datetime import datetime, timezone

import nfl_news as nn

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>NFL</title>
<item><title>Chiefs' Patrick Mahomes limited in practice with ankle</title>
<link>https://example.com/a</link><description>&lt;p&gt;Mahomes was a limited participant.&lt;/p&gt;</description>
<pubDate>Fri, 16 Oct 2026 18:00:00 GMT</pubDate></item>
<item><title>Raiders sign a kicker</title><link>https://example.com/b</link>
<pubDate>Sat, 17 Oct 2026 12:00:00 +0000</pubDate></item>
<item><title>Josh Allen throws four touchdowns</title><link>https://example.com/c</link>
<pubDate>Sun, 18 Oct 2026 23:00:00 GMT</pubDate></item>
<item><title>Allen and the Bills roll</title><link>https://example.com/d</link>
<pubDate>Sat, 17 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>No link</title></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Kelce questionable for Sunday</title><link href="https://example.com/e"/>
<updated>2026-10-17T15:00:00Z</updated><summary>Travis Kelce (knee)</summary></entry></feed>"""

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
KICK = datetime(2026, 10, 18, 20, 25, tzinfo=timezone.utc)


def _cache():
    c = nn.empty()
    nn.merge(c, nn.parse_feed(RSS, "Test RSS"), NOW)
    nn.merge(c, nn.parse_feed(ATOM, "Test Atom"), NOW)
    return c


def test_both_feed_formats_parse_and_markup_is_stripped():
    items = nn.parse_feed(RSS, "Test RSS")
    assert len(items) == 4                       # the item with no link is dropped
    assert items[0]["summary"] == "Mahomes was a limited participant."
    assert items[0]["published"] == datetime(2026, 10, 16, 18, 0, tzinfo=timezone.utc)
    atom = nn.parse_feed(ATOM, "Test Atom")
    assert atom[0]["link"] == "https://example.com/e" and atom[0]["published"].day == 17


def test_items_are_kept_once_with_when_they_were_gathered():
    c = _cache()
    assert nn.merge(c, nn.parse_feed(RSS, "Test RSS"), NOW) == 0
    assert c["items"]["https://example.com/a"]["gathered"] == "2026-10-18T12:00:00Z"


def test_a_game_gets_its_teams_and_players_news_from_before_kickoff():
    items = nn.for_game(_cache(), ("KC", "LV"), {"Patrick Mahomes", "Travis Kelce"}, KICK)
    titles = [i["title"] for i in items]
    # Player items first, newest first; then team items.
    assert titles == ["Kelce questionable for Sunday", "Chiefs' Patrick Mahomes limited in practice with ankle",
                      "Raiders sign a kicker"]
    assert items[1]["players"] == ["Patrick Mahomes"] and items[1]["teams"] == ["KC"]


def test_published_after_kickoff_is_never_pre_game_news():
    items = nn.for_game(_cache(), ("BUF", "MIA"), {"Josh Allen"}, KICK)
    assert [i["link"] for i in items] == ["https://example.com/d"]   # c is after kickoff


def test_a_surname_alone_is_not_a_player():
    items = nn.for_game(_cache(), ("NYJ", "NE"), {"Josh Allen"}, KICK)
    assert items == []                           # "Allen and the Bills" names neither team
