"""Full professional records from Sherdog, for fighters Wikipedia never covered.

The honest world model had no skill on the Contender Series (57.8%, log
loss 0.688 against a coin's 0.693): most prospects' regional careers are on
no Wikipedia results page, and the one source that had them - fighters' own
articles - exists because they later succeeded, which leaked the result.
Sherdog lists every professional bout of every fighter, whoever they became,
and its robots.txt allows all agents. Each fighter has a numeric ID, so two
men called Bruno Silva are two records, not one.

Parsed from pages the probe saved (tests/fixtures/sherdog), not from
memory of the markup:

    <div class="module fight_history"> ... FIGHT HISTORY - PRO
    <tr><td><span class="final_result win">win</span></td>
        <td><a href="/fighter/Iwo-Baraniewski-381634">Iwo Baraniewski</a></td>
        <td><a href="/events/...">UFC 331 - Van vs. Pantoja 2</a><br />
            <span class="sub_line">Sep / 19 / 2026</span></td>
        <td class="winby"><b>Decision (Split)</b>...</td><td>3</td><td>5:00</td>

    candidates(html) -> the fighters in a search's results table
    pick(html, name) -> the result whose slug IS the name
    record(html)     -> [{date, result, opponent, opponent_id, event, method,
                          round, time}] - professional bouts only
    bio(html)        -> birth date and height (constants, safe at any date)
    confirms(record, date, opponent)
                     -> the record has THIS bout: a search by name finds
                        namesakes, and a record is only taken when it
                        contains the bout the fighter was looked up for
"""

import datetime
import html as htmllib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from name_resolution import norm_name

BASE = "https://www.sherdog.com"
PAUSE = 3.0
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"], 1)}

ROW = re.compile(r"<tr>\s*<td>\s*<span class=\"final_result (\w+)\">.*?</tr>",
                 re.S)
OPPONENT = re.compile(r"href=\"/fighter/([^\"]+?)-(\d+)\"[^>]*>(.*?)</a>", re.S)
EVENT = re.compile(r"href=\"/events/[^\"]+\"[^>]*>(?:<span[^>]*>)?(.*?)"
                   r"(?:</span>)?</a>\s*<br\s*/?>\s*<span class=\"sub_line\">"
                   r"\s*(\w{3})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", re.S)
METHOD = re.compile(r"class=\"winby\"><b>(.*?)</b>", re.S)
CELLS = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def _text(fragment):
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment or "")).strip()


def _pro_section(page):
    """The PRO fight-history module only - amateur bouts are not a record."""
    start = page.find("FIGHT HISTORY - PRO")
    if start < 0:
        start = page.find("module fight_history")
    if start < 0:
        return ""
    end = page.find("FIGHT HISTORY - AMATEUR", start)
    return page[start:end if end > 0 else len(page)]


def record(page):
    bouts = []
    for row in ROW.finditer(_pro_section(page)):
        result = row.group(1).lower()
        chunk = row.group(0)
        opp = OPPONENT.search(chunk)
        event = EVENT.search(chunk)
        method = METHOD.search(chunk)
        cells = [_text(c) for c in CELLS.findall(chunk)]
        if not opp or not event:
            continue
        month = MONTHS.get(event.group(2).lower()[:3])
        if not month:
            continue
        bouts.append({
            "date": f"{int(event.group(4)):04d}-{month:02d}-{int(event.group(3)):02d}",
            "result": {"win": "win", "loss": "loss", "draw": "draw"}.get(
                result, "nc"),
            "opponent": _text(opp.group(3)), "opponent_id": int(opp.group(2)),
            "event": _text(event.group(1)),
            "method": _text(method.group(1)) if method else "",
            "round": cells[-2] if len(cells) >= 2 else "",
            "time": cells[-1] if cells else ""})
    return bouts


RESULTS = re.compile(r"<table class=\"new_table fightfinder_result\">(.*?)</table>",
                     re.S)
LINK = re.compile(r"<a href=\"/fighter/([^\"]+?)-(\d+)\">(.*?)</a>", re.S)
BIRTH = re.compile(r"itemprop=\"birthDate\">\s*(\w{3})\w*\s+(\d{1,2}),\s*(\d{4})")
HEIGHT = re.compile(r"itemprop=\"height\">\s*(\d)'(\d{1,2})\"")


def candidates(search_page):
    """[(slug, id, name)] from the results table only - not the sidebars."""
    table = RESULTS.search(search_page)
    if not table:
        return []
    seen, out = set(), []
    for slug, fid, name in LINK.findall(table.group(1)):
        if int(fid) not in seen:
            seen.add(int(fid))
            out.append((slug, int(fid), _text(name)))
    return out


def bio(page):
    """Birth date and height. Both are constants of the person, so reading
    today's page leaks nothing about any bout. The listed weight and gym
    are NOT read: those are today's, and a gym joined after a win would
    credit the win to the gym."""
    out = {"birth_date": None, "height_cm": None}
    b = BIRTH.search(page)
    if b and MONTHS.get(b.group(1).lower()):
        out["birth_date"] = (f"{int(b.group(3)):04d}-"
                             f"{MONTHS[b.group(1).lower()]:02d}-"
                             f"{int(b.group(2)):02d}")
    h = HEIGHT.search(page)
    if h:
        out["height_cm"] = round((int(h.group(1)) * 12 + int(h.group(2)))
                                 * 2.54, 1)
    return out


def _surname(name):
    parts = norm_name(name).split()
    return parts[-1] if parts else ""


def confirms(record, date, opponent=None, days=3):
    """True if the record holds a bout within `days` of `date` against
    someone with the opponent's surname, or on a Contender Series card
    (a transliterated surname can differ between Wikipedia and Sherdog; a
    namesake on the same Contender Series week cannot happen)."""
    when = _day(date)
    for bout in record:
        if abs((_day(bout["date"]) - when).days) > days:
            continue
        if (opponent is None or _surname(bout["opponent"]) == _surname(opponent)
                or "contender series" in bout["event"].lower()):
            return True
    return False


def _day(text):
    return datetime.date.fromisoformat(str(text)[:10])


def pick(search_page, name):
    """(slug, id) of the search result whose slug IS this name, or None.

    The results page also carries sidebar links to unrelated fighters; the
    probe's first version took the alphabetically first link and saved
    Alonzo Menifield's page three times.
    """
    want = norm_name(name).replace(" ", "")
    for slug, fid in re.findall(r"href=\"/fighter/([^\"]+?)-(\d+)\"", search_page):
        if norm_name(slug.replace("-", " ")).replace(" ", "") == want:
            return slug, int(fid)
    return None


def fetch(url, agent, pause=PAUSE):
    request = urllib.request.Request(url, headers={"User-Agent": agent})
    time.sleep(pause)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def find(name, agent, bouts=(), tries=3):
    """The Sherdog fighter this name means, or None.

    bouts: [(date, opponent)] the fighter is known to have fought. When
    given, only a record containing one of them is accepted, so a namesake
    is never taken for the fighter. Results whose slug is exactly the name
    are tried first, then the rest of the table, `tries` pages at most.
    Returns {id, slug, name, verified, bio, record} or None.
    """
    page = fetch(f"{BASE}/stats/fightfinder?SearchTxt="
                 f"{urllib.parse.quote(name)}", agent)
    want = norm_name(name).replace(" ", "")
    found = candidates(page)
    exact = [c for c in found
             if norm_name(c[0].replace("-", " ")).replace(" ", "") == want]
    ordered = exact + [c for c in found if c not in exact
                       and _surname(c[2]) == _surname(name)]
    if not bouts:
        ordered = exact[:1] if len(exact) == 1 else []
    for slug, fid, shown in ordered[:tries]:
        html = fetch(f"{BASE}/fighter/{slug}-{fid}", agent)
        rec = record(html)
        ok = any(confirms(rec, d, o) for d, o in bouts)
        if ok or not bouts:
            return {"id": fid, "slug": slug, "name": shown, "verified": ok,
                    "bio": bio(html), "record": rec}
    return None
