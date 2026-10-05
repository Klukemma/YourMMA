"""Bouts from every promotion Wikipedia covers, not just the UFC.

The model rates fighters from UFC bouts alone, so a Contender Series
debutant - a career spent in LFA, Cage Warriors, Brave, Oktagon - is a blank
to it. Wikipedia records those careers in two forms, both parsed here:

  EVENT RESULTS, on event pages and "YEAR in <promotion>" pages:
      ==LFA 174: Jones vs. Gennrich==
      {{Infobox MMA event | date= January 12, 2024 ...}}
      {{MMAevent card|Main Card (UFC Fight Pass)}}
      {{MMAevent bout|Lightweight|Kegan Gennrich|def.|JaCobi Jones|
                      Submission (arm-triangle choke)|5|2:08|...}}
      {{MMAevent card|Amateur bouts|header=no}}     <- excluded
  FIGHTER RECORDS, the "MMA record start" table on a fighter's article:
      |{{no2}}Loss |22–10 |[[Mike Malott]] |TKO (punches)
      |[[UFC Fight Night: Burns vs. Malott]] |{{dts|2026|April|18}} |3 |2:08

Both were written against pages saved from Wikipedia by probe_world.py
(engine/tests/fixtures/wiki), not from memory of the markup.

A bout is (date, winner, loser, result, method, round, time, weight,
event). "result" is win, draw or nc - a draw or no contest has no winner,
and treating its first-named fighter as one would invent a result.
"""

import re

from fighter_profile import LINK, infobox

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9


def split_params(inner):
    """Top-level "|" split of a template's inside, respecting [[..]]/{{..}}."""
    out, depth, buf, i = [], 0, [], 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
            continue
        if two in ("}}", "]]"):
            depth -= 1
            buf.append(two)
            i += 2
            continue
        if inner[i] == "|" and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(inner[i])
        i += 1
    out.append("".join(buf))
    return out


def templates(text, name):
    """Every {{name ...}} in text, as (start, inner text), brace-balanced."""
    pattern = re.compile(r"\{\{\s*" + name + r"\b", re.I)
    for hit in pattern.finditer(text):
        depth, i = 0, hit.start()
        while i < len(text) - 1:
            pair = text[i:i + 2]
            if pair == "{{":
                depth += 1
                i += 2
                continue
            if pair == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    yield hit.start(), text[hit.start() + 2:i - 2]
                    break
                continue
            i += 1


def plain(cell):
    """Visible text of a cell: link labels, no templates, refs or markup."""
    cell = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", " ", cell or "",
                  flags=re.S)
    cell = LINK.sub(lambda m: m.group(2) or m.group(1), cell)
    cell = re.sub(r"\{\{[^{}]*\}\}", " ", cell)
    cell = re.sub(r"'''?|<[^>]+>|&nbsp;", " ", cell)
    return re.sub(r"\s+", " ", cell).strip(" |")


def parse_date(text):
    """ISO date from {{dts|..}}, {{Start date|..}} or prose, else None."""
    text = text or ""
    t = re.search(r"\{\{\s*(?:dts|[Ss]tart date|[Dd]ate)\s*\|\s*"
                  r"(?:format=\w+\s*\|\s*)?(\d{4})\s*\|\s*(\w+)\s*\|\s*"
                  r"(\d{1,2})", text)
    if t:
        month = int(t.group(2)) if t.group(2).isdigit() else \
            MONTHS.get(t.group(2).lower()[:4].rstrip("."), None) or \
            MONTHS.get(t.group(2).lower()[:3])
        if month:
            return f"{int(t.group(1)):04d}-{month:02d}-{int(t.group(3)):02d}"
    t = re.search(r"\{\{\s*dts\s*\|\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})",
                  text)
    if t:                                  # {{dts|2012.04.21}}
        return f"{int(t.group(1)):04d}-{int(t.group(2)):02d}-{int(t.group(3)):02d}"
    text = plain(text)
    m = re.search(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2}),?\s+(\d{4})\b", text)
    if m and m.group(1).lower()[:3] in MONTHS:
        return (f"{int(m.group(3)):04d}-{MONTHS[m.group(1).lower()[:3]]:02d}"
                f"-{int(m.group(2)):02d}")
    m = re.search(r"\b(\d{1,2})\s+([A-Z][a-z]{2,8})\s+(\d{4})\b", text)
    if m and m.group(2).lower()[:3] in MONTHS:
        return (f"{int(m.group(3)):04d}-{MONTHS[m.group(2).lower()[:3]]:02d}"
                f"-{int(m.group(1)):02d}")
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return m.group(0)
    return None


def _result(word, method):
    word = (word or "").strip().lower().rstrip(".")
    low = (method or "").lower()
    if word == "def":
        return "win"
    if "no contest" in low or low.startswith("nc"):
        return "nc"
    if "draw" in low:
        return "draw"
    return None


HEADING = re.compile(r"^(={2,4})\s*(.*?)\s*\1\s*$", re.M)
SKIP = re.compile(r"background|results|bonus|payout|weights|aftermath|"
                  r"see also|references|external|reported|notes|format|"
                  r"standings|tournament|title fights|list of events",
                  re.I)


def event_bouts(wikitext, page=""):
    """Every professional bout on an event or year page.

    The event a bout belongs to is the nearest event heading above it (on a
    year page) or the page itself (on an event page); its date is that
    event's infobox date. Bouts under an "Amateur" card are dropped.
    """
    text = wikitext or ""
    box = infobox(text)
    page_event = plain(box.get("name", "")) or page
    page_date = parse_date(box.get("date", ""))
    page_promotion = plain(box.get("promotion", ""))

    # Event sections on a year page: a heading followed by its own
    # "Infobox MMA event"; results headings inside it do not start one.
    anchors = []
    for h in HEADING.finditer(text):
        title = plain(h.group(2))
        if SKIP.search(title) and len(title) < 40:
            continue
        window = text[h.end():h.end() + 3000]
        box_here = re.match(r"\s*\{\{\s*Infobox MMA event", window, re.I)
        date = None
        if box_here:
            date = parse_date(infobox(window).get("date", ""))
        anchors.append((h.start(), title, date))

    cards = [(s, plain(split_params(inner)[1]) if len(split_params(inner)) > 1
              else "") for s, inner in templates(text, "MMAevent card")]
    bouts = []
    for start, inner in templates(text, "MMAevent bout"):
        params = [p.strip() for p in split_params(inner)[1:]]
        if len(params) < 4:
            continue
        params += [""] * (8 - len(params))
        weight, first, word, second, method, rnd, time, notes = params[:8]
        result = _result(plain(word), plain(method))
        if result is None:
            continue
        card = ""
        for s, label in cards:
            if s < start:
                card = label
        if re.search(r"amateur", card, re.I):
            continue
        event, date = page_event, page_date
        for s, title, d in anchors:
            if s < start and d:
                event, date = title, d
        bouts.append({
            "date": date, "event": event, "promotion": page_promotion,
            "weight": plain(weight),
            # For a draw or no contest these are just the two names, in
            # page order; "result" says there was no winner.
            "winner": plain(first), "loser": plain(second),
            "result": result, "method": plain(method),
            "round": plain(rnd), "time": plain(time),
            "card": card, "source": page})
    return bouts


def _cell(part):
    pieces = split_params(part)
    if len(pieces) > 1 and re.match(r"\s*[a-z-]+\s*=", pieces[0]):
        return "|".join(pieces[1:]).strip()
    return part.strip()


RECORD = re.compile(
    r"\{\{\s*MMA record start\s*\}\}(.*?)(?:\{\{\s*[Ee]nd\s*\}\}|^\|\})",
    re.S | re.M)


def record_bouts(wikitext, subject):
    """Every professional bout in a fighter's MMA record table.

    Cells: result, record, opponent, method, event, date, round, time,
    location, notes. The result is the SUBJECT's, so a Loss makes the
    opponent the winner.
    """
    bouts = []
    for table in RECORD.findall(wikitext or ""):
        for raw in re.split(r"^\|-.*$", table, flags=re.M)[1:]:
            cells = []
            for line in raw.split("\n"):
                line = line.strip()
                if line.startswith("|") and not line.startswith("|}"):
                    # "||" separates cells on one line; a single "|" inside
                    # a cell separates its attributes (align=center) from
                    # its content. Splitting on every "|" made the
                    # attribute a cell of its own and shifted every column
                    # after it - the date landed in the round.
                    cells += [_cell(part) for part in line[1:].split("||")]
            if len(cells) < 6:
                continue
            outcome = plain(cells[0]).lower()
            opponent = plain(cells[2])
            method = plain(cells[3])
            if not opponent:
                continue
            if outcome.startswith("win"):
                winner, loser, result = subject, opponent, "win"
            elif outcome.startswith("loss"):
                winner, loser, result = opponent, subject, "win"
            elif outcome.startswith("draw"):
                winner, loser, result = subject, opponent, "draw"
            elif outcome.startswith("nc") or "no contest" in outcome:
                winner, loser, result = subject, opponent, "nc"
            else:
                continue
            bouts.append({
                "date": parse_date(cells[5]), "event": plain(cells[4]),
                "promotion": "", "weight": "", "winner": winner,
                "loser": loser, "result": result, "method": method,
                "round": plain(cells[6]) if len(cells) > 6 else "",
                "time": plain(cells[7]) if len(cells) > 7 else "",
                "card": "", "source": f"record:{subject}"})
    return bouts
