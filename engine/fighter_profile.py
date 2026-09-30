"""A fighter's camp, coaches and credentials, from their Wikipedia infobox.

"Camp power" - who a fighter trains with and under - is something the model
has never seen. The infobox states it in a form that can be read without
guessing:

    | team    = [[American Top Team]] (2015–present)<br />
                [[Team Alpha Male]] (2010–2015)
    | trainer = [[Mike Brown (fighter)|Mike Brown]]
    | rank    = [[Black belt]] in [[Brazilian jiu-jitsu]] under [[Ricardo Liborio]]

THE SAME RULE AS EVERY OTHER HISTORY IN THIS PROJECT: an affiliation may
only inform a fight that happened while it was known. A team listed with its
years ("2010–2015") is placeable in time. A team listed with no years is what
the article says TODAY - true of the fighter now, and of nobody's past - so
it is kept with dated=False and must not be used for a historical fight.
The canonical name of a gym is its link target, when the infobox links it:
"[[American Kickboxing Academy|AKA]]" is American Kickboxing Academy, and
the alias problem mostly disappears.

Camp STRENGTH is not read from anywhere. It is measured, as of each fight,
from the archive's results for the gym's fighters before that fight (see
camp_strength), so a gym's reputation cannot leak in from a page written
after its fighters won.
"""

import re

LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")
YEARS = re.compile(
    r"\(\s*(\d{4})\s*(?:[-–—]|to)?\s*(present|current|\d{4})?\s*\)", re.I)


def _strip_refs(text):
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text or "", flags=re.S)
    text = re.sub(r"<ref[^>]*/>", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return text


def infobox(wikitext):
    """{field: raw value} of the page's first Infobox template.

    Walks braces rather than matching a pattern, because a field's value is
    full of nested templates ({{flagicon}}, {{convert}}, {{birth date}}) and
    any regex that stops at "}}" stops at the wrong one.
    """
    text = wikitext or ""
    start = re.search(r"\{\{\s*Infobox", text, re.I)
    if not start:
        return {}
    depth, i, end = 0, start.start(), None
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
                end = i
                break
            continue
        i += 1
    body = text[start.end():(end or len(text)) - 2]
    fields, depth, current, buf = {}, 0, None, []

    def flush():
        if current is not None:
            fields[current] = "".join(buf).strip()

    j = 0
    while j < len(body):
        two = body[j:j + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            j += 2
            continue
        if two in ("}}", "]]"):
            depth -= 1
            buf.append(two)
            j += 2
            continue
        ch = body[j]
        if ch == "|" and depth == 0:
            flush()
            rest = body[j + 1:]
            m = re.match(r"\s*([\w ]+?)\s*=", rest)
            if m:
                current = m.group(1).strip().lower().replace(" ", "_")
                buf = []
                j += 1 + m.end()
                continue
            current, buf = None, []
            j += 1
            continue
        buf.append(ch)
        j += 1
    flush()
    return fields


def _plain(text):
    """Visible text: links to their labels, templates and markup gone."""
    text = LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"'''?|<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,;")


def _entries(value):
    """A field split into its listed items: <br>, bullets, list templates."""
    value = _strip_refs(value)
    value = re.sub(r"\{\{\s*(?:plainlist|ubl|unbulleted list|flatlist)\s*\|",
                   "\n", value, flags=re.I)
    parts = re.split(r"<br\s*/?>|\n\s*\*|\n|\|(?![^\[]*\]\])", value)
    return [p.strip(" *}{") for p in parts if p and p.strip(" *}{")]


def affiliations(value):
    """[{"team", "canonical", "start", "end", "dated"}] from a team field.

    "(2015–present)" gives start 2015 and end None; "(2010–2015)" gives both;
    "(2012)" a single year; no parentheses, dated=False: it is the current
    listing and says nothing about when it began.
    """
    out = []
    for entry in _entries(value):
        links = LINK.findall(entry)
        canonical = links[0][0].strip() if links else None
        years = YEARS.search(entry)
        name = _plain(YEARS.sub(" ", entry))
        if not name or len(name) < 2:
            continue
        start = end = None
        if years:
            start = int(years.group(1))
            tail = (years.group(2) or "").lower()
            if tail.isdigit():
                end = int(tail)
            elif not tail and not re.search(r"[-–—]", years.group(0)):
                end = start           # a single year: that year only
        out.append({"team": name, "canonical": canonical or name,
                    "start": start, "end": end, "dated": start is not None})
    return out


def people(value):
    """Names in a trainer/coach field, by link target where linked."""
    out = []
    for entry in _entries(value):
        links = LINK.findall(entry)
        if links:
            for target, label in links:
                out.append({"name": (label or target).strip(),
                            "canonical": target.strip()})
        else:
            name = _plain(entry)
            if name:
                out.append({"name": name, "canonical": name})
    return out


BELT = re.compile(r"(black|brown|purple|blue|red|coral)\s+belt", re.I)


def ranks(value):
    """[{"belt", "art", "under", "text"}] from a rank field."""
    out = []
    for entry in _entries(value):
        text = _plain(entry)
        belt = BELT.search(text)
        art = re.search(r"\bin\s+([A-Z][\w\- ]+?)(?:\s+under\b|$|,)", text)
        under = re.search(r"\bunder\s+(.+)$", text)
        if belt or art:
            out.append({"belt": belt.group(1).lower() if belt else None,
                        "art": art.group(1).strip() if art else None,
                        "under": under.group(1).strip() if under else None,
                        "text": text})
    return out


def profile(wikitext):
    """The camp-and-credentials part of a fighter's infobox."""
    box = infobox(wikitext)
    team = box.get("team") or box.get("teams") or ""
    trainer = box.get("trainer") or box.get("trainers") or \
        box.get("coach") or ""
    return {
        "affiliations": affiliations(team),
        "trainers": people(trainer),
        "ranks": ranks(box.get("rank", "")),
        "fighting_out_of": _plain(_strip_refs(box.get("fightingoutof", "")
                                              or box.get("residence", ""))),
        "style": _plain(_strip_refs(box.get("style", ""))),
        "years_active": _plain(_strip_refs(box.get("years_active", "")
                                           or box.get("yearsactive", ""))),
    }
