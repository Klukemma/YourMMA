#!/usr/bin/env python3
"""Daily Albion Online production-bonus check.

Fetches today's daily crafting bonus categories and sends a push notification
when one of them is potions (alchemy). Run once a day, after server downtime
(10:00 UTC), from cron or any scheduler. Standard library only.

Config (environment variables):
  BONUS_SOURCE_URL   page listing today's bonuses (default: albiondatabase.com/events)
  NTFY_TOPIC         ntfy.sh topic to push to (install the ntfy app, subscribe to it)
  DISCORD_WEBHOOK    optional Discord webhook URL, used in addition to / instead of ntfy
  BONUS_KEYWORDS     comma-separated match words (default: potion,alchemy)
  STATE_FILE         where the last-notified date is kept (default: ~/.albion_bonus_state)
"""
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.request

SOURCE_URL = os.environ.get("BONUS_SOURCE_URL", "https://albiondatabase.com/events")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
KEYWORDS = [k.strip().lower() for k in os.environ.get("BONUS_KEYWORDS", "potion,alchemy").split(",") if k.strip()]
STATE_FILE = os.path.expanduser(os.environ.get("STATE_FILE", "~/.albion_bonus_state"))
UA = "albion-bonus-notifier/1.0"


def http(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def page_text(raw):
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


def todays_bonus_text():
    # UNVERIFIED: the source page's layout hasn't been inspected yet. This takes
    # the text following the first "daily bonus" heading. Replace it with a
    # proper selector / JSON field once the page structure is known.
    text = page_text(http(SOURCE_URL))
    m = re.search(r"(?i)daily\s+(crafting\s+|production\s+)?bonus", text)
    if not m:
        raise RuntimeError(f"no daily-bonus section found on {SOURCE_URL}")
    return text[m.start():m.start() + 400]


def notify(title, body):
    sent = False
    if NTFY_TOPIC:
        http(f"https://ntfy.sh/{NTFY_TOPIC}", body.encode(), {"Title": title, "Tags": "test_tube"})
        sent = True
    if DISCORD_WEBHOOK:
        http(DISCORD_WEBHOOK, json.dumps({"content": f"**{title}**\n{body}"}).encode(),
             {"Content-Type": "application/json"})
        sent = True
    if not sent:
        print(f"[no notifier configured] {title}: {body}")


def main():
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    try:
        if open(STATE_FILE).read().strip() == today:
            return 0  # already notified today
    except OSError:
        pass

    try:
        section = todays_bonus_text()
    except Exception as e:
        notify("Albion bonus check failed", f"{e}")
        return 1

    hit = [k for k in KEYWORDS if k in section.lower()]
    print(f"{today}: {section[:200]!r} -> match={hit}")
    if hit:
        notify("Potion bonus today", "Daily production bonus is on potions. Start potion crafting.")
        with open(STATE_FILE, "w") as f:
            f.write(today)
    return 0


if __name__ == "__main__":
    sys.exit(main())
