#!/usr/bin/env python3
"""Daily Albion Online production-bonus check.

Fetches today's daily crafting bonus categories and sends a push notification
when one of them is potions (alchemy). Run once a day, after server downtime
(10:00 UTC), from cron or any scheduler. Standard library only.

Config (environment variables):
  BONUS_API_URL      JSON feed of today's bonuses (default: Albion Free Market's feed)
  SERVER_ID          only check this server's entry (feed "serverId"); unset = any server
  NTFY_TOPIC         ntfy.sh topic to push to (install the ntfy app, subscribe to it)
  DISCORD_WEBHOOK    optional Discord webhook URL, used in addition to / instead of ntfy
  BONUS_KEYWORDS     comma-separated match words (default: potion,alchemy)
  STATE_FILE         where the last-notified date is kept (default: ~/.albion_bonus_state)
"""
import datetime as dt
import json
import os
import sys
import urllib.request

API_URL = os.environ.get("BONUS_API_URL",
                         "https://api.albionfreemarket.com/be/dataclient/festivities/current")
SERVER_ID = os.environ.get("SERVER_ID", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
KEYWORDS = [k.strip().lower() for k in os.environ.get("BONUS_KEYWORDS", "potion,alchemy").split(",") if k.strip()]
PRODUCTION_KIND = 2  # event "kind" the feed uses for daily production bonuses
STATE_FILE = os.path.expanduser(os.environ.get("STATE_FILE", "~/.albion_bonus_state"))
UA = "albion-bonus-notifier/1.0"


def http(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def parse_time(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def active_production_bonuses(feed, now):
    """Return the uniqueNames of production bonuses active right now.

    Feed shape (as read by albionfreemarket.com's own front end):
      {"servers": [{"serverId": ..., "updatedAt": ...,
                    "events": [{"uniqueName": "COMMON_POTION", "kind": 2,
                                "startTime": "...", "endTime": "..."}]}]}
    uniqueName is COMMON_<category> or RARE_<category>.
    """
    names = []
    for server in feed.get("servers") or []:
        if SERVER_ID and str(server.get("serverId")) != SERVER_ID:
            continue
        for ev in server.get("events") or []:
            if ev.get("kind") != PRODUCTION_KIND or not isinstance(ev.get("uniqueName"), str):
                continue
            try:
                if parse_time(ev["startTime"]) <= now < parse_time(ev["endTime"]):
                    names.append(ev["uniqueName"])
            except (KeyError, ValueError):
                continue
    return names


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
        feed = json.loads(http(API_URL, headers={"Origin": "https://albionfreemarket.com"}))
        bonuses = active_production_bonuses(feed, dt.datetime.now(dt.timezone.utc))
    except Exception as e:
        notify("Albion bonus check failed", f"{e}")
        return 1
    if not bonuses:
        notify("Albion bonus check failed", "feed returned no active production bonuses")
        return 1

    hit = [b for b in bonuses if any(k in b.lower() for k in KEYWORDS)]
    print(f"{today}: active={bonuses} potion_match={hit}")
    if hit:
        notify("Potion bonus today",
               f"Daily production bonus: {', '.join(hit)}. Start potion crafting.")
        with open(STATE_FILE, "w") as f:
            f.write(today)
    return 0


if __name__ == "__main__":
    sys.exit(main())
