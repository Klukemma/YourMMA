---
name: scout
description: Research fighter-specific news for an upcoming card - injuries, late replacements, camp changes, weight trouble - and record it as sourced, point-in-time observations. Use when asked to scout a card, gather intel on fighters, or find news the model cannot see.
---

# Scouting a card

The model reads records. It does not know that somebody tore a meniscus in
camp, took the fight on nine days' notice, or moved to a gym with a
world-class wrestling coach. You go and find out.

## What you are actually producing

Sourced observations, filed under a closed list of kinds, each one knowable
**before the fight**. Not a summary. Not your impression. A claim a named
outlet made, with the URL, that a reader could check.

**Nothing you record moves a probability.** `engine/experiments/short_notice_weight.py`
measured the one soft factor we can label at scale - 298 late replacements
harvested from Wikipedia - and got −3.5% against the model and −4.4% against
the blended prediction, both intervals spanning zero, with roughly 1,800
labels needed to resolve an effect that size. So no weight is fitted for any
kind, and `applies` stays false. What you gather is SHOWN beside the number so
a reader knows what the model cannot see. That is the job. It is a real one.

## The procedure

1. **Get the brief.**
   ```
   python engine/intel_search.py --brief
   ```
   It prints every fighter on the current card and the exact queries to run.
   Run them with WebSearch. Direct page fetches are blocked in this
   environment; the search results carry enough.

2. **Read the results, not your memory.** For each hit, ask: does a named
   outlet actually say this, and does it predate the fight? If you cannot
   point at a URL, there is no observation.

3. **Write findings to a JSON list:**
   ```json
   [{"fighter": "Brady Hiestand", "event_date": "2026-09-26",
     "kind": "injury", "confidence": "reported",
     "note": "knee trouble reported in the final week of camp",
     "source": "https://www.sherdog.com/news/...",
     "gathered": "2026-09-24T10:00:00Z"}]
   ```
   `kind` is one of: injury, short_notice, hard_weight_cut, missed_weight,
   new_camp, layoff_return, personal, opponent_switch, motivation,
   confidence, other.
   `motivation` is what the fighter said about WANTING it (retirement talk,
   last fight on contract, "business fight", a stated title chase).
   `confidence` is what they said about WINNING it ("I finish him in one").
   They are different things - a confident champion is not a hungry one -
   and are never filed under each other.
   `confidence` is one of: reported, confirmed, rumoured.
   An `opponent_switch` also needs `replaced_opponent` - the fighter
   originally booked - because the interesting half of a late replacement is
   that a camp prepared for a wrestler and walks in against a counter-striker.

4. **Ingest:**
   ```
   python engine/intel_search.py --ingest findings.json
   ```
   It validates every record and prints a reason for each rejection.

## The rules that matter

**Point in time, or it is worthless.** `gathered` is when YOU found it, and
anything gathered after the first bell is rejected. This is not bureaucracy:
a search today about last Saturday's fight returns coverage written
afterwards, phrased as though it were known beforehand. Two of this project's
worst bugs were a model reading its own answer. A note saying "he looked slow
in camp", written the morning after a knockout, would be a third.

**Upcoming cards only.** `gathered` is when YOU found it, not when the story
ran, so this cannot backfill history: a piece published the day before a 2019
fight is rejected today along with everything else. That is deliberate. A
publication date is exactly what a language model supplies confidently and
wrongly, and page fetches are blocked here so it cannot be verified. Labels
for fitting weights come from `engine/fetch_fight_changes.py`, which reads
dated event articles instead of trusting a researcher.

**Sourced or it did not happen.** Every observation needs a URL. A claim
nobody can check cannot be told apart from one you invented, which is the
specific failure mode of having a language model fill this file.

**Record the claim, not your inference.** "Coach told MMA Junkie he cut 22lb
in the final week" is an observation. "Looks like a hard cut" is not.

**A gap is a finding.** If seven queries about a fighter turn up nothing, that
fighter has no reported problems, which is worth knowing. Do not manufacture
something to fill the row.

**Rumours are labelled `rumoured`, never upgraded.** A forum post is not ESPN.

## Where it shows up

`fight_intel.for_fight()` pulls the observations for a bout; the card export
carries them so the app can show them under the pick. `context_from()` is the
path from a gathered fact to a moved probability and it returns nothing today,
by design - it exists so that the day a weight IS fitted, that path is one
place with a test on it rather than something improvised on a fight night.
