"""Fighter name resolution.

Split out of predict_card.py so it can be unit tested without running the whole
training pipeline, and reused by the prediction API.

The rule is simple: resolve a name confidently, or say NO DATA. Never silently
substitute a different fighter. The previous implementation ranked fuzzy
candidates alphabetically and took the first, which turned "Leon Shahbazyan"
into "Cameron Saaiman" (a 0.60 match) while "Edmen Shahbazyan" (0.84) was right
there in the candidate list.
"""

import difflib
import json
import re
import unicodedata


def norm_name(s):
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_aliases(path):
    """Load {typed name: canonical name}. Keys starting with _ are comments."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {norm_name(k): v for k, v in raw.items() if not str(k).startswith("_")}


class NameResolver:
    """Resolves typed fighter names against the names present in the dataset.

    Args:
        names: iterable of canonical fighter names as they appear in the data
        appearances: optional {name: fight_count}, used only to break ties
        aliases: optional {normalized typed name: canonical name}
        auto_accept_ratio: fuzzy ratio at/above which a typo is accepted, and
            only when the surname also matches exactly
        suggest_ratio: fuzzy ratio at/above which near-misses are suggested
        strict: when False, fall back to the best-ranked guess with a warning
    """

    def __init__(self, names, appearances=None, aliases=None,
                 auto_accept_ratio=0.90, suggest_ratio=0.60, strict=True):
        self.aliases = dict(aliases or {})
        self.appearances = dict(appearances or {})
        self.auto_accept_ratio = auto_accept_ratio
        self.suggest_ratio = suggest_ratio
        self.strict = strict

        self.names = sorted({str(n) for n in names if n and str(n).strip()})
        self.norm_to_canon = {}
        self.token_index = {}
        for nm in self.names:
            key = norm_name(nm)
            if not key:
                continue
            self.norm_to_canon.setdefault(key, set()).add(nm)
            for tok in set(key.split()):
                self.token_index.setdefault(tok, set()).add(nm)
        # Built once. The old code rebuilt this on every fuzzy lookup.
        self._norm_list = [norm_name(n) for n in self.names]

    # -- internals ---------------------------------------------------------

    def _rank(self, candidates, query):
        """Best match first: quality, then fight count, then name."""
        scored = [
            (difflib.SequenceMatcher(None, query, norm_name(c)).ratio(),
             self.appearances.get(c, 0), c)
            for c in candidates
        ]
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return scored

    def _fuzzy(self, query, n=5):
        """Near-miss names, best first. Never used to auto-pick a fighter."""
        out, seen = [], set()
        for m in difflib.get_close_matches(query, self._norm_list, n=n,
                                           cutoff=self.suggest_ratio):
            ratio = difflib.SequenceMatcher(None, query, m).ratio()
            for canon in self.norm_to_canon.get(m, ()):
                if canon not in seen:
                    seen.add(canon)
                    out.append((ratio, canon))
        out.sort(key=lambda t: (-t[0], t[1]))
        return out

    @staticmethod
    def _result(status, name=None, how=None, note=None, suggestions=None):
        return {'status': status, 'name': name, 'how': how, 'note': note,
                'suggestions': suggestions or []}

    # -- public API --------------------------------------------------------

    def resolve(self, user_input):
        """Resolve a typed name.

        Returns a dict with status (OK / AMBIGUOUS / NOT_FOUND), name, how,
        note and suggestions.
        """
        raw = "" if user_input is None else str(user_input).strip()
        q = norm_name(raw)
        if not q:
            return self._result('NOT_FOUND', note='Empty fighter name.')

        if q in self.aliases:
            canon = self.aliases[q]
            return self._result('OK', canon, 'alias', f"Alias: '{raw}' -> '{canon}'")

        if q in self.norm_to_canon:
            cands = self.norm_to_canon[q]
            if len(cands) == 1:
                return self._result('OK', next(iter(cands)), 'exact')
            best = self._rank(cands, q)[0][2]
            return self._result('OK', best, 'exact',
                                f"'{raw}' matches {len(cands)} spellings; using '{best}'.")

        toks = q.split()

        if len(toks) >= 2:
            pool = None
            for t in toks:
                hits = self.token_index.get(t, set())
                pool = hits if pool is None else (pool & hits)
            cands = sorted(pool or set())
            if len(cands) == 1:
                return self._result('OK', cands[0], 'tokens')
            if len(cands) > 1:
                ranked = self._rank(cands, q)
                return self._result('AMBIGUOUS', None, 'tokens',
                                    f"'{raw}' matches {len(cands)} fighters.",
                                    [(r, n) for r, _, n in ranked[:5]])

        if len(toks) == 1:
            cands = sorted(self.token_index.get(toks[0], set()))
            if len(cands) == 1:
                return self._result('OK', cands[0], 'surname',
                                    f"'{raw}' -> '{cands[0]}' (only fighter with that name).")
            if len(cands) > 1:
                ranked = self._rank(cands, q)
                return self._result('AMBIGUOUS', None, 'surname',
                                    f"'{raw}' matches {len(cands)} fighters - be more specific.",
                                    [(r, n) for r, _, n in ranked[:5]])

        # Typo: near-identical spelling AND the same surname. Both required,
        # so a wrong first name can never drag in a different fighter.
        fuzzy = self._fuzzy(q)
        if fuzzy:
            top_ratio, top_name = fuzzy[0]
            if (top_ratio >= self.auto_accept_ratio
                    and norm_name(top_name).split()[-1:] == toks[-1:]):
                return self._result('OK', top_name, 'typo',
                                    f"Read '{raw}' as '{top_name}' (typo, {top_ratio:.0%} match).")

        return self._result('NOT_FOUND', note=f"No data for '{raw}'.",
                            suggestions=fuzzy[:5])

    def search(self, query, limit=10):
        """Ranked fighter suggestions for autocomplete. Never refuses."""
        q = norm_name(query)
        if not q:
            return []
        prefix = [n for n in self.names if norm_name(n).startswith(q)]
        substr = [n for n in self.names if q in norm_name(n) and n not in prefix]
        ranked = [n for _, _, n in self._rank(prefix + substr, q)]
        for _, name in self._fuzzy(q, n=limit):
            if name not in ranked:
                ranked.append(name)
        return ranked[:limit]


def format_failure(user_input, res):
    """One-line explanation plus suggestions, for printing."""
    msg = f"NO DATA: {res['note']}"
    if res['suggestions']:
        opts = ", ".join(f"{n} ({r:.0%})" for r, n in res['suggestions'])
        msg += f"  Did you mean: {opts}?"
    return msg
