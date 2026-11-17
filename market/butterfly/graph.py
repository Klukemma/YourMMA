"""Dependency map: what moves what, and the ripple an event sends through it.

An edge  src -> dst  (sign s, weight w) reads: "when src rises, dst tends to
move in direction s, with strength w in (0, 1]". `lag` says roughly how long
it takes to show up (days / weeks / quarters).

A ripple multiplies signs and weights along every path out of the starting
node, so a third-hop effect is weak unless every link is strong. That's
intended: long chains are where reasoning goes wrong most easily.

Every edge records where it came from. Edges with verified=false are seeds
from general knowledge and must be confirmed against a filing (the `source`
field then becomes the filing URL) before a prediction leans on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config

NODE_TYPES = {"company", "etf", "commodity", "macro", "currency", "route", "sector", "event", "factor"}


@dataclass
class Graph:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = config.GRAPH_PATH) -> "Graph":
        raw = json.loads(Path(path).read_text())
        g = cls(raw.get("nodes", {}), raw.get("edges", []))
        g.validate()
        return g

    def save(self, path: Path = config.GRAPH_PATH) -> None:
        self.validate()
        Path(path).write_text(json.dumps({"nodes": self.nodes, "edges": self.edges}, indent=1, ensure_ascii=False) + "\n")

    def validate(self) -> None:
        for nid, n in self.nodes.items():
            if n.get("type") not in NODE_TYPES:
                raise ValueError(f"node {nid}: type must be one of {sorted(NODE_TYPES)}")
        seen = set()
        for e in self.edges:
            for end in ("src", "dst"):
                if e[end] not in self.nodes:
                    raise ValueError(f"edge {e['src']}->{e['dst']}: unknown node {e[end]}")
            if e["sign"] not in (1, -1):
                raise ValueError(f"edge {e['src']}->{e['dst']}: sign must be +1 or -1")
            if not 0 < e["weight"] <= 1:
                raise ValueError(f"edge {e['src']}->{e['dst']}: weight must be in (0, 1]")
            if not e.get("source"):
                raise ValueError(f"edge {e['src']}->{e['dst']}: every edge needs a source")
            key = (e["src"], e["dst"])
            if key in seen:
                raise ValueError(f"duplicate edge {key}")
            seen.add(key)

    def add_edge(self, src, dst, sign, weight, *, relation, source, lag="weeks", verified=False, note=""):
        self.edges.append({"src": src, "dst": dst, "sign": sign, "weight": weight, "relation": relation,
                           "lag": lag, "source": source, "verified": verified, "note": note})
        self.validate()

    def out_edges(self, nid: str) -> list[dict]:
        return [e for e in self.edges if e["src"] == nid]

    def ripple(self, start: str, shock: int = 1, max_hops: int = 3, min_strength: float = 0.05,
               verified_only: bool = False) -> list[dict]:
        """Every node reachable from `start`, with its net signed impact and
        the paths that carry it, strongest first. Paths never revisit a node."""
        if start not in self.nodes:
            raise KeyError(start)
        hits: dict[str, list[dict]] = {}
        stack = [(start, shock * 1.0, [start], [], True)]
        while stack:
            node, strength, path, edges, all_verified = stack.pop()
            if len(path) - 1 >= max_hops:
                continue
            for e in self.out_edges(node):
                if e["dst"] in path or (verified_only and not e.get("verified")):
                    continue
                s = strength * e["sign"] * e["weight"]
                if abs(s) < min_strength:
                    continue
                p = path + [e["dst"]]
                ver = all_verified and bool(e.get("verified"))
                hits.setdefault(e["dst"], []).append({"impact": s, "path": p, "lags": edges + [e["lag"]], "verified": ver})
                stack.append((e["dst"], s, p, edges + [e["lag"]], ver))
        out = []
        for nid, paths in hits.items():
            paths.sort(key=lambda x: -abs(x["impact"]))
            net = sum(x["impact"] for x in paths)
            out.append({
                "node": nid,
                "type": self.nodes[nid]["type"],
                "ticker": self.nodes[nid].get("ticker"),
                "net_impact": max(-1.0, min(1.0, net)),
                "conflicted": any(x["impact"] > 0 for x in paths) and any(x["impact"] < 0 for x in paths),
                "all_verified": all(x["verified"] for x in paths),
                "paths": paths,
            })
        out.sort(key=lambda x: -abs(x["net_impact"]))
        return out


def describe(g: Graph, r: dict) -> str:
    best = r["paths"][0]
    arrow = " -> ".join(best["path"])
    flags = []
    if r["conflicted"]:
        flags.append("paths disagree")
    if not r["all_verified"]:
        flags.append("unverified links")
    sign = "+" if r["net_impact"] > 0 else "-"
    return f"{sign}{abs(r['net_impact']):.2f}  {r['node']:<26} via {arrow}" + (f"  [{', '.join(flags)}]" if flags else "")
