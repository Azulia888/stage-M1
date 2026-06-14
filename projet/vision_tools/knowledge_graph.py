"""
knowledge_graph.py — KnowledgeGraphTool: builds a persistent co-occurrence graph
from NER output, with entity normalization, fuzzy deduplication, and optional
Wikidata QID enrichment.

Dependencies: networkx, rapidfuzz (pip install networkx rapidfuzz)
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import networkx as nx
from rapidfuzz import fuzz, process

from data_manager import DataManager
from vision_tools.base import VisionTool, _make_tool_json


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Fuzzy match threshold (0–100) using token_set_ratio.
# token_set_ratio handles subset matches ("Macron" ↔ "Emmanuel Macron" → 100)
# but is permissive, so 90 is used to avoid false positives like
# "macron" ↔ "macaroni" (86). Abbreviated first names ("E. Macron") will
# not auto-merge — this is intentional to avoid incorrect merges.
_FUZZY_THRESHOLD = 90

# Wikidata SPARQL endpoint
_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Wikidata entity-type → P31 (instance of) / search type mapping.
# Used to bias the label search toward the right kind of entity.
_WIKIDATA_TYPE_HINT: dict[str, str] = {
    "PERSON":       "wd:Q5",            # human
    "ORGANIZATION": "wd:Q43229",        # organization
    "LOCATION":     "wd:Q618123",       # geographical object
    "EVENT":        "wd:Q1656682",      # event
    "WORK_OF_ART":  "wd:Q838948",       # work of art
    "LAW":          "wd:Q820655",       # statute
    "PRODUCT":      "wd:Q2424752",      # product
}

# Per-entity Wikidata request timeout (seconds). Keep short so it doesn't
# block the pipeline when the endpoint is slow.
_WIKIDATA_TIMEOUT = 8

# Maximum entities to enrich per run (avoid hammering the public endpoint).
_WIKIDATA_MAX_ENTITIES = 30


# ---------------------------------------------------------------------------
# 1. String normalization
# ---------------------------------------------------------------------------

_TITLES = re.compile(
    r"\b(mr|mrs|ms|dr|prof|gen|col|lt|sgt|cpl|pte|sir|dame|lord|lady"
    r"|président|ministre|général|colonel|docteur)\b\.?",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s{2,}")


def _normalize(name: str) -> str:
    """Lower-case, strip honorifics, collapse whitespace."""
    name = _TITLES.sub("", name)
    name = _MULTI_SPACE.sub(" ", name).strip().lower()
    return name


# ---------------------------------------------------------------------------
# 2. Fuzzy entity resolution
# ---------------------------------------------------------------------------

def _resolve_node_id(
    G: nx.Graph,
    entity_type: str,
    name: str,
    threshold: int = _FUZZY_THRESHOLD,
) -> str:
    """
    Return the canonical node_id for (entity_type, name).

    If an existing node of the same type has a normalized label whose
    fuzzy similarity to the normalized `name` exceeds `threshold`, we
    reuse that node's ID (merging the two mentions).  Otherwise we create
    a new node_id.
    """
    norm = _normalize(name)
    # Collect existing nodes of the same type
    candidates: dict[str, str] = {}   # node_id → normalized_label
    for nid, attrs in G.nodes(data=True):
        if attrs.get("entity_type") == entity_type:
            candidates[nid] = attrs.get("normalized_label", "")

    if candidates:
        # process.extractOne on a dict returns (value, score, key)
        best = process.extractOne(
            norm,
            candidates,         # mapping: node_id → normalized_label
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
        )
        if best is not None:
            _matched_label, _score, best_node_id = best
            return best_node_id

    # New entity
    return f"{entity_type}:{name}"


# ---------------------------------------------------------------------------
# 3. Wikidata enrichment
# ---------------------------------------------------------------------------

def _wikidata_lookup(
    name: str,
    entity_type: str,
    lang: str = "en",
) -> Optional[dict]:
    """
    Query Wikidata for the best matching QID for `name`.

    Returns a dict with keys: qid, label, description, wikidata_url
    or None on failure / no match.
    """
    type_hint = _WIKIDATA_TYPE_HINT.get(entity_type, "")
    filter_clause = (
        f"?item wdt:P31/wdt:P279* {type_hint} ." if type_hint else ""
    )

    sparql = f"""
SELECT ?item ?itemLabel ?itemDescription WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:endpoint "www.wikidata.org";
                    wikibase:api "EntitySearch";
                    mwapi:search "{name}";
                    mwapi:language "{lang}";
                    mwapi:limit "5" .
    ?item wikibase:apiOutputItem mwapi:item .
  }}
  {filter_clause}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{lang},en" .
  }}
}}
LIMIT 1
"""
    params = urllib.parse.urlencode({
        "query": sparql,
        "format": "json",
    })
    url = f"{_WIKIDATA_SPARQL}?{params}"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "AFC-FactChecker/1.0 (research tool; contact: your@email.com)",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_WIKIDATA_TIMEOUT) as resp:
            data = json.loads(resp.read())
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return None
        row = bindings[0]
        qid_uri: str = row["item"]["value"]          # e.g. http://www.wikidata.org/entity/Q76
        qid = qid_uri.rsplit("/", 1)[-1]
        label = row.get("itemLabel", {}).get("value", "")
        description = row.get("itemDescription", {}).get("value", "")
        return {
            "qid": qid,
            "label": label,
            "description": description,
            "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
        }
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
        print(f"KnowledgeGraphTool: Wikidata lookup failed for '{name}': {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class KnowledgeGraphTool(VisionTool):
    TOOL_NAME = "Knowledge Graph"
    INPUTS = ["NER"]

    def __init__(
        self,
        graph_path: str = "afc_graph.graphml",
        fuzzy_threshold: int = _FUZZY_THRESHOLD,
        enrich_wikidata: bool = True,
        wikidata_lang: str = "en",
    ):
        self.graph_path = graph_path
        self.fuzzy_threshold = fuzzy_threshold
        self.enrich_wikidata = enrich_wikidata
        self.wikidata_lang = wikidata_lang

    # ------------------------------------------------------------------
    # Graph persistence
    # ------------------------------------------------------------------

    def _load_graph(self) -> nx.Graph:
        p = Path(self.graph_path)
        if p.exists():
            try:
                return nx.read_graphml(str(p))
            except Exception as e:
                print(
                    f"KnowledgeGraphTool: could not load graph ({e}), starting fresh.",
                    file=sys.stderr,
                )
        return nx.Graph()

    def _save_graph(self, G: nx.Graph) -> None:
        nx.write_graphml(G, self.graph_path)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def _upsert_node(
        self,
        G: nx.Graph,
        entity_type: str,
        name: str,
        media_id: str,
    ) -> str:
        """Add or update a node; returns canonical node_id."""
        node_id = _resolve_node_id(G, entity_type, name, self.fuzzy_threshold)

        if not G.has_node(node_id):
            G.add_node(
                node_id,
                label=name,
                normalized_label=_normalize(name),
                entity_type=entity_type,
                seen_in=json.dumps([media_id]),
                mention_count=1,
                aliases=json.dumps([name]),
                qid="",
                wikidata_label="",
                wikidata_description="",
                wikidata_url="",
            )
        else:
            # Merge: update seen_in, mention count, aliases
            seen = set(json.loads(G.nodes[node_id].get("seen_in", "[]")))
            seen.add(media_id)
            G.nodes[node_id]["seen_in"] = json.dumps(sorted(seen))
            G.nodes[node_id]["mention_count"] = (
                G.nodes[node_id].get("mention_count", 1) + 1
            )
            aliases = set(json.loads(G.nodes[node_id].get("aliases", "[]")))
            aliases.add(name)
            G.nodes[node_id]["aliases"] = json.dumps(sorted(aliases))

        return node_id

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    @staticmethod
    def _upsert_edge(G: nx.Graph, a: str, b: str, media_id: str) -> None:
        if G.has_edge(a, b):
            G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
            co = set(json.loads(G[a][b].get("co_media", "[]")))
            co.add(media_id)
            G[a][b]["co_media"] = json.dumps(sorted(co))
        else:
            G.add_edge(
                a, b,
                weight=1,
                relation="CO_OCCURS_IN",
                co_media=json.dumps([media_id]),
            )

    # ------------------------------------------------------------------
    # Wikidata enrichment pass
    # ------------------------------------------------------------------

    def _enrich(self, G: nx.Graph, node_ids: list[str]) -> int:
        """
        For nodes that lack a QID, query Wikidata and store the result.
        Returns the number of nodes successfully enriched.
        """
        enriched = 0
        candidates = [
            nid for nid in node_ids
            if not G.nodes[nid].get("qid")   # empty string or missing = unenriched
        ][:_WIKIDATA_MAX_ENTITIES]

        for nid in candidates:
            attrs = G.nodes[nid]
            name = attrs.get("label", nid)
            entity_type = attrs.get("entity_type", "")
            print(
                f"KnowledgeGraphTool: Wikidata lookup for '{name}' ({entity_type}) ...",
                file=sys.stderr,
            )
            result = _wikidata_lookup(name, entity_type, self.wikidata_lang)
            if result:
                G.nodes[nid]["qid"] = result["qid"]
                G.nodes[nid]["wikidata_label"] = result["label"]
                G.nodes[nid]["wikidata_description"] = result["description"]
                G.nodes[nid]["wikidata_url"] = result["wikidata_url"]
                enriched += 1
            # Small delay to be polite to the public endpoint
            time.sleep(0.3)

        return enriched

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, data: DataManager) -> dict | None:
        ner_result = data.toolResult.get("NER")
        if not ner_result or not ner_result.get("Output"):
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="NER must run before KnowledgeGraphTool.",
                has_run=0,
            )

        entities: dict[str, list[str]] = ner_result["Output"]
        media_id = Path(data.originalMedia).stem

        G = self._load_graph()
        nodes_before = G.number_of_nodes()
        edges_before = G.number_of_edges()

        # 1. Normalize + fuzzy-resolve + upsert nodes
        node_ids: list[str] = []
        for entity_type, names in entities.items():
            for name in names:
                nid = self._upsert_node(G, entity_type, name, media_id)
                node_ids.append(nid)

        # 2. Co-occurrence edges
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                if a != b:
                    self._upsert_edge(G, a, b, media_id)

        # 3. Wikidata enrichment (only for new/unenriched nodes this run)
        enriched_count = 0
        if self.enrich_wikidata:
            enriched_count = self._enrich(G, node_ids)

        self._save_graph(G)

        output = {
            "graph_path": self.graph_path,
            "nodes_total": G.number_of_nodes(),
            "edges_total": G.number_of_edges(),
            "nodes_added_this_run": G.number_of_nodes() - nodes_before,
            "edges_added_this_run": G.number_of_edges() - edges_before,
            "nodes_this_media": len(node_ids),
            "wikidata_enriched_this_run": enriched_count,
        }

        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=output,
            explanation=(
                f"Graph updated for media '{media_id}': "
                f"{output['nodes_added_this_run']} new node(s), "
                f"{output['edges_added_this_run']} new edge(s). "
                f"{enriched_count} entity/ies enriched via Wikidata."
            ),
            confidence=-1,
            corroborating_tools=["NER", "Metadata Gatherer", "Geolocation"],
        )