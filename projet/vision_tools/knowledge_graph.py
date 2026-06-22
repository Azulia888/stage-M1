"""
knowledge_graph.py — KnowledgeGraphTool: builds a persistent co-occurrence graph
from NER output, with entity normalization, fuzzy deduplication, and optional
Wikidata QID enrichment.

Dependencies: networkx, rapidfuzz (pip install networkx rapidfuzz)
"""

from __future__ import annotations

import json
import time
import requests

from data_manager import DataManager
from vision_tools.base import VisionTool, _make_tool_json



OUTPUT_FILE = "wikidata_enriched.json"

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Delay between requests to be polite to Wikidata's servers
REQUEST_DELAY = 0.5  # seconds

# SPARQL query template – fetches a broad set of commonly useful properties.
# Adjust or extend as needed for your domain.
SPARQL_QUERY_TEMPLATE = """
SELECT ?item ?itemLabel ?itemDescription ?instanceOfLabel ?subclassOfLabel
       ?countryLabel ?dateOfBirth ?dateOfDeath ?officialWebsite ?image
WHERE {{
  BIND(wd:{qid} AS ?item)
  OPTIONAL {{ ?item wdt:P31 ?instanceOf. }}
  OPTIONAL {{ ?item wdt:P279 ?subclassOf. }}
  OPTIONAL {{ ?item wdt:P17 ?country. }}
  OPTIONAL {{ ?item wdt:P569 ?dateOfBirth. }}
  OPTIONAL {{ ?item wdt:P570 ?dateOfDeath. }}
  OPTIONAL {{ ?item wdt:P856 ?officialWebsite. }}
  OPTIONAL {{ ?item wdt:P18 ?image. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 10
"""

HEADERS = {
    "User-Agent": "NER-Enrichment-Bot/1.0 (research project; contact@example.com)",
    "Accept": "application/json",
}


def search_entity(entity_name: str) -> dict | None:
    """Return the top Wikidata search result for *entity_name*, or None."""
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "type": "item",
        "limit": 1,
    }
    try:
        response = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        results = response.json().get("search", [])
        if results:
            return results[0]
    except requests.RequestException as e:
        print(f"  [search error] {entity_name}: {e}")
    return None


def fetch_sparql(qid: str) -> list[dict]:
    """Run the SPARQL query for *qid* and return the bindings list."""
    query = SPARQL_QUERY_TEMPLATE.format(qid=qid)
    try:
        response = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        return bindings
    except requests.RequestException as e:
        print(f"  [sparql error] {qid}: {e}")
    return []


def simplify_bindings(bindings: list[dict]) -> list[dict]:
    """Flatten SPARQL result bindings to plain dicts."""
    simplified = []
    for row in bindings:
        simplified.append({key: val.get("value") for key, val in row.items()})
    return simplified


def enrich_entity(entity_name: str) -> dict:
    """Search Wikidata for *entity_name*, query SPARQL, return enriched record."""
    print(f"Processing: {entity_name}")
    record = {"entity": entity_name, "wikidata_id": None, "search_result": None, "sparql_data": []}

    search_hit = search_entity(entity_name)
    time.sleep(REQUEST_DELAY)

    if not search_hit:
        print(f"  No result found.")
        return record

    qid = search_hit["id"]
    record["wikidata_id"] = qid
    record["search_result"] = {
        "id": qid,
        "label": search_hit.get("label"),
        "description": search_hit.get("description"),
        "url": search_hit.get("url"),
    }
    print(f"  Found: {qid} – {search_hit.get('label')} ({search_hit.get('description', '')})")

    bindings = fetch_sparql(qid)
    time.sleep(REQUEST_DELAY)

    record["sparql_data"] = simplify_bindings(bindings)
    print(f"  SPARQL rows: {len(record['sparql_data'])}")

    return record



class KnowledgeGraphTool(VisionTool):
    TOOL_NAME = "Knowledge Graph"
    INPUTS = ["NER"]

    def __init__(
        self,
        enrich_wikidata: bool = True,
        wikidata_lang: str = "en",
    ):
        self.enrich_wikidata = enrich_wikidata
        self.wikidata_lang = wikidata_lang

    

    def run(self, data: DataManager) -> dict | None:
        ner_result = data.toolResult.get("NER")
        if not ner_result or not ner_result.get("Output"):
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="NER must run before KnowledgeGraphTool.",
                has_run=0,
            )

        ner_dict: dict[str, list[str]] = ner_result["Output"]
        entities = ner_dict["entities"]
        list_ners = []
        for key, value in entities.items():
            list_ners.extend(value)
        
        results = []
        for name in list_ners:
            results.append(enrich_entity(name))
            time.sleep(5) #waiting for 5 seconds to try to prevent sending too many requests to wikidata and getting blocked

        path = Path(data.originalMedia) / OUTPUT_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=results,
            explanation=(""
            ),
            confidence=-1,
            corroborating_tools=["NER", "Metadata Gatherer", "Geolocation"],
        )