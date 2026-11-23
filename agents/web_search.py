"""Recherche web réelle via DuckDuckGo (bibliothèque ddgs), sans clé API.

Remplace Google Search grounding, refusé sur le palier gratuit de Gemini (429). Les résultats
(titre, URL, extrait) sont collectés ici, numérotés [S1], [S2]… puis transmis à Gemini, qui ne
peut citer que ces sources : la vérification est faite par Python (agents/web_agent.py).
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

SEARCH_TIMEOUT_S = 15
PER_QUERY = 5
PAUSE_S = 1.0          # politesse entre deux requêtes (limitation de débit côté DuckDuckGo)
MAX_SNIPPET = 400


class WebSearchError(RuntimeError):
    pass


@dataclass
class WebHit:
    id: str          # « S1 », « S2 »…
    title: str
    url: str
    snippet: str
    query: str


SearchFn = Callable[[str, int], list[dict]]


def ddgs_search(query: str, max_results: int) -> list[dict]:
    """Une requête DuckDuckGo ; renvoie des dicts {title, href, body}."""
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise WebSearchError("Bibliothèque ddgs absente (pip install ddgs).") from e
    return DDGS(timeout=SEARCH_TIMEOUT_S).text(query, max_results=max_results) or []


def run_searches(queries: list[str], per_query: int = PER_QUERY, search: SearchFn | None = None,
                 pause: float | None = None) -> tuple[list[WebHit], list[str]]:
    """Exécute chaque requête (2 tentatives), déduplique par URL et numérote les sources.

    Renvoie (sources, journal). Lève WebSearchError si AUCUNE requête n'a donné de résultat.
    """
    search = search or ddgs_search
    pause = PAUSE_S if pause is None else pause
    hits: list[WebHit] = []
    seen: set[str] = set()
    journal: list[str] = []
    for i, q in enumerate(queries):
        if i and pause:
            time.sleep(pause)
        results, err = None, None
        for attempt in range(2):
            try:
                results = search(q, per_query)
                break
            except Exception as e:  # noqa: BLE001  (réseau, limitation de débit, format)
                err = e
                log.warning("DuckDuckGo « %s » tentative %d : %s", q, attempt + 1, e)
                if pause:
                    time.sleep(pause * 2)
        if results is None:
            journal.append(f"« {q} » : échec ({type(err).__name__})")
            continue
        added = 0
        for r in results:
            url = (r.get("href") or r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            if not url.startswith(("http://", "https://")) or url in seen or not title:
                continue
            seen.add(url)
            snippet = " ".join((r.get("body") or "").split())[:MAX_SNIPPET]
            hits.append(WebHit(f"S{len(hits) + 1}", title, url, snippet, q))
            added += 1
        journal.append(f"« {q} » : {len(results)} résultat(s), {added} nouveau(x)")
    if not hits:
        raise WebSearchError("Aucun résultat DuckDuckGo exploitable.\n" + "\n".join(journal))
    return hits, journal
