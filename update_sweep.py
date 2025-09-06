#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep.py
Recolecta feeds RSS/Atom, aplica 'hard clamp' para arXiv, dedupe,
crea updates.json y updates.md.

Requisitos:
    pip install requests feedparser

Uso:
    - Edita ARXIV_LIMIT abajo (2 o 3).
    - Ejecuta: python sweep.py
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List

import feedparser
import requests

# ------------------------------
# CONFIG / AJUSTABLES
# ------------------------------
ARXIV_LIMIT = 2               # <- pon 2 o 3 según prefieras ("espectacular 2" o "espectacular 3")
HARD_CLAMP_ARXIV = True       # si True, limitamos por fuente de arXiv
SCRIPT_VERSION = "2025-09-06-hard-clamp-v7"

# Agente HTTP (identificar tu bot / proyecto)
HTTP_AGENT = "RIA-IntelBot/1.0 (+https://randallmind.github.io/ria-updates) Python-feedparser"

# Timeout en segundos para requests
HTTP_TIMEOUT = 15

# Salidas
OUT_JSON = "updates.json"
OUT_MD = "updates.md"

# Prioridad orden (ALTA primero)
PRIORITY_ORDER = {"ALTA": 0, "MEDIA": 1, "BAJA": 2, "": 3}

# Mapeo simple de "categoría" por fuente (se puede ajustar)
SOURCE_TITLE_OVERRIDES = {
    "https://openai.com/blog/rss.xml": "OpenAI News",
    "https://huggingface.co/blog/feed.xml": "Hugging Face - Blog",
    "https://arxiv.org/rss/cs.AI": "cs.AI updates on arXiv.org (fallback API)",
    "https://arxiv.org/rss/cs.LG": "cs.LG updates on arXiv.org (fallback API)",
    "https://arxiv.org/rss/cs.CL": "cs.CL updates on arXiv.org (fallback API)",
}

# Fuentes (puedes modificar/añadir)
SOURCES = [
    "https://openai.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.LG",
    "https://arxiv.org/rss/cs.CL",
    # puedes añadir más feeds aquí
]

# ------------------------------
# LOGGING
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sweep")

# ------------------------------
# HELPERS
# ------------------------------
def safe_request(url: str, headers: Dict = None, timeout: int = HTTP_TIMEOUT):
    headers = headers or {}
    headers.setdefault("User-Agent", HTTP_AGENT)
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        raise

def md5_of(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def is_arxiv_source(url: str) -> bool:
    return "arxiv.org" in url or url.startswith("http://arxiv.org") or url.startswith("https://arxiv.org")

def normalize_entry(entry) -> Dict:
    """
    Normaliza los campos que esperamos de feedparser.entry
    """
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    # published fallback
    published = getattr(entry, "published", "") or getattr(entry, "updated", "")
    return {"title": title.strip(), "link": link.strip(), "summary": summary.strip(), "published": published}

# ------------------------------
# CORE
# ------------------------------
def fetch_feed(url: str) -> List[Dict]:
    """
    Baja y parsea un feed. Devuelve lista de entradas normalizadas.
    """
    logger.debug(f"Fetching feed: {url}")
    try:
        # fetch with requests to control headers
        resp = safe_request(url)
        content = resp.content
        parsed = feedparser.parse(content)
        if parsed.bozo and parsed.bozo_exception:
            logger.warning(f"feedparser reported bozo for {url}: {parsed.bozo_exception}")
        entries = parsed.entries or []
        results = [normalize_entry(e) for e in entries]
        return results, ""
    except Exception as e:
        logger.error(f"Error fetching/parsing {url}: {e}")
        return [], str(e)

def main():
    start_ts = datetime.now(timezone.utc).isoformat()
    items = []
    sources_stats = {}
    seen_hashes = set()
    arxiv_counts_per_feed = {}

    for src in SOURCES:
        entries, error = fetch_feed(src)
        source_title = SOURCE_TITLE_OVERRIDES.get(src, src)
        count = 0
        used_fallback = False  # en tu flujo original tenías fallback api; aquí lo marcamos solo si error
        if error:
            sources_stats[src] = {"source_title": source_title, "count": 0, "used_fallback": False, "error": error}
            continue

        for ent in entries:
            # Si es arXiv y aplicamos hard clamp, limitamos cuantos items por feed
            if is_arxiv_source(src) and HARD_CLAMP_ARXIV:
                cnt = arxiv_counts_per_feed.get(src, 0)
                if cnt >= ARXIV_LIMIT:
                    continue  # omitimos restantes de ese feed
                arxiv_counts_per_feed[src] = cnt + 1

            normalized = ent
            # Simple dedupe por link+title
            h = md5_of((normalized["link"] + "|" + normalized["title"]).strip().lower())
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Clasificación básica (puedes mejorar reglas)
            category = "Otros"
            title_l = normalized["title"].lower()
            if "arxiv" in normalized["link"] or is_arxiv_source(src):
                category = "Investigación"
            elif "openai" in src or "openai" in normalized["link"]:
                category = "Productos"
            elif "huggingface" in src:
                category = "Herramientas"
            elif "kernel" in title_l or "cuda" in title_l or "gpu" in title_l:
                category = "Otros"

            # Prioridad heurística (puedes ajustar)
            priority = ""
            # si es investigación arXiv -> ALTA por default (pero puedes ajustar)
            if category == "Investigación":
                priority = "ALTA"
            elif category == "Productos":
                priority = "ALTA"
            elif category == "Herramientas":
                priority = "MEDIA"
            else:
                priority = "BAJA"

            item = {
                "source": source_title,
                "title": normalized["title"],
                "link": normalized["link"],
                "summary": normalized["summary"],
                "category": category,
                "priority": priority,
                "impact": "",
                "risks": "",
                "flags": {"is_arxiv": is_arxiv_source(src) or "arxiv.org" in normalized["link"]},
            }
            items.append(item)
            count += 1

        sources_stats[src] = {"source_title": source_title, "count": count, "used_fallback": used_fallback, "error": ""}

    # Ordenar items por prioridad y luego por título (estabilidad)
    def sort_key(it):
        p = it.get("priority", "")
        return (PRIORITY_ORDER.get(p, 99), it.get("title", ""))

    items.sort(key=sort_key)

    # Información meta
    meta = {
        "script_version": SCRIPT_VERSION,
        "arxiv_limit": ARXIV_LIMIT,
        "hard_clamp_arxiv": HARD_CLAMP_ARXIV,
        "http_agent": HTTP_AGENT,
        "sources_stats": sources_stats,
        "generated_at": start_ts,
        "top5_count": 5,
    }

    # Top5: toma los primeros N preservando orden ya ordenado por prioridad
    top5 = []
    for it in items:
        top5.append({
            "source": it["source"],
            "title": it["title"],
            "link": it["link"],
            "category": it["category"],
            "priority": it["priority"],
            "is_arxiv": it["flags"].get("is_arxiv", False)
        })
        if len(top5) >= meta["top5_count"]:
            break

    payload = {"generated_at": start_ts, "meta": meta, "items": items, "top5": top5}

    # Escribir JSON (ensure_ascii=False para mantener acentos)
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote {OUT_JSON} (items: {len(items)})")
    except Exception as e:
        logger.error(f"Error writing {OUT_JSON}: {e}")

    # Escribir Markdown resumen (updates.md)
    try:
        with open(OUT_MD, "w", encoding="utf-8") as mf:
            mf.write(f"# INTEL Diario - {start_ts}\n\n")
            mf.write("## Top 5\n\n")
            for idx, t in enumerate(top5, start=1):
                is_arx = " (arXiv)" if t.get("is_arxiv") else ""
                mf.write(f"{idx}. **{t['title']}** - _{t['source']}_ - {t['category']} - {t['priority']}{is_arx}\n")
                mf.write(f"   {t['link']}\n\n")
            mf.write("---\n\n")
            mf.write("## Items completos\n\n")
            for it in items:
                is_arx = " (arXiv)" if it["flags"].get("is_arxiv") else ""
                mf.write(f"### {it['title']}{is_arx}\n")
                mf.write(f"- Source: {it['source']}\n")
                mf.write(f"- Category: {it['category']}\n")
                mf.write(f"- Priority: {it['priority']}\n")
                mf.write(f"- Link: {it['link']}\n\n")
                if it["summary"]:
                    # keep it short in md
                    s = it["summary"].replace("\n", " ").strip()
                    if len(s) > 1000:
                        s = s[:1000] + "..."
                    mf.write(f"> {s}\n\n")
            mf.write("\n")
        logger.info(f"Wrote {OUT_MD}")
    except Exception as e:
        logger.error(f"Error writing {OUT_MD}: {e}")

    # Print a short status to stdout
    logger.info("Summary:")
    logger.info(f"  total items: {len(items)}")
    logger.info(f"  top5: {len(top5)}")
    for src, st in sources_stats.items():
        logger.info(f"  {src} -> count={st['count']} error={bool(st['error'])}")

if __name__ == "__main__":
    main()
