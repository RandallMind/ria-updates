#!/usr/bin/env python3
# sweep.py — 2025-09-06-hard-clamp-final
# Recolecta feeds, clasifica, deduplica, y aplica hard clamp para arXiv.
# Requisitos: feedparser (pip install feedparser)

import feedparser
import json
import hashlib
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
from difflib import SequenceMatcher

# ----------------- CONFIG -----------------
SCRIPT_VERSION = "2025-09-06-hard-clamp-final"
ARXIV_LIMIT = 2              # límite global para items desde arXiv
HARD_CLAMP_ARXIV = True      # si True fuerza que solo ARXIV_LIMIT arxiv items queden en salida
MAX_PER_SOURCE = 6           # máximo por fuente tomada del feed (salvo arXiv limitado por ARXIV_LIMIT)
MAX_ITEMS = 50               # máximo total de ítems en output
FUZZY_DEDUP_THRESHOLD = 0.86 # similaridad > esto -> considerar duplicado
HTTP_AGENT = "RIA-IntelBot/1.0 (+https://randallmind.github.io/ria-updates) Python-feedparser"

# Fuentes (editar/activar ES-LATAM si conviene)
SOURCES = [
  "https://openai.com/blog/rss.xml",
  "https://ai.googleblog.com/atom.xml",
  "https://azure.microsoft.com/en-us/updates/feed/",
  "https://about.fb.com/news/tag/ai/feed/",
  "https://www.anthropic.com/news.xml",
  "https://mistral.ai/news/feed.xml",
  "https://huggingface.co/blog/feed.xml",
  "https://arxiv.org/rss/cs.AI",
  "https://arxiv.org/rss/cs.LG",
  "https://arxiv.org/rss/cs.CL",
  # "https://www.xataka.com/tag/inteligencia-artificial/rss2.xml",
  # "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/tecnologia/ia/portada",
]

# Reglas de clasificación (EN + ES básicos)
CAT_RULES = [
  ("Modelos", r"\b(model|gpt|llama|mistral|gemma|embedding|diffusion|clip|transformer|foundation|llm)\w*\b"),
  ("Productos", r"\b(launch|release|introduc|announce|annunci|lanza|plataforma|app|feature|availability|presenta)\w*\b"),
  ("Herramientas", r"\b(sdk|api|tool|agent|workflow|automation|rpa|plugin|mcp|library)\w*\b"),
  ("Investigación", r"\b(arxiv|paper|research|benchmark|state[- ]of[- ]the[- ]art|sota|investigaci[oó]n|study|method)\w*\b"),
  ("Compliance/Regulación", r"\b(eu ai act|gdpr|compliance|regulation|policy|licen|copyright|privacy|regulaci[oó]n)\w*\b"),
  ("Oportunidades", r"\b(grant|program|jobs|certification|partner|beta|funding|beca|convocatoria)\w*\b"),
]

PRIORITY_RULES = [
  ("ALTA", r"\b(launch\w*|releas\w*|introduc\w*|announc\w*|anunci\w*|lanza\w*|general availability|deprecat\w*|sunset|end[- ]of[- ]life|price|pricing|security|breach|vulnerab\w*|cve-\d{4}-\d+|patch|incident)\b"),
  ("MEDIA", r"\b(update\w*|beta|preview|research|paper|arxiv|benchmark|api|sdk|agent|study)\b"),
  ("BAJA", r".*"),
]

PRIORITY_ORDER = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}

# Palabras para filtrar/denoise (configurable)
DENY_KEYWORDS = [
    # temas usualmente ruido para un feed ejecutivo o muy técnico irrelevante
    r"\b(dataset\b|code\b|supplementary\b|appendix\b|workshop\b|workshops?\b|poster\b)\b",
]
# Palabras que bajan prioridad (si aparecen en título/resumen)
DEPRIORITIZE_KEYWORDS = [
    r"\b(virtual try-on|video generation|text-based video games|game)\b"
]

# --------------- UTILIDADES ----------------
def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x or "").strip()

def url_text(link: str) -> str:
    try:
        p = urlparse(link)
        return unquote((p.netloc + " " + p.path).replace("-", " "))
    except Exception:
        return ""

def pick_summary(e):
    for key in ("summary", "description"):
        v = getattr(e, key, None)
        if v:
            return norm(v)
    c = getattr(e, "content", None)
    if c:
        try:
            if isinstance(c, list) and isinstance(c[0], dict) and "value" in c[0]:
                return norm(c[0]["value"])
            return norm(c[0].value)  # type: ignore[attr-defined]
        except Exception:
            pass
    return ""

def fuzzy_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def matches_any(text: str, patterns):
    for p in patterns:
        if re.search(p, text, flags=re.I):
            return True
    return False

def classify(text: str, link: str = ""):
    t = (text + " " + url_text(link)).lower()
    cat = next((c for c, rx in CAT_RULES if re.search(rx, t)), "Otros")
    pr  = next((p for p, rx in PRIORITY_RULES if re.search(rx, t)), "BAJA")
    # deprioritize heuristics
    if matches_any(t, DEPRIORITIZE_KEYWORDS) and pr == "ALTA":
        pr = "MEDIA"
    return cat, pr

def is_arxiv_link(link: str) -> bool:
    try:
        return "arxiv.org" in urlparse(link).netloc.lower()
    except Exception:
        return False

# ------------- RECOLECCIÓN -----------------
items = []
sources_stats = {}

for src_url in SOURCES:
    try:
        feed = feedparser.parse(src_url, agent=HTTP_AGENT)
        title = getattr(getattr(feed, "feed", {}), "title", src_url)
        entries = list(getattr(feed, "entries", []))
        # choose per-source limit
        per_source_limit = MAX_PER_SOURCE
        if "arxiv.org" in src_url:
            per_source_limit = max(1, ARXIV_LIMIT)  # take at most ARXIV_LIMIT per arXiv feed segment
        used_count = 0
        for e in entries[:per_source_limit]:
            title_e = norm(getattr(e, "title", ""))
            link_e  = norm(getattr(e, "link", ""))
            if not link_e.startswith("http"):
                continue
            summ = pick_summary(e)
            combined_text = f"{title_e} {summ}"
            # deny-filters
            if matches_any(combined_text, DENY_KEYWORDS):
                continue
            cat, pr = classify(combined_text, link_e)
            items.append({
                "source": norm(title),
                "title": title_e,
                "link": link_e,
                "summary": summ,
                "category": cat,
                "priority": pr,
                "impact": "",
                "risks": "",
                "flags": {
                    "is_arxiv": is_arxiv_link(link_e)
                },
                "published": getattr(e, "published", "") or getattr(e, "updated", "")
            })
            used_count += 1
        sources_stats[src_url] = {
            "source_title": norm(title),
            "count": used_count,
            "used_fallback": getattr(feed, "bozo", False),
            "error": getattr(feed, "bozo_exception", "")
        }
    except Exception as ex:
        sources_stats[src_url] = {"source_title": src_url, "count": 0, "used_fallback": False, "error": str(ex)}

# -------------- DEDUP (link + fuzzy title) --------------
clean = []
seen_links = set()
titles = []

for it in items:
    key = it["link"] or it["title"].lower()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    if h in seen_links:
        continue
    # fuzzy compare with existing titles
    duplicate = False
    for t in titles:
        if fuzzy_sim(t, it["title"]) > FUZZY_DEDUP_THRESHOLD:
            duplicate = True
            break
    if duplicate:
        continue
    seen_links.add(h)
    titles.append(it["title"])
    clean.append(it)

# -------------- ARXIV HARD-CLAMP & PRIORITIZACION --------------
# Extraemos arxiv items, ordenamos por priority+published recency y limitamos
arxiv_items = [c for c in clean if c.get("flags", {}).get("is_arxiv")]
non_arxiv_items = [c for c in clean if not c.get("flags", {}).get("is_arxiv")]

def sort_key(it):
    pr_ord = PRIORITY_ORDER.get(it.get("priority","BAJA"), 9)
    # try to use published timestamp for recency if available (fallback 0)
    try:
        dt = it.get("published","")
        # no strict parsing: use string as fallback, sort lexicographically reversed (newer first)
        return (pr_ord, 0 if not dt else -1)
    except Exception:
        return (pr_ord, 0)

# keep arxiv items limited
if HARD_CLAMP_ARXIV:
    # sort arxiv by priority then keep top ARXIV_LIMIT
    arxiv_items_sorted = sorted(arxiv_items, key=lambda it: (PRIORITY_ORDER.get(it.get("priority","BAJA"),9), it.get("title","")))
    arxiv_items_kept = arxiv_items_sorted[:max(0, ARXIV_LIMIT)]
else:
    arxiv_items_kept = arxiv_items

# Combine back, then sort globally by priority + category + title and clamp to MAX_ITEMS
combined = non_arxiv_items + arxiv_items_kept
combined.sort(key=lambda it: (PRIORITY_ORDER.get(it.get("priority","BAJA"), 9), it.get("category",""), it.get("title","")))

payload_items = combined[:MAX_ITEMS]

# enrich impact/risks heuristics simple
for it in payload_items:
    t = (it.get("title","") + " " + it.get("summary","")).lower()
    if re.search(r"\b(security|breach|vulnerab|cve-|privacy|exploit)\b", t):
        it["risks"] = "Alto — evaluar impacto de seguridad"
    if it.get("priority") == "ALTA":
        it["impact"] = "Relevante — evaluar acción/monitoreo"

# --------------- METADATA Y TOP5 ----------------
meta = {
    "script_version": SCRIPT_VERSION,
    "arxiv_limit": ARXIV_LIMIT,
    "hard_clamp_arxiv": HARD_CLAMP_ARXIV,
    "http_agent": HTTP_AGENT,
    "sources_stats": sources_stats,
    "top5_count": min(5, len(payload_items))
}

# compute top5 (simple slice of ordered payload)
top5 = []
for it in payload_items[:5]:
    top5.append({
        "source": it["source"],
        "title": it["title"],
        "link": it["link"],
        "category": it["category"],
        "priority": it["priority"],
        "is_arxiv": it.get("flags", {}).get("is_arxiv", False)
    })

out = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "meta": meta,
    "items": payload_items,
    "top5": top5
}

# --------------- SALIDA A ARCHIVOS ----------------
with open("updates.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

with open("updates.md", "w", encoding="utf-8") as f:
    f.write(f"# INTEL Diario – {out['generated_at']}\n\n")
    for it in out["items"]:
        f.write(f"- **[{it['category']}/{it['priority']}] {it['title']}** — {it['source']} | {it['link']}\n")

# --------------- LOG POR CONSOLA ----------------
print("Wrote updates.json and updates.md")
print("script_version:", SCRIPT_VERSION)
print("total_collected:", len(items), "deduped:", len(clean), "final:", len(payload_items))
print("arxiv kept:", sum(1 for i in payload_items if i.get("flags",{}).get("is_arxiv")))
# exit cleanly
sys.exit(0)
