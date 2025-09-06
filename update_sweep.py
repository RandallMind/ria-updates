import feedparser, json, hashlib, re, os
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote

SCRIPT_VERSION = "2025-09-05-hard-clamp-v4"

# --------- Fuentes ----------
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
  # --- ES / LATAM (activa quitando el "#") ---
  # "https://www.xataka.com/tag/inteligencia-artificial/rss2.xml",
  # "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/tecnologia/ia/portada",
]

DEFAULT_LIMIT = 6
ARXIV_LIMIT   = 2  # Recomendación experta: 2 para bajar ruido de papers

# --------- Reglas de clasificación ----------
CAT_RULES = [
  # INVESTIGACIÓN primero (papers/arXiv)
  ("Investigación", r"\b(arxiv|paper|research|benchmark|state[- ]of[- ]the[- ]art|sota|investigaci[oó]n)\b"),
  ("Modelos", r"\b(model|gpt|llama|mistral|gemma|embedding|diffusion|clip|transformer)\w*\b"),
  ("Productos", r"\b(launch|releas|introduc|announc|plataforma|app|feature|availability|presenta|lanza|anuncia)\w*\b"),
  ("Herramientas", r"\b(sdk|api|tool|agent|workflow|automation|rpa|plugin|mcp)\w*\b"),
  ("Compliance/Regulación", r"\b(eu ai act|gdpr|compliance|regulation|policy|licen|copyright|privacy|regulaci[oó]n)\w*\b"),
  ("Oportunidades", r"\b(grant|program|jobs|certification|partner|beta|funding|convocatoria|beca|certificaci[oó]n)\w*\b"),
]

# (Solo aplica a NO-arXiv; arXiv será forzado a MEDIA)
ALTA_STRONG_RX = r"\b(launch\w*|releas\w*|introduc\w*|announc\w*|anunci\w*|lanza\w*|general availability|(?<![a-z])ga(?![a-z])|deprecat\w*|sunset\w*|retirad\w*|end[- ]of[- ]life|eol|price|pricing|cost|billing|security|breach\w*|leak\w*|vulnerab\w*|cve-\d{4}-\d+|patch|zero[- ]day|incident\w*|eu ai act|gdpr|licen\w*|policy|terms)\b"

PRIORITY_RULES = [
  ("ALTA",  ALTA_STRONG_RX),
  ("MEDIA", r"\b(update\w*|beta|preview|roll[- ]?out|research|paper|arxiv|benchmark|api|sdk|agent)\b"),
  ("BAJA",  r".*"),
]

PRIORITY_ORDER = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}

# --------- Utilidades ----------
def norm(x: str) -> str:
    if not x:
        return ""
    return re.sub(r"\s+", " ", x).strip()

def url_text(link: str) -> str:
    try:
        p = urlparse(link)
        return unquote((p.netloc + " " + p.path).replace("-", " "))
    except Exception:
        return ""

def pick_summary(e) -> str:
    for key in ("summary", "description"):
        v = getattr(e, key, None)
        if v: return v
    c = getattr(e, "content", None)
    if c:
        try:
            if isinstance(c[0], dict) and "value" in c[0]:
                return c[0]["value"]
            return c[0].value  # type: ignore[attr-defined]
        except Exception:
            pass
    return ""

def classify(text: str):
    t = text.lower()
    cat = next((c for c, rx in CAT_RULES if re.search(rx, t)), "Otros")
    pr  = next((p for p, rx in PRIORITY_RULES if re.search(rx, t)), "BAJA")
    return cat, pr

# Detección robusta de arXiv (host, fuente, o texto "arXiv:2509…")
_ARXIV_TAG_RX = re.compile(r"\barxiv\s*:\s*\d", re.IGNORECASE)
def is_arxiv(link: str, source: str, text: str) -> bool:
    host = urlparse(link).netloc.lower()
    return ("arxiv.org" in host) or ("arxiv" in (source or "").lower()) or bool(_ARXIV_TAG_RX.search(text or ""))

# --------- Recolección ----------
items = []
for url in SOURCES:
    feed = feedparser.parse(url)
    src = getattr(getattr(feed, "feed", {}), "title", "") or url
    entries = getattr(feed, "entries", []) or []
    per_feed_limit = ARXIV_LIMIT if "arxiv.org" in url else DEFAULT_LIMIT

    for e in entries[:per_feed_limit]:
        title = norm(getattr(e, "title", ""))
        link  = norm(getattr(e, "link", ""))
        if not link.startswith("http"):
            continue
        summ  = norm(pick_summary(e))
        base_text = f"{title} {summ} {url_text(link)}"

        cat, pr = classify(base_text)

        # HARD CLAMP arXiv: SIEMPRE Investigación/MEDIA
        arxiv_hit = is_arxiv(link, src, base_text)
        if arxiv_hit:
            cat, pr = "Investigación", "MEDIA"
        else:
            # Reaplica prioridad solo para no-arXiv
            if re.search(ALTA_STRONG_RX, base_text, flags=re.IGNORECASE):
                pr = "ALTA"
            elif re.search(r"\b(update\w*|beta|preview|roll[- ]?out|api|sdk|agent|research|paper|benchmark)\b", base_text, flags=re.IGNORECASE):
                pr = "MEDIA"
            else:
                pr = "BAJA"

        items.append({
            "source": norm(src),
            "title": title,
            "link": link,
            "summary": summ,
            "category": cat,
            "priority": pr,
            "impact": "",
            "risks": "",
            "flags": {"is_arxiv": bool(arxiv_hit)}
        })

# De-dup por link/título
seen, clean = set(), []
for it in items:
    key = it["link"] or it["title"].lower()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    clean.append(it)

# Saneamiento final (doble candado arXiv)
for it in clean:
    if it.get("flags", {}).get("is_arxiv"):
        it["category"] = "Investigación"
        it["priority"] = "MEDIA"

# Orden por prioridad y alfabético
clean.sort(key=lambda it: (PRIORITY_ORDER.get(it["priority"], 9), it["category"], it["title"]))

payload = {
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "meta": {
    "script_version": SCRIPT_VERSION,
    "arxiv_limit": ARXIV_LIMIT,
    "hard_clamp_arxiv": True
  },
  "items": clean[:50]
}

# --------- Salida ----------
with open("updates.json","w",encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

with open("updates.md","w",encoding="utf-8") as f:
    f.write(f"# INTEL Diario – {payload['generated_at']}\n\n")
    f.write(f"_Script: {SCRIPT_VERSION} · arXiv_limit={ARXIV_LIMIT} · hard_clamp_arxiv=True_\n\n")
    for it in payload["items"]:
        f.write(f"- **[{it['category']}/{it['priority']}] {it['title']}** — {it['source']} | {it['link']}\n")
