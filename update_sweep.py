# update_sweep.py — v6 (Top-5, dedupe mejorado, clamp arXiv, fallback API, telemetría)
import feedparser, json, hashlib, re, os, time
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote

SCRIPT_VERSION = "2025-09-06-hard-clamp-v6"
HTTP_AGENT     = "RIA-IntelBot/1.0 (+https://randallmind.github.io/ria-updates) Python-feedparser"
HTTP_RETRIES   = 2  # reintentos ligeros

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
ARXIV_LIMIT   = 2  # señal máxima con mínimo ruido (recomendación experta)

# --------- Clasificación ----------
CAT_RULES = [
  ("Investigación", r"\b(arxiv|paper|research|benchmark|state[- ]of[- ]the[- ]art|sota|investigaci[oó]n)\b"),
  ("Modelos", r"\b(model|gpt|llama|mistral|gemma|embedding|diffusion|clip|transformer)\w*\b"),
  ("Productos", r"\b(launch|releas|introduc|announc|plataforma|app|feature|availability|presenta|lanza|anuncia)\w*\b"),
  ("Herramientas", r"\b(sdk|api|tool|agent|workflow|automation|rpa|plugin|mcp)\w*\b"),
  ("Compliance/Regulación", r"\b(eu ai act|gdpr|compliance|regulation|policy|licen|copyright|privacy|regulaci[oó]n)\w*\b"),
  ("Oportunidades", r"\b(grant|program|jobs|certification|partner|beta|funding|convocatoria|beca|certificaci[oó]n)\w*\b"),
]

ALTA_STRONG_RX = r"\b(launch\w*|releas\w*|introduc\w*|announc\w*|anunci\w*|lanza\w*|general availability|(?<![a-z])ga(?![a-z])|deprecat\w*|sunset\w*|retirad\w*|end[- ]of[- ]life|eol|price|pricing|cost|billing|security|breach\w*|leak\w*|vulnerab\w*|cve-\d{4}-\d+|patch|zero[- ]day|incident\w*|eu ai act|gdpr|licen\w*|policy|terms)\b"

PRIORITY_RULES = [
  ("ALTA",  ALTA_STRONG_RX),
  ("MEDIA", r"\b(update\w*|beta|preview|roll[- ]?out|research|paper|arxiv|benchmark|api|sdk|agent)\b"),
  ("BAJA",  r".*"),
]
PRIORITY_ORDER = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}

CATEGORY_ORDER = [
  "Productos", "Compliance/Regulación", "Herramientas",
  "Modelos", "Investigación", "Oportunidades", "Otros"
]
CATEGORY_SCORE = {c:i for i,c in enumerate(CATEGORY_ORDER)}

# --------- Utilidades ----------
def norm(x: str) -> str:
    if not x: return ""
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
            return c[0].value  # type: ignore
        except Exception:
            pass
    return ""

def classify(text: str):
    t = text.lower()
    cat = next((c for c, rx in CAT_RULES if re.search(rx, t)), "Otros")
    pr  = next((p for p, rx in PRIORITY_RULES if re.search(rx, t)), "BAJA")
    return cat, pr

_ARXIV_TAG_RX = re.compile(r"\barxiv\s*:\s*\d", re.IGNORECASE)
def is_arxiv(link: str, source: str, text: str) -> bool:
    host = urlparse(link).netloc.lower()
    return ("arxiv.org" in host) or ("arxiv" in (source or "").lower()) or bool(_ARXIV_TAG_RX.search(text or ""))

# --- Fallback API de arXiv (ATOM) si RSS viene vacío ---
def arxiv_api_from_rss_url(url: str) -> str | None:
    try:
        p = urlparse(url)
        if "arxiv.org" not in p.netloc or not p.path.startswith("/rss/"):
            return None
        cat = p.path.split("/rss/")[1].strip("/")
        if not cat: return None
        mr = max(ARXIV_LIMIT, 2)
        return f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results={mr}&sortBy=submittedDate&sortOrder=descending"
    except Exception:
        return None

def parse_with_retries(url: str):
    last_err = ""
    for _ in range(HTTP_RETRIES + 1):
        try:
            feed = feedparser.parse(url, agent=HTTP_AGENT)
            return feed, ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    return None, last_err

# --------- Recolección ----------
items = []
stats = {}  # telemetría por fuente

for url in SOURCES:
    per_feed_limit = ARXIV_LIMIT if "arxiv.org" in url else DEFAULT_LIMIT
    feed, err = parse_with_retries(url)
    src_title = url
    entries = []
    used_fallback = False

    if feed:
        src_title = getattr(getattr(feed, "feed", {}), "title", "") or url
        entries = getattr(feed, "entries", []) or []

    # Fallback: si es arXiv y no hay entries, usa export API
    if (not entries) and ("arxiv.org" in url):
        fb = arxiv_api_from_rss_url(url)
        if fb:
            fb_feed, fb_err = parse_with_retries(fb)
            if fb_feed:
                fb_title = getattr(getattr(fb_feed, "feed", {}), "title", "") or src_title
                src_title = f"{src_title} (fallback API)"
                entries = getattr(fb_feed, "entries", []) or []
                used_fallback = True
                err = ""
            else:
                err = err or fb_err

    stats[url] = {
        "source_title": src_title,
        "count": len(entries[:per_feed_limit]),
        "used_fallback": used_fallback,
        "error": err
    }

    for e in entries[:per_feed_limit]:
        title = norm(getattr(e, "title", ""))
        link  = norm(getattr(e, "link", ""))
        if not link.startswith("http"):
            continue
        summ  = norm(pick_summary(e))
        base_text = f"{title} {summ} {url_text(link)}"
        cat, pr = classify(base_text)

        # HARD CLAMP arXiv: SIEMPRE Investigación/MEDIA
        arxiv_hit = is_arxiv(link, src_title, base_text)
        if arxiv_hit:
            cat, pr = "Investigación", "MEDIA"
        else:
            if re.search(ALTA_STRONG_RX, base_text, flags=re.IGNORECASE):
                pr = "ALTA"
            elif re.search(r"\b(update\w*|beta|preview|roll[- ]?out|api|sdk|agent|research|paper|benchmark)\b",
                           base_text, flags=re.IGNORECASE):
                pr = "MEDIA"
            else:
                pr = "BAJA"

        items.append({
            "source": norm(src_title),
            "title": title,
            "link": link,
            "summary": summ,
            "category": cat,
            "priority": pr,
            "impact": "",
            "risks": "",
            "flags": {"is_arxiv": bool(arxiv_hit)}
        })

# --------- De-dup 1: por link/título exacto ----------
seen, clean = set(), []
for it in items:
    key = it["link"] or it["title"].lower()
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    clean.append(it)

# --------- De-dup 2: similitud de títulos (Jaccard tokens) ----------
STOPWORDS = set("a an the and or of on en de la el los las un una para por con y del al".split())
def title_tokens(t: str) -> set[str]:
    t = re.sub(r"[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ ]+", " ", t.lower())
    toks = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
    return set(toks)

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

dedup2 = []
for it in clean:
    tset = title_tokens(it["title"])
    dup = False
    for jt in dedup2:
        # solo consideramos duplicado si (misma categoría o misma prioridad) y títulos muy similares
        if (it["category"] == jt["category"]) or (it["priority"] == jt["priority"]):
            if jaccard(tset, title_tokens(jt["title"])) >= 0.82:
                dup = True
                break
    if not dup:
        dedup2.append(it)
clean = dedup2

# --------- Saneamiento final (doble candado arXiv) ----------
for it in clean:
    if it.get("flags", {}).get("is_arxiv"):
        it["category"] = "Investigación"
        it["priority"] = "MEDIA"

# --------- Orden principal ----------
clean.sort(key=lambda it: (
    PRIORITY_ORDER.get(it["priority"], 9),
    CATEGORY_SCORE.get(it["category"], 99),
    0 if not it.get("flags", {}).get("is_arxiv") else 1,
    it["title"]
))

# --------- Top-5 ejecutivo ----------
def choose_top5(items):
    top = []
    def pick(filter_fn, limit=None):
        nonlocal top
        for it in items:
            if it in top: continue
            if filter_fn(it):
                top.append(it)
                if limit and len(top) >= limit: break

    # 1) ALTA no arXiv
    pick(lambda it: it["priority"]=="ALTA" and not it["flags"].get("is_arxiv"))
    # 2) MEDIA no arXiv (ordenado por categoría preferente)
    if len(top) < 5:
        pick(lambda it: it["priority"]=="MEDIA" and not it["flags"].get("is_arxiv"))
    # 3) MEDIA arXiv (para completar si hiciera falta)
    if len(top) < 5:
        pick(lambda it: it["priority"]=="MEDIA" and it["flags"].get("is_arxiv"))
    # 4) Resto
    if len(top) < 5:
        pick(lambda it: True)

    return top[:5]

top5 = choose_top5(clean)

payload = {
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "meta": {
    "script_version": SCRIPT_VERSION,
    "arxiv_limit": ARXIV_LIMIT,
    "hard_clamp_arxiv": True,
    "http_agent": HTTP_AGENT,
    "sources_stats": {},  # se rellena debajo
    "top5_count": len(top5)
  },
  "items": clean[:50],
  "top5": [
      {
        "source": it["source"], "title": it["title"], "link": it["link"],
        "category": it["category"], "priority": it["priority"],
        "is_arxiv": it["flags"].get("is_arxiv", False)
      } for it in top5
  ]
}
payload["meta"]["sources_stats"] = {k:v for k,v in stats.items()}

# --------- Salida ----------
with open("updates.json","w",encoding="utf-8") as f:
  json.dump(payload, f, ensure_ascii=False, indent=2)

with open("updates.md","w",encoding="utf-8") as f:
  f.write(f"# INTEL Diario – {payload['generated_at']}\n\n")
  f.write(f"_Script: {SCRIPT_VERSION} · arXiv_limit={ARXIV_LIMIT} · hard_clamp_arXiv=True_\n\n")
  # Top 5 ejecutivo
  f.write("## Top 5 ejecutivo\n\n")
  for it in payload["top5"]:
      f.write(f"- **[{it['category']}/{it['priority']}] {it['title']}** — {it['source']} | {it['link']}\n")
  f.write("\n---\n\n")
  # Resto de novedades
  f.write("## Resto de novedades\n\n")
  for it in payload["items"]:
      f.write(f"- **[{it['category']}/{it['priority']}] {it['title']}** — {it['source']} | {it['link']}\n")
