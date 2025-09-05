import feedparser, json, hashlib, re
from datetime import datetime, timezone

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
  # --- Fuentes ES/LATAM (actívalas quitando el # del inicio de cada línea) ---
  # "https://www.xataka.com/tag/inteligencia-artificial/rss2.xml",
  # "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/tecnologia/ia/portada",
]

CAT_RULES = [
  ("Modelos", r"\b(model|gpt|llama|mistral|gemma|embedding|diffusion|clip|transformer)\b"),
  ("Productos", r"\b(launch|release|introducing|announce|plataforma|app|feature|availability)\b"),
  ("Herramientas", r"\b(sdk|api|tool|agent|workflow|automation|rpa|plugin|mcp)\b"),
  ("Investigación", r"\b(arxiv|paper|research|benchmark|state[- ]of[- ]the[- ]art|sota)\b"),
  ("Compliance/Regulación", r"\b(eu ai act|gdpr|compliance|regulation|policy|licensing|copyright|privacy)\b"),
  ("Oportunidades", r"\b(grant|program|jobs|certification|partner|beta|funding)\b"),
]

PRIORITY_RULES = [
  ("ALTA", r"\b(launch|release|deprecat|price|pricing|security|breach|incident|eu ai act|gdpr|licensing)\b"),
  ("MEDIA", r"\b(update|beta|preview|research|benchmark|api|sdk|agent)\b"),
  ("BAJA", r".*"),
]

def norm(x): return x.replace("\n"," ").strip() if x else ""

def classify(text):
    t = text.lower()
    cat = next((c for c,rx in CAT_RULES if re.search(rx, t)), "Otros")
    pr  = next((p for p,rx in PRIORITY_RULES if re.search(rx, t)), "BAJA")
    return cat, pr

items=[]
for url in SOURCES:
    feed = feedparser.parse(url)
    src = getattr(getattr(feed, "feed", {}), "title", url)
    for e in feed.entries[:5]:
        title = norm(getattr(e,"title",""))
        link  = norm(getattr(e,"link",""))
        summ  = norm(getattr(e,"summary",""))
        cat, pr = classify(f"{title} {summ}")
        items.append({
          "source": norm(src),
          "title": title,
          "link": link,
          "summary": summ,
          "category": cat,
          "priority": pr,
          "impact": "",
          "risks": ""
        })

# De-dup por URL
seen=set(); clean=[]
for it in items:
    h = hashlib.md5(it["link"].encode("utf-8")).hexdigest()
    if h in seen: continue
    seen.add(h); clean.append(it)

payload = {
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "items": clean[:50]
}

with open("updates.json","w",encoding="utf-8") as f:
    json.dump(payload,f,ensure_ascii=False,indent=2)

with open("updates.md","w",encoding="utf-8") as f:
    f.write(f"# INTEL Diario – {payload['generated_at']}\n\n")
    for it in payload["items"]:
        f.write(f"- **[{it['category']}/{it['priority']}] {it['title']}** — {it['source']} | {it['link']}\n")
