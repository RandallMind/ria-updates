import feedparser, json, time, hashlib
from datetime import datetime, timezone

SOURCES = [
  "https://openai.com/blog/rss.xml",
  "https://ai.googleblog.com/atom.xml",
  "https://azure.microsoft.com/en-us/updates/feed/",
  "https://about.fb.com/news/tag/ai/feed/",
  "https://www.anthropic.com/news.xml",
  "https://mistral.ai/news/feed.xml",
  "https://huggingface.co/blog/feed.xml",
  "https://arxiv.org/rss/cs.AI"
]

def norm(x): return x.replace("\n"," ").strip() if x else ""
items=[]
for url in SOURCES:
    feed = feedparser.parse(url)
    src = getattr(getattr(feed, "feed", {}), "title", url)
    for e in feed.entries[:5]:
        items.append({
          "source": norm(src),
          "title": norm(getattr(e,"title","")),
          "link": norm(getattr(e,"link","")),
          "summary": norm(getattr(e,"summary","")),
          "impact": "",
          "risks": ""
        })

seen=set(); clean=[]
for it in items:
    h = hashlib.md5(it["link"].encode("utf-8")).hexdigest()
    if h in seen: continue
    seen.add(h); clean.append(it)

payload = {
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "items": clean[:40]
}

with open("updates.json","w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
with open("updates.md","w",encoding="utf-8") as f:
    f.write(f"# INTEL Diario – {payload['generated_at']}\n\n")
    for it in payload["items"]:
        f.write(f"- **{it['title']}** — {it['source']} | {it['link']}\n")
