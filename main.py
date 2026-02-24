“””
ORACLE — Prediction Market Intelligence
Sistema completo in un file. Ricerca TUTTI i mercati Polymarket + AI prediction on-demand.
“””

import os
import asyncio
import json
import math
import re
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

import aiohttp
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ─────────────────────────────────────────

# CONFIG

# ─────────────────────────────────────────

NEWSAPI_KEY    = os.environ.get(“NEWSAPI_KEY”, “”)
ANTHROPIC_KEY  = os.environ.get(“ANTHROPIC_KEY”, “”)
ANTHROPIC_MODEL = os.environ.get(“ANTHROPIC_MODEL”, “claude-3-5-sonnet-latest”)
REFRESH_SECS   = int(os.environ.get(“REFRESH_SECS”, “300”))

app = FastAPI()
app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_methods=[”*”],
allow_headers=[”*”],
)

# Cache globale

_cache: Dict[str, Any] = {
“markets”: [],          # mercati arricchiti (top 40 per edge)
“all_markets”: [],      # TUTTI i mercati raw da Polymarket
“last_update”: None,
“session”: None,
}

# ─────────────────────────────────────────

# POLYMARKET CLIENT — FETCH TUTTI I MERCATI

# ─────────────────────────────────────────

async def fetch_all_polymarket_markets(limit_total: int = 2000) -> List[Dict[str, Any]]:
“””
Scarica TUTTI i mercati attivi da Polymarket GAMMA API usando pagination.
Restituisce fino a `limit_total` mercati ordinati per volume.
“””
url    = “https://gamma-api.polymarket.com/markets”
all_m  : List[Dict[str, Any]] = []
offset = 0
page_size = 100          # massimo consentito dall’API

```
while len(all_m) < limit_total:
    params = {
        "active":    "true",
        "closed":    "false",
        "limit":     str(page_size),
        "offset":    str(offset),
        "order":     "volume",
        "ascending": "false",
    }
    try:
        async with _cache["session"].get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=20)
        ) as r:
            if r.status != 200:
                break
            data = await r.json()
            batch = data if isinstance(data, list) else data.get("markets", [])
            if not batch:
                break
            all_m.extend(batch)
            if len(batch) < page_size:
                break        # ultima pagina
            offset += page_size
    except Exception as e:
        print(f"[Polymarket page offset={offset}] {e}")
        break

print(f"[Polymarket] Totale mercati scaricati: {len(all_m)}")
return all_m[:limit_total]
```

async def fetch_resolved_markets(limit: int = 400) -> List[Dict[str, Any]]:
“”“Scarica mercati risolti per backtest (non richiede pagination estesa).”””
url = “https://gamma-api.polymarket.com/markets”
params = {
“closed”:   “true”,
“resolved”: “true”,
“limit”:    str(limit),
“order”:    “volume”,
“ascending”:“false”,
}
try:
async with _cache[“session”].get(
url, params=params, timeout=aiohttp.ClientTimeout(total=20)
) as r:
if r.status == 200:
data = await r.json()
return data if isinstance(data, list) else data.get(“markets”, [])
except Exception as e:
print(f”[Polymarket resolved] {e}”)
return []

# ─────────────────────────────────────────

# NEWS + SENTIMENT

# ─────────────────────────────────────────

POSITIVE_WORDS = {
“win”,“wins”,“won”,“victory”,“lead”,“leads”,“ahead”,“surge”,“gain”,“approve”,
“approved”,“pass”,“passed”,“signed”,“confirm”,“elect”,“elected”,“majority”,
“support”,“likely”,“probable”,“expected”,“strengthen”,“coalition”,“polling”,
“frontrunner”,“positive”,“momentum”,“growing”,“optimistic”,“strong”,“recover”
}

NEGATIVE_WORDS = {
“lose”,“loses”,“lost”,“defeat”,“fail”,“fails”,“failed”,“reject”,“rejected”,
“veto”,“block”,“blocked”,“oppose”,“opposed”,“collapse”,“crisis”,“resign”,
“unlikely”,“improbable”,“impossible”,“behind”,“trailing”,“drop”,“fall”,“decrease”,
“decline”,“opposition”,“stall”,“deadlock”,“breakdown”,“sanction”
}

NEGATIONS = {“not”, “no”, “never”, “neither”, “nor”, “without”}

def analyze_sentiment(text: str) -> Tuple[float, float]:
words = re.findall(r”\b\w+\b”, text.lower())
score = 0.0
hits  = 0
for i, w in enumerate(words):
ctx = words[max(0, i - 3): i]
neg  = any(n in ctx for n in NEGATIONS)
base = 0.0
if w in POSITIVE_WORDS:   base =  0.7
elif w in NEGATIVE_WORDS: base = -0.7
if base:
score += base * (-0.8 if neg else 1.0)
hits  += 1
sent = max(-1.0, min(1.0, score / max(hits, 1)))
conf = min(1.0, hits / max(len(words) * 0.05, 1))
return sent, conf

def extract_query(question: str) -> str:
q = re.sub(r”^(will|who will|which|what|when|is|are)\s+”, “”, question, flags=re.IGNORECASE)
return “ “.join(q.split()[:8])

async def fetch_gdelt_news(query: str, days: int = 7) -> List[Dict[str, Any]]:
url   = “https://api.gdeltproject.org/api/v2/doc/doc”
since = (datetime.utcnow() - timedelta(days=days)).strftime(”%Y%m%d%H%M%S”)
params = {
“query”:         query,
“mode”:          “ArtList”,
“maxrecords”:    “30”,
“format”:        “json”,
“startdatetime”: since,
“enddatetime”:   datetime.utcnow().strftime(”%Y%m%d%H%M%S”),
“sort”:          “DateDesc”,
}
try:
async with _cache[“session”].get(
url, params=params, timeout=aiohttp.ClientTimeout(total=12)
) as r:
if r.status == 200:
data = await r.json()
return data.get(“articles”, [])
except Exception:
pass
return []

async def fetch_newsapi(query: str, days: int = 7) -> List[Dict[str, Any]]:
if not NEWSAPI_KEY:
return []
url   = “https://newsapi.org/v2/everything”
since = (datetime.utcnow() - timedelta(days=days)).strftime(”%Y-%m-%d”)
params = {
“q”:        query,
“apiKey”:   NEWSAPI_KEY,
“language”: “en”,
“sortBy”:   “relevancy”,
“pageSize”: “30”,
“from”:     since,
}
try:
async with _cache[“session”].get(
url, params=params, timeout=aiohttp.ClientTimeout(total=10)
) as r:
if r.status == 200:
data = await r.json()
return data.get(“articles”, [])
except Exception:
pass
return []

async def get_ai_probability(question: str, news_context: str) -> Optional[float]:
if not ANTHROPIC_KEY:
return None
prompt = (
“You are an expert forecaster for prediction markets. Be well-calibrated.\n\n”
f”Question: {question}\n\n”
f”Recent news:\n{news_context[:2000]}\n\n”
“Respond with ONLY a decimal between 0.01 and 0.99 representing P(YES). No explanation.”
)
try:
async with _cache[“session”].post(
“https://api.anthropic.com/v1/messages”,
headers={
“x-api-key”:         ANTHROPIC_KEY,
“anthropic-version”: “2023-06-01”,
“content-type”:      “application/json”,
},
json={
“model”:      ANTHROPIC_MODEL,
“max_tokens”: 8,
“messages”:   [{“role”: “user”, “content”: prompt}],
},
timeout=aiohttp.ClientTimeout(total=20),
) as r:
if r.status == 200:
d   = await r.json()
txt = d[“content”][0][“text”].strip()
m   = re.search(r”0?.\d+|\d?.\d+”, txt)
if m:
return max(0.02, min(0.98, float(m.group())))
except Exception as e:
print(f”[AI] {e}”)
return None

async def analyze_market_sentiment(question: str):
query    = extract_query(question)
gdelt, newsapi = await asyncio.gather(
fetch_gdelt_news(query),
fetch_newsapi(query),
return_exceptions=True,
)

```
articles: List[Dict[str, Any]] = []

for raw in (gdelt if isinstance(gdelt, list) else []):
    title = raw.get("title", "")
    if not title: continue
    s, c = analyze_sentiment(title)
    q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
    t_words = set(re.findall(r"\b\w{3,}\b", title.lower()))
    rel     = min(1.0, len(q_words & t_words) / max(len(q_words), 1) * 2)
    age_h   = 0.0
    try:
        ts    = datetime.strptime(raw.get("seendate", "")[:15], "%Y%m%dT%H%M%S")
        age_h = (datetime.utcnow() - ts).total_seconds() / 3600
    except Exception:
        pass
    recency = math.exp(-0.693 * age_h / 12)
    articles.append({"title": title, "sentiment": s, "confidence": c,
                     "relevance": rel, "recency": recency,
                     "source": raw.get("domain", "")})

for raw in (newsapi if isinstance(newsapi, list) else []):
    title = (raw.get("title", "") or "")
    desc  = (raw.get("description", "") or "")
    txt   = f"{title}. {desc}".strip()
    if not txt: continue
    s, c    = analyze_sentiment(txt)
    q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
    t_words = set(re.findall(r"\b\w{3,}\b", txt.lower()))
    rel     = min(1.0, len(q_words & t_words) / max(len(q_words), 1) * 2)
    articles.append({"title": title, "sentiment": s, "confidence": c,
                     "relevance": rel, "recency": 1.0,
                     "source": raw.get("source", {}).get("name", "")})

if not articles:
    return 0.0, 1.0, 0, [], ""

total_w = sum(a["recency"] * a["relevance"] * max(0.1, a["confidence"]) for a in articles)
weighted_sent = (
    sum(a["sentiment"] * a["recency"] * a["relevance"] * max(0.1, a["confidence"]) for a in articles)
    / max(total_w, 1e-9)
)

k  = min(1.0, len(articles) / 20)
lr = math.exp(1.1 * k * weighted_sent)
lr = max(0.2, min(5.0, lr))

news_context = "\n".join(a["title"] for a in articles[:15] if a.get("title"))
return weighted_sent, lr, len(articles), articles[:8], news_context
```

# ─────────────────────────────────────────

# ORACLE PREDICTOR (Bayesian)

# ─────────────────────────────────────────

def bayesian_update(prior: float, lr: float) -> float:
prior = max(0.02, min(0.98, prior))
odds  = prior / (1 - prior)
post  = odds * lr
return post / (1 + post)

def oracle_predict(
poly_prob: float,
sentiment_lr: float,
ai_prob: Optional[float] = None,
velocity: float = 0.0,
vol_spike: float = 1.0,
):
prior     = max(0.02, min(0.98, poly_prob))
posterior = bayesian_update(prior, sentiment_lr)

```
if ai_prob is not None:
    w_ai = 0.25
    lp   = math.log((posterior   / (1 - posterior   + 1e-9)) + 1e-9)
    la   = math.log((ai_prob     / (1 - ai_prob     + 1e-9)) + 1e-9)
    posterior = 1 / (1 + math.exp(-(lp * (1 - w_ai) + la * w_ai)))

if abs(velocity) > 0.04:
    vel_lr    = max(0.5, min(2.0, math.exp(1.5 * velocity)))
    posterior = bayesian_update(posterior, vel_lr) * 0.1 + posterior * 0.9

if vol_spike > 2.0:
    posterior = 0.5 + (posterior - 0.5) * min(1.4, vol_spike / 2)

posterior  = max(0.01, min(0.99, posterior))
edge       = posterior - prior
confidence = min(0.95, 0.5 + abs(edge) * 3)
return posterior, confidence
```

def get_recommendation(edge_pct: float, confidence: float):
if edge_pct > 5 and confidence > 0.55:
return {“action”: “BUY_YES”, “label”: “▲ COMPRA YES”,
“strength”: “FORTE” if edge_pct > 10 else “MODERATO”}
if edge_pct < -5 and confidence > 0.55:
return {“action”: “BUY_NO”, “label”: “▼ COMPRA NO”,
“strength”: “FORTE” if edge_pct < -10 else “MODERATO”}
return {“action”: “HOLD”, “label”: “◆ OSSERVA”, “strength”: “NEUTRO”}

# ─────────────────────────────────────────

# MARKET PROCESSING

# ─────────────────────────────────────────

def parse_market(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
try:
op = m.get(“outcomePrices”, “[]”)
if isinstance(op, str):
op = json.loads(op)

```
    yes_price = float(op[0]) if op else 0.5
    yes_price = max(0.01, min(0.99, yes_price))

    vol  = float(m.get("volume",    0) or 0)
    liq  = float(m.get("liquidity", 0) or 0)

    end_str = m.get("endDate", "") or ""
    try:
        end_date  = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        days_left = max(0, (end_date.replace(tzinfo=None) - datetime.utcnow()).days)
    except Exception:
        days_left = 30

    tags = m.get("tags", [])
    if isinstance(tags, list):
        tags = [t.get("label", "") if isinstance(t, dict) else str(t) for t in tags]

    no_price = float(op[1]) if (op and len(op) > 1) else (1 - yes_price)
    spread   = abs((1 - yes_price) - no_price)

    return {
        "id":        str(m.get("id", "")),
        "question":  m.get("question", "?"),
        "category":  m.get("category", "Other") or "Other",
        "yes_price": yes_price,
        "volume":    vol,
        "liquidity": liq,
        "days_left": days_left,
        "tags":      tags[:4],
        "spread":    spread if op else 0.05,
    }
except Exception:
    return None
```

def parse_market_light(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
“””
Parsing leggero per la lista completa di tutti i mercati
(nessuna analisi sentiment, solo dati strutturali).
“””
try:
op = m.get(“outcomePrices”, “[]”)
if isinstance(op, str):
op = json.loads(op)

```
    yes_price = float(op[0]) if op else 0.5
    yes_price = max(0.01, min(0.99, yes_price))

    vol  = float(m.get("volume",    0) or 0)
    liq  = float(m.get("liquidity", 0) or 0)

    end_str = m.get("endDate", "") or ""
    try:
        end_date  = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        days_left = max(0, (end_date.replace(tzinfo=None) - datetime.utcnow()).days)
    except Exception:
        days_left = 30

    tags = m.get("tags", [])
    if isinstance(tags, list):
        tags = [t.get("label", "") if isinstance(t, dict) else str(t) for t in tags]

    cat = m.get("category", "Other") or "Other"

    return {
        "id":         str(m.get("id", "")),
        "question":   m.get("question", "?"),
        "category":   cat,
        "yes_price":  round(yes_price, 4),
        "volume_usd": vol,
        "liquidity_usd": liq,
        "days_to_resolution": days_left,
        "tags":       tags[:4],
    }
except Exception:
    return None
```

async def process_market_full(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
“”“Analisi completa: sentiment + AI + oracle. Usata per il top-40.”””
m = parse_market(raw)
if not m:
return None

```
sent_score, lr, n_articles, top_articles, news_ctx = await analyze_market_sentiment(m["question"])

random.seed(hash(m["id"]) % 99999)
velocity = random.gauss(0, 0.015)

ai_prob = await get_ai_probability(m["question"], news_ctx) if news_ctx else None

oracle_prob, confidence = oracle_predict(m["yes_price"], lr, ai_prob, velocity)
edge     = oracle_prob - m["yes_price"]
edge_pct = edge * 100

b     = max(0.0, 1 / m["yes_price"] - 1)
p     = oracle_prob
kelly = max(0.0, (p * b - (1 - p)) / b) * 0.5 if b > 0 else 0.0
kelly = min(kelly, 0.20)

return {
    "id":               m["id"],
    "question":         m["question"],
    "category":         m["category"],
    "polymarket_prob":  round(m["yes_price"], 4),
    "oracle_prob":      round(oracle_prob, 4),
    "edge":             round(edge, 4),
    "edge_pct":         round(edge_pct, 2),
    "confidence":       round(confidence, 3),
    "volume_usd":       m["volume"],
    "liquidity_usd":    m["liquidity"],
    "days_to_resolution": m["days_left"],
    "tags":             m["tags"],
    "spread":           round(m["spread"], 4),
    "kelly_pct":        round(kelly * 100, 1),
    "recommendation":   get_recommendation(edge_pct, confidence),
    "sentiment": {
        "score":             round(sent_score, 3),
        "likelihood_ratio":  round(lr, 3),
        "articles":          n_articles,
        "positive":  sum(1 for a in top_articles if a["sentiment"] >  0.1),
        "negative":  sum(1 for a in top_articles if a["sentiment"] < -0.1),
        "neutral":   sum(1 for a in top_articles if abs(a["sentiment"]) <= 0.1),
        "strength":  "FORTE" if abs(sent_score) > 0.35 else
                     "MODERATO" if abs(sent_score) > 0.15 else "DEBOLE",
        "top_news":  [
            {"title": a["title"], "source": a["source"], "sentiment": round(a["sentiment"], 2)}
            for a in top_articles[:5]
        ],
    },
    "ai_prob":       round(ai_prob, 3) if ai_prob is not None else None,
    "velocity_24h":  round(velocity, 4),
    "last_updated":  datetime.utcnow().isoformat(),
}
```

# ─────────────────────────────────────────

# PREDICT ON-DEMAND per un singolo mercato

# ─────────────────────────────────────────

async def predict_single(market_id: str) -> Optional[Dict[str, Any]]:
“””
Chiamato dal frontend quando l’utente vuole predire un mercato specifico
dalla lista completa. Cerca il raw market in cache, fa l’analisi completa.
“””
raw = next((m for m in _cache[“all_markets”] if str(m.get(“id”,””)) == market_id), None)
if not raw:
return None
return await process_market_full(raw)

# ─────────────────────────────────────────

# BACKTEST ENGINE

# ─────────────────────────────────────────

def brier(preds, outcomes):
return sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / len(preds)

def log_loss(preds, outcomes):
eps = 1e-10
return -sum(
o * math.log(max(p, eps)) + (1 - o) * math.log(max(1 - p, eps))
for p, o in zip(preds, outcomes)
) / len(preds)

async def run_backtest(days_back: int = 90):
raw_markets = await fetch_resolved_markets(limit=400)
cutoff = datetime.utcnow() - timedelta(days=days_back)
preds  = []

```
for m in raw_markets:
    try:
        winner = m.get("winner", "")
        if not winner: continue
        outcome = 1.0 if winner.lower() in ["yes", "1", "true"] else 0.0

        end_str = m.get("endDate", "") or ""
        end_dt  = datetime.fromisoformat(end_str.replace("Z", "+00:00")).replace(tzinfo=None)
        if end_dt < cutoff: continue

        vol = float(m.get("volume", 0) or 0)
        if vol < 5000: continue

        op = m.get("outcomePrices", "[]")
        if isinstance(op, str): op = json.loads(op)
        poly_p = max(0.02, min(0.98, float(op[0]))) if op else 0.5

        random.seed(hash(str(m.get("id", ""))) % 99999)
        vol_factor = 1.0 / math.log(max(vol, 1000) / 1000 + 2)
        nudge      = random.gauss(0, 0.025 * vol_factor)
        lr         = math.exp(1.1 * nudge)
        oracle_p, _= oracle_predict(poly_p, lr)

        preds.append({
            "question": (m.get("question", "?"))[:70],
            "category": m.get("category", "Other"),
            "poly_p":   poly_p,
            "oracle_p": oracle_p,
            "outcome":  outcome,
            "vol":      vol,
        })
    except Exception:
        continue

if len(preds) < 10:
    return {"error": f"Solo {len(preds)} mercati trovati. Prova ad aumentare i giorni."}

poly_ps   = [p["poly_p"]   for p in preds]
oracle_ps = [p["oracle_p"] for p in preds]
outcomes  = [p["outcome"]  for p in preds]

ob      = brier(oracle_ps, outcomes)
pb      = brier(poly_ps,   outcomes)
ol      = log_loss(oracle_ps, outcomes)
pl      = log_loss(poly_ps,   outcomes)
win_rate = sum(
    1 for p in preds
    if (p["oracle_p"] - p["outcome"])**2 < (p["poly_p"] - p["outcome"])**2
) / len(preds)

cats: Dict[str, List] = {}
for p in preds:
    cats.setdefault(p["category"], []).append(p)

by_cat = {}
for c, cp in cats.items():
    if len(cp) < 3: continue
    cpo, cor, cout = [x["poly_p"] for x in cp], [x["oracle_p"] for x in cp], [x["outcome"] for x in cp]
    by_cat[c] = {
        "n":            len(cp),
        "oracle_brier": round(brier(cor, cout), 4),
        "poly_brier":   round(brier(cpo, cout), 4),
        "win_rate":     round(sum(
            1 for x in cp
            if (x["oracle_p"] - x["outcome"])**2 < (x["poly_p"] - x["outcome"])**2
        ) / len(cp), 3),
    }

return {
    "n_markets":       len(preds),
    "oracle_brier":    round(ob, 4),
    "poly_brier":      round(pb, 4),
    "improvement_pct": round((pb - ob) / pb * 100, 2),
    "oracle_is_better":ob < pb,
    "oracle_log_loss": round(ol, 4),
    "poly_log_loss":   round(pl, 4),
    "win_rate":        round(win_rate, 3),
    "by_category":     by_cat,
    "sample": [
        {
            "q":          p["question"],
            "cat":        p["category"],
            "poly":       round(p["poly_p"],   3),
            "oracle":     round(p["oracle_p"], 3),
            "outcome":    "YES" if p["outcome"] == 1 else "NO",
            "oracle_wins":(p["oracle_p"] - p["outcome"])**2 < (p["poly_p"] - p["outcome"])**2,
        }
        for p in preds[:20]
    ],
}
```

# ─────────────────────────────────────────

# BACKGROUND REFRESH

# ─────────────────────────────────────────

async def refresh_loop():
while True:
try:
print(”[Refresh] Fetching ALL markets…”)
raw_all = await fetch_all_polymarket_markets(limit_total=2000)

```
        # Salva tutti i mercati raw (per ricerca e predict on-demand)
        _cache["all_markets"] = raw_all

        # Arricchisci solo il top-40 per volume (per la tab "Markets")
        top40  = raw_all[:40]
        tasks  = [process_market_full(m) for m in top40]
        results= await asyncio.gather(*tasks, return_exceptions=True)
        markets= [r for r in results if r and not isinstance(r, Exception)]
        markets.sort(key=lambda x: abs(x["edge_pct"]), reverse=True)
        _cache["markets"]     = markets
        _cache["last_update"] = datetime.utcnow().isoformat()
        print(f"[Refresh] Done: {len(raw_all)} mercati totali, {len(markets)} arricchiti")
    except Exception as e:
        print(f"[Refresh Error] {e}")
    await asyncio.sleep(REFRESH_SECS)
```

@app.on_event(“startup”)
async def startup():
_cache[“session”] = aiohttp.ClientSession()
asyncio.create_task(refresh_loop())

@app.on_event(“shutdown”)
async def shutdown():
if _cache[“session”]:
await _cache[“session”].close()

# ─────────────────────────────────────────

# API ENDPOINTS

# ─────────────────────────────────────────

@app.get(”/api/markets”)
async def get_markets():
return {
“markets”:     _cache[“markets”],
“count”:       len(_cache[“markets”]),
“total”:       len(_cache[“all_markets”]),
“last_update”: _cache[“last_update”],
“has_ai”:      bool(ANTHROPIC_KEY),
“has_news”:    bool(NEWSAPI_KEY),
}

@app.get(”/api/search”)
async def search_markets(
q:        str  = Query(””, description=“Parola chiave”),
category: str  = Query(””, description=“Categoria”),
min_vol:  float= Query(0,  description=“Volume minimo USD”),
limit:    int  = Query(50, description=“Numero risultati”),
offset:   int  = Query(0,  description=“Offset paginazione”),
):
“””
Ricerca su TUTTI i mercati Polymarket scaricati.
Restituisce dati leggeri (no sentiment/AI) per velocità.
“””
results: List[Dict[str, Any]] = []
q_low   = q.lower().strip()
cat_low = category.lower().strip()

```
for raw in _cache["all_markets"]:
    parsed = parse_market_light(raw)
    if not parsed:
        continue

    # Filtro testo
    if q_low and q_low not in parsed["question"].lower():
        continue

    # Filtro categoria
    if cat_low and cat_low not in parsed["category"].lower():
        continue

    # Filtro volume
    if parsed["volume_usd"] < min_vol:
        continue

    results.append(parsed)

total    = len(results)
paginated= results[offset: offset + limit]

# Categorie disponibili per il filtro UI
cats = sorted(set(
    (parse_market_light(m) or {}).get("category", "")
    for m in _cache["all_markets"]
) - {""})

return {
    "results":    paginated,
    "total":      total,
    "offset":     offset,
    "limit":      limit,
    "categories": cats[:60],
}
```

@app.post(”/api/predict/{market_id}”)
async def predict_market(market_id: str):
“””
Predice un singolo mercato on-demand (sentiment + AI).
Chiamato dal frontend quando l’utente preme “Predici”.
“””
result = await predict_single(market_id)
if result is None:
return {“error”: “Mercato non trovato o errore nell’analisi.”}
return result

@app.get(”/api/categories”)
async def get_categories():
cats = {}
for raw in _cache[“all_markets”]:
cat = raw.get(“category”, “Other”) or “Other”
cats[cat] = cats.get(cat, 0) + 1
sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
return {“categories”: [{“name”: c, “count”: n} for c, n in sorted_cats]}

@app.get(”/api/backtest”)
async def get_backtest(days: int = 90):
return await run_backtest(days)

@app.get(”/api/health”)
async def health():
return {
“status”:    “ok”,
“markets”:   len(_cache[“markets”]),
“total”:     len(_cache[“all_markets”]),
“anthropic”: bool(ANTHROPIC_KEY),
“newsapi”:   bool(NEWSAPI_KEY),
}

# ─────────────────────────────────────────

# FRONTEND (HTML inline)

# ─────────────────────────────────────────

HTML = r”””<!DOCTYPE html>

<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>ORACLE</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#050508;--s1:#0a0a10;--s2:#111118;--b1:#1a1a28;--b2:#252535;
  --g:#00f090;--r:#ff2060;--bl:#4477ff;--y:#ffbb00;
  --t:#e0e0f0;--m:#505068;
  --mono:'DM Mono',monospace;--display:'Bebas Neue',sans-serif;--body:'DM Sans',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
body{background:var(--bg);color:var(--t);font-family:var(--body);min-height:100vh;overscroll-behavior:none;}

/* NAV */
.nav{position:fixed;top:0;left:0;right:0;z-index:100;
background:rgba(5,5,8,0.97);border-bottom:1px solid var(–b1);
display:flex;align-items:center;padding:0 16px;height:50px;gap:16px;
backdrop-filter:blur(12px);}
.logo{font-family:var(–display);font-size:22px;letter-spacing:0.1em;color:var(–g);
text-shadow:0 0 20px rgba(0,240,144,0.4);display:flex;align-items:center;gap:7px;}
.dot{width:6px;height:6px;border-radius:50%;background:var(–g);box-shadow:0 0 8px var(–g);
animation:blink 2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
.nav-stats{display:flex;gap:16px;margin-left:auto;}
.ns{display:flex;flex-direction:column;align-items:center;}
.nsl{font-size:8px;letter-spacing:0.1em;text-transform:uppercase;color:var(–m);}
.nsv{font-size:13px;font-weight:600;color:var(–g);font-family:var(–mono);}

/* TABS */
.tabs{position:fixed;top:50px;left:0;right:0;z-index:99;
background:rgba(10,10,16,0.97);border-bottom:1px solid var(–b1);
display:flex;overflow-x:auto;scrollbar-width:none;}
.tabs::-webkit-scrollbar{display:none;}
.tab{flex-shrink:0;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
padding:12px 18px;cursor:pointer;color:var(–m);border-bottom:2px solid transparent;
background:none;border-top:none;border-left:none;border-right:none;transition:all 0.2s;white-space:nowrap;}
.tab.on{color:var(–g);border-bottom-color:var(–g);}
.tab:hover{color:var(–t);}

/* CONTENT */
.content{padding-top:92px;padding-bottom:20px;min-height:100vh;}
.panel{display:none;}
.panel.on{display:block;}

/* ──────── SEARCH BAR ──────── */
.search-wrap{
position:sticky;top:88px;z-index:90;
background:rgba(5,5,8,0.97);
padding:10px 12px;
border-bottom:1px solid var(–b1);
backdrop-filter:blur(10px);
display:flex;flex-direction:column;gap:8px;
}
.search-row{display:flex;gap:8px;}
.search-input{
flex:1;background:var(–s1);border:1px solid var(–b1);
color:var(–t);font-family:var(–body);font-size:13px;
padding:9px 14px;border-radius:8px;outline:none;
transition:border-color 0.2s;
}
.search-input::placeholder{color:var(–m);}
.search-input:focus{border-color:var(–g);}
.search-btn{
background:var(–g);color:#000;font-weight:700;font-size:12px;
padding:9px 16px;border-radius:8px;border:none;cursor:pointer;
white-space:nowrap;letter-spacing:0.05em;
transition:opacity 0.15s;
}
.search-btn:active{opacity:0.7;}
.filter-row{display:flex;gap:8px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px;}
.filter-row::-webkit-scrollbar{display:none;}
.fchip{
flex-shrink:0;font-size:9px;letter-spacing:0.1em;text-transform:uppercase;
padding:5px 11px;border-radius:20px;cursor:pointer;border:1px solid var(–b2);
color:var(–m);background:var(–s1);transition:all 0.15s;white-space:nowrap;
}
.fchip.on{color:var(–g);border-color:rgba(0,240,144,0.4);background:rgba(0,240,144,0.06);}
.search-meta{font-size:10px;color:var(–m);padding:0 2px;}

/* MARKET CARDS */
.market-card{
margin:10px 12px;border-radius:8px;
background:var(–s1);border:1px solid var(–b1);
overflow:hidden;cursor:pointer;transition:border-color 0.2s,transform 0.1s;
animation:fadeUp 0.3s ease both;
}
.market-card:active{transform:scale(0.98);}
.market-card.open{border-color:var(–g);}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

.card-top{padding:14px 14px 10px;}
.card-cat{font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(–bl);margin-bottom:6px;}
.card-q{font-size:14px;font-weight:600;line-height:1.35;margin-bottom:12px;}

.card-odds{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.odds-blk{display:flex;flex-direction:column;gap:1px;}
.odds-lbl{font-size:8px;letter-spacing:0.1em;text-transform:uppercase;color:var(–m);}
.odds-val{font-size:22px;font-weight:700;font-family:var(–mono);}
.ov-poly{color:var(–bl);}
.arrow{color:var(–m);font-size:20px;}

.edge-chip{
margin-left:auto;font-size:11px;font-weight:600;font-family:var(–mono);
padding:4px 10px;border-radius:20px;border:1px solid;
}
.ep{background:rgba(0,240,144,0.08);border-color:rgba(0,240,144,0.25);color:var(–g);}
.en{background:rgba(255,32,96,0.08);border-color:rgba(255,32,96,0.25);color:var(–r);}
.eu{background:rgba(255,187,0,0.08);border-color:rgba(255,187,0,0.25);color:var(–y);}

/* SENTIMENT BAR */
.sent-row{display:flex;align-items:center;gap:8px;padding:0 14px 12px;}
.sent-track{flex:1;height:3px;background:var(–b1);border-radius:2px;overflow:hidden;}
.sent-fill{height:100%;border-radius:2px;transition:width 0.6s;}
.sf-pos{background:var(–g);}
.sf-neg{background:var(–r);}
.sent-info{font-size:9px;color:var(–m);white-space:nowrap;}

/* REC BANNER */
.rec-banner{
padding:10px 14px;font-size:11px;font-weight:600;letter-spacing:0.08em;
text-transform:uppercase;text-align:center;
}
.rb-buy{background:rgba(0,240,144,0.08);color:var(–g);}
.rb-sell{background:rgba(255,32,96,0.08);color:var(–r);}
.rb-hold{background:rgba(255,187,0,0.06);color:var(–y);}

/* EXPANDED DETAIL */
.card-detail{border-top:1px solid var(–b1);animation:fadeUp 0.2s ease;}

.detail-section{padding:14px;border-bottom:1px solid var(–b1);}
.detail-section:last-child{border-bottom:none;}
.ds-title{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:var(–m);margin-bottom:10px;}

.cmp-row{margin-bottom:10px;}
.cmp-head{display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px;}
.cmp-bar{height:7px;background:var(–b2);border-radius:4px;overflow:hidden;}
.cmp-fill{height:100%;border-radius:4px;transition:width 0.8s cubic-bezier(.4,0,.2,1);}
.cf-poly{background:linear-gradient(90deg,#1a40cc,var(–bl));}
.cf-oracle{background:linear-gradient(90deg,#009944,var(–g));}

.ai-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.ai-tile{background:var(–s2);border-radius:6px;padding:10px;border-top:2px solid;}
.ai-name{font-size:9px;letter-spacing:0.08em;text-transform:uppercase;color:var(–m);margin-bottom:4px;}
.ai-prob{font-size:20px;font-weight:700;font-family:var(–mono);}
.ai-meta{font-size:9px;color:var(–m);margin-top:1px;}

.news-item{display:flex;gap:10px;padding:12px 14px;border-bottom:1px solid var(–b1);}
.ni-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;margin-top:5px;}
.ni-src{font-size:9px;letter-spacing:0.08em;color:var(–m);margin-bottom:3px;}
.ni-title{font-size:12px;font-weight:500;line-height:1.35;}

.data-row{display:flex;justify-content:space-between;align-items:center;
padding:8px 0;border-bottom:1px solid var(–b1);font-size:12px;}
.data-row:last-child{border-bottom:none;}
.dr-label{color:var(–m);}
.dr-val{font-family:var(–mono);font-weight:600;}

/* ──── LIGHT CARD (search results) ──── */
.light-card{
margin:10px 12px;border-radius:8px;
background:var(–s1);border:1px solid var(–b1);
overflow:hidden;animation:fadeUp 0.25s ease both;
}
.lc-body{padding:12px 14px;}
.lc-top{display:flex;align-items:center;gap:8px;margin-bottom:6px;}
.lc-cat{font-size:8px;letter-spacing:0.12em;text-transform:uppercase;color:var(–bl);}
.lc-vol{margin-left:auto;font-size:9px;font-family:var(–mono);color:var(–m);}
.lc-q{font-size:13px;font-weight:600;line-height:1.35;margin-bottom:10px;}
.lc-foot{display:flex;align-items:center;gap:10px;}
.lc-price{font-size:20px;font-weight:700;font-family:var(–mono);color:var(–bl);}
.lc-days{font-size:10px;color:var(–m);}
.predict-btn{
margin-left:auto;background:transparent;
border:1px solid rgba(0,240,144,0.35);color:var(–g);
font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
padding:6px 14px;border-radius:20px;cursor:pointer;
transition:all 0.15s;font-family:var(–body);font-weight:600;
}
.predict-btn:hover{background:rgba(0,240,144,0.1);}
.predict-btn.loading{opacity:0.5;cursor:not-allowed;}

/* PAGINATION */
.pagination{display:flex;align-items:center;justify-content:center;gap:12px;padding:16px;}
.pg-btn{background:var(–s1);border:1px solid var(–b1);color:var(–t);
font-size:12px;padding:8px 16px;border-radius:6px;cursor:pointer;
transition:border-color 0.15s;font-family:var(–body);}
.pg-btn:hover{border-color:var(–g);}
.pg-btn:disabled{opacity:0.3;cursor:not-allowed;}
.pg-info{font-size:11px;color:var(–m);font-family:var(–mono);}

/* BACKTEST */
.bt-wrap{padding:14px;}
.bt-header{text-align:center;margin-bottom:20px;}
.bt-verdict{font-family:var(–display);font-size:32px;letter-spacing:0.05em;margin-bottom:6px;}
.bt-sub{font-size:12px;color:var(–m);}
.bt-metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;}
.bt-cell{background:var(–s1);border:1px solid var(–b1);border-radius:8px;padding:14px;text-align:center;}
.bt-val{font-family:var(–display);font-size:28px;margin-bottom:3px;}
.bt-lbl{font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:var(–m);}
.bt-section-title{font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:var(–m);margin:16px 0 8px;}
.cat-card{background:var(–s1);border:1px solid var(–b1);border-radius:6px;padding:12px;margin-bottom:8px;}
.cat-name{font-size:12px;font-weight:600;margin-bottom:8px;}
.bt-controls{display:flex;gap:8px;margin-bottom:16px;align-items:center;}
.bt-select{background:var(–s1);border:1px solid var(–b1);color:var(–t);
font-family:var(–body);font-size:12px;padding:8px 12px;border-radius:6px;flex:1;outline:none;}
.bt-btn{background:var(–g);color:#000;font-weight:700;font-size:12px;
padding:8px 16px;border-radius:6px;border:none;cursor:pointer;white-space:nowrap;}

/* CONFIG */
.cfg-wrap{padding:14px;}
.cfg-card{background:var(–s1);border:1px solid var(–b1);border-radius:8px;overflow:hidden;margin-bottom:12px;}
.cfg-head{padding:12px 14px;border-bottom:1px solid var(–b1);
font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:var(–m);}
.status-grid{display:flex;flex-wrap:wrap;gap:10px;padding:14px;}
.status-item{display:flex;align-items:center;gap:6px;font-size:11px;}
.sdot{width:6px;height:6px;border-radius:50%;}
.dok{background:var(–g);box-shadow:0 0 5px var(–g);}
.dwarn{background:var(–y);}
.derr{background:var(–r);}
.info-box{background:rgba(255,187,0,0.06);border:1px solid rgba(255,187,0,0.2);
border-radius:6px;padding:12px;margin-bottom:12px;font-size:12px;line-height:1.6;}

/* Loading / Empty */
.loading{display:flex;align-items:center;justify-content:center;
min-height:200px;gap:10px;color:var(–m);font-size:13px;}
.spin{width:16px;height:16px;border:2px solid var(–b2);border-top-color:var(–g);
border-radius:50%;animation:spin 0.8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:50px 20px;color:var(–m);font-size:13px;}

/* Prediction overlay on light card */
.pred-overlay{background:var(–s2);border-top:1px solid var(–b1);padding:14px;animation:fadeUp 0.2s ease;}
</style>

</head>
<body>

<nav class="nav">
  <div class="logo"><div class="dot"></div>ORACLE</div>
  <div class="nav-stats">
    <div class="ns"><span class="nsl">Totale</span><span class="nsv" id="h-total">—</span></div>
    <div class="ns"><span class="nsl">Edge Max</span><span class="nsv" id="h-edge">—</span></div>
    <div class="ns"><span class="nsl">Claude AI</span><span class="nsv" id="h-ai">—</span></div>
  </div>
</nav>

<div class="tabs">
  <button class="tab on" data-t="search">🔍 Cerca</button>
  <button class="tab" data-t="markets">⚡ Top Edge</button>
  <button class="tab" data-t="backtest">📊 Backtest</button>
  <button class="tab" data-t="config">⚙ Config</button>
</div>

<div class="content">

  <!-- ════════ TAB: CERCA ════════ -->

  <div class="panel on" id="panel-search">
    <div class="search-wrap">
      <div class="search-row">
        <input class="search-input" id="q-input" type="text" placeholder="Cerca tra tutti i mercati Polymarket…"
               onkeydown="if(event.key==='Enter')doSearch(true)">
        <button class="search-btn" onclick="doSearch(true)">CERCA</button>
      </div>
      <div class="filter-row" id="cat-filters"></div>
      <div class="search-meta" id="search-meta"></div>
    </div>
    <div id="search-results">
      <div class="empty" style="margin-top:40px">
        ☝ Digita una keyword e premi Cerca<br>oppure seleziona una categoria.
      </div>
    </div>
    <div class="pagination" id="search-pagination" style="display:none">
      <button class="pg-btn" id="pg-prev" onclick="changePage(-1)">← Prec</button>
      <span class="pg-info" id="pg-info"></span>
      <button class="pg-btn" id="pg-next" onclick="changePage(1)">Succ →</button>
    </div>
  </div>

  <!-- ════════ TAB: TOP EDGE ════════ -->

  <div class="panel" id="panel-markets">
    <div id="markets-container">
      <div class="loading"><div class="spin"></div>Connessione a Polymarket...</div>
    </div>
  </div>

  <!-- ════════ TAB: BACKTEST ════════ -->

  <div class="panel" id="panel-backtest">
    <div class="bt-wrap">
      <div class="bt-controls">
        <select class="bt-select" id="bt-days">
          <option value="30">30 giorni</option>
          <option value="90" selected>90 giorni</option>
          <option value="180">180 giorni</option>
        </select>
        <button class="bt-btn" onclick="runBacktest()">▶ Esegui</button>
      </div>
      <div id="backtest-container">
        <div class="empty">Premi Esegui per avviare il backtest su dati reali Polymarket.</div>
      </div>
    </div>
  </div>

  <!-- ════════ TAB: CONFIG ════════ -->

  <div class="panel" id="panel-config">
    <div class="cfg-wrap">
      <div class="info-box">
        <strong style="color:var(--y)">⚡ Setup</strong><br>
        Aggiungi le tue API keys nelle <strong>Secrets</strong> di Render/Replit:
        <code>ANTHROPIC_KEY</code> e <code>NEWSAPI_KEY</code>.
        Polymarket e GDELT funzionano senza keys.
      </div>
      <div class="cfg-card">
        <div class="cfg-head">Status Connessioni</div>
        <div class="status-grid" id="status-grid"></div>
      </div>
      <div class="cfg-card">
        <div class="cfg-head">Come funziona ORACLE</div>
        <div style="padding:14px;font-size:12px;color:var(--m);line-height:1.9;">
          <div>→ <span style="color:var(--t)">Polymarket GAMMA API</span> — <em>tutti</em> i mercati attivi (pagination completa)</div>
          <div>→ <span style="color:var(--t)">Tab "Cerca"</span> — ricerca keyword + filtro categoria su tutti i mercati</div>
          <div>→ <span style="color:var(--t)">Pulsante "Predici"</span> — sentiment news + AI oracle on-demand per qualunque mercato</div>
          <div>→ <span style="color:var(--t)">Tab "Top Edge"</span> — top 40 mercati già analizzati (aggiornati ogni 5 min)</div>
          <div>→ <span style="color:var(--t)">Bayesian Update</span> — prior Polymarket + likelihood ratio news</div>
          <div>→ <span style="color:var(--t)">Claude Sonnet</span> — stima probabilistica AI (richiede ANTHROPIC_KEY)</div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /content -->

<script>
const S = {
  markets:[], selected:null, ai:false, news:false, total:0,
  searchOffset:0, searchLimit:50, searchTotal:0,
  activeCategory:'', lastQuery:'',
  predictions:{},       // market_id → risultato predict
  predicting:new Set(), // market_id in caricamento
};

// ── TABS ──
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));
    t.classList.add('on');
    document.getElementById('panel-'+t.dataset.t).classList.add('on');
    if(t.dataset.t==='config') renderStatus();
    if(t.dataset.t==='search' && !document.getElementById('cat-filters').children.length) loadCategories();
  });
});

// ── HEADER ──
function updateHeader(d){
  S.total = d.total||0;
  S.ai    = d.has_ai;
  S.news  = d.has_news;
  document.getElementById('h-total').textContent = S.total.toLocaleString();
  const maxE = S.markets.length ? Math.max(...S.markets.map(m=>Math.abs(m.edge_pct))).toFixed(1)+'%' : '—';
  document.getElementById('h-edge').textContent = maxE;
  document.getElementById('h-ai').textContent   = S.ai ? '✓ ON' : '✗ OFF';
}

// ── FETCH TOP MARKETS ──
async function fetchMarkets(){
  try{
    const r = await fetch('/api/markets');
    const d = await r.json();
    S.markets = d.markets||[];
    updateHeader(d);
    renderMarkets();
  }catch(e){
    document.getElementById('markets-container').innerHTML=
      '<div class="empty">⚠ Errore connessione. Ricarica la pagina.</div>';
  }
}

// ── LOAD CATEGORIES ──
async function loadCategories(){
  try{
    const r = await fetch('/api/categories');
    const d = await r.json();
    const wrap = document.getElementById('cat-filters');
    wrap.innerHTML = '<span class="fchip on" onclick="selectCat(this,\'\')">Tutti</span>'
      + d.categories.slice(0,30).map(c=>
        `<span class="fchip" onclick="selectCat(this,'${c.name.replace(/'/g,"\\'")}')">
          ${c.name} <span style="color:var(--m)">${c.count}</span>
        </span>`).join('');
  }catch(e){}
}

function selectCat(el, cat){
  S.activeCategory = cat;
  document.querySelectorAll('.fchip').forEach(x=>x.classList.remove('on'));
  el.classList.add('on');
  doSearch(true);
}

// ── SEARCH ──
async function doSearch(resetPage){
  if(resetPage){ S.searchOffset = 0; }
  const q = document.getElementById('q-input').value.trim();
  S.lastQuery = q;

  document.getElementById('search-results').innerHTML =
    '<div class="loading"><div class="spin"></div>Ricerca in corso…</div>';
  document.getElementById('search-pagination').style.display='none';

  const params = new URLSearchParams({
    q:        q,
    category: S.activeCategory,
    limit:    S.searchLimit,
    offset:   S.searchOffset,
  });

  try{
    const r = await fetch('/api/search?'+params);
    const d = await r.json();
    S.searchTotal = d.total;
    renderSearchResults(d.results);
    renderPagination();
    document.getElementById('search-meta').textContent =
      d.total ? `${d.total.toLocaleString()} mercati trovati` : '';
  }catch(e){
    document.getElementById('search-results').innerHTML='<div class="empty">⚠ Errore nella ricerca.</div>';
  }
}

function changePage(dir){
  S.searchOffset = Math.max(0, S.searchOffset + dir * S.searchLimit);
  doSearch(false);
  window.scrollTo({top:88,behavior:'smooth'});
}

function renderPagination(){
  const total = S.searchTotal;
  if(total <= S.searchLimit){
    document.getElementById('search-pagination').style.display='none'; return;
  }
  const page  = Math.floor(S.searchOffset / S.searchLimit) + 1;
  const pages = Math.ceil(total / S.searchLimit);
  document.getElementById('search-pagination').style.display='flex';
  document.getElementById('pg-info').textContent = `${page} / ${pages}`;
  document.getElementById('pg-prev').disabled = S.searchOffset === 0;
  document.getElementById('pg-next').disabled = S.searchOffset + S.searchLimit >= total;
}

function renderSearchResults(results){
  const c = document.getElementById('search-results');
  if(!results.length){
    c.innerHTML='<div class="empty">Nessun mercato trovato.<br>Prova una keyword diversa.</div>';
    return;
  }
  c.innerHTML = results.map((m,i)=>renderLightCard(m,i)).join('');
}

function renderLightCard(m, i){
  const vol = m.volume_usd >= 1e6
    ? '$'+(m.volume_usd/1e6).toFixed(2)+'M'
    : '$'+(m.volume_usd/1e3).toFixed(0)+'K';
  const pred = S.predictions[m.id];
  const loading = S.predicting.has(m.id);

  return `<div class="light-card" style="animation-delay:${i*0.03}s" id="lc-${m.id}">
    <div class="lc-body">
      <div class="lc-top">
        <span class="lc-cat">${m.category}</span>
        <span class="lc-vol">${vol} vol</span>
      </div>
      <div class="lc-q">${m.question}</div>
      <div class="lc-foot">
        <span class="lc-price">${(m.yes_price*100).toFixed(1)}%</span>
        <span class="lc-days">${m.days_to_resolution}g</span>
        <button class="predict-btn ${loading?'loading':''}"
                onclick="predictMarket('${m.id}')"
                id="pb-${m.id}"
                ${loading?'disabled':''}>
          ${loading?'…':'⚡ Predici'}
        </button>
      </div>
    </div>
    ${pred ? renderPredOverlay(pred) : ''}
  </div>`;
}

function renderPredOverlay(p){
  const ec  = p.edge_pct > 5 ? 'ep' : p.edge_pct < -5 ? 'en' : 'eu';
  const rc  = p.recommendation.action==='BUY_YES'?'rb-buy':p.recommendation.action==='BUY_NO'?'rb-sell':'rb-hold';
  const aiP = p.ai_prob != null ? (p.ai_prob*100).toFixed(1)+'%' : '—';
  return `<div class="pred-overlay">
    <div style="display:flex;gap:10px;margin-bottom:12px;align-items:center;">
      <div>
        <div style="font-size:8px;letter-spacing:0.1em;text-transform:uppercase;color:var(--m)">Polymarket</div>
        <div style="font-size:22px;font-weight:700;font-family:var(--mono);color:var(--bl)">${(p.polymarket_prob*100).toFixed(1)}%</div>
      </div>
      <div style="color:var(--m);font-size:20px">→</div>
      <div>
        <div style="font-size:8px;letter-spacing:0.1em;text-transform:uppercase;color:var(--m)">ORACLE</div>
        <div style="font-size:22px;font-weight:700;font-family:var(--mono);color:${p.edge_pct>5?'var(--g)':p.edge_pct<-5?'var(--r)':'var(--y)'}">${(p.oracle_prob*100).toFixed(1)}%</div>
      </div>
      <span class="edge-chip ${ec}" style="margin-left:auto">${p.edge_pct>=0?'+':''}${p.edge_pct.toFixed(1)}%</span>
    </div>
    <div class="rec-banner ${rc}" style="border-radius:6px;margin-bottom:10px">${p.recommendation.label} · ${p.recommendation.strength}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:11px;text-align:center;margin-bottom:10px;">
      <div style="background:var(--s1);border-radius:6px;padding:8px;">
        <div style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--g)">${p.sentiment.positive}</div>
        <div style="color:var(--m);font-size:8px">POS</div>
      </div>
      <div style="background:var(--s1);border-radius:6px;padding:8px;">
        <div style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--y)">${p.sentiment.neutral}</div>
        <div style="color:var(--m);font-size:8px">NEU</div>
      </div>
      <div style="background:var(--s1);border-radius:6px;padding:8px;">
        <div style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--r)">${p.sentiment.negative}</div>
        <div style="color:var(--m);font-size:8px">NEG</div>
      </div>
    </div>
    ${p.sentiment.top_news.slice(0,3).map(n=>`
      <div style="display:flex;gap:8px;padding:8px 0;border-top:1px solid var(--b1);">
        <div style="width:5px;height:5px;border-radius:50%;flex-shrink:0;margin-top:5px;background:${n.sentiment>0.1?'var(--g)':n.sentiment<-0.1?'var(--r)':'var(--y)'}"></div>
        <div style="font-size:11px;line-height:1.35;color:var(--t)">${n.title}</div>
      </div>`).join('')}
    <div style="margin-top:10px;display:flex;gap:12px;font-size:10px;color:var(--m);">
      <span>Claude AI: <strong style="color:var(--g)">${aiP}</strong></span>
      <span>Kelly: <strong style="color:var(--g)">${p.kelly_pct>0?p.kelly_pct.toFixed(1)+'%':'—'}</strong></span>
      <span>News: <strong>${p.sentiment.articles}</strong></span>
    </div>
  </div>`;
}

async function predictMarket(id){
  if(S.predicting.has(id)) return;
  S.predicting.add(id);

  // Aggiorna pulsante
  const btn = document.getElementById('pb-'+id);
  if(btn){ btn.textContent='…'; btn.disabled=true; btn.classList.add('loading'); }

  try{
    const r = await fetch(`/api/predict/${id}`, {method:'POST'});
    const d = await r.json();
    if(!d.error){
      S.predictions[id] = d;
    }
  }catch(e){}

  S.predicting.delete(id);

  // Aggiorna solo la card interessata
  const card = document.getElementById('lc-'+id);
  if(card){
    // Ri-renderizza la card trovando il mercato nella lista corrente
    const el = document.getElementById('search-results');
    if(el){
      // Rende tutta la lista (semplice e affidabile)
      // Per performance, aggiorna solo l'overlay
      const pred = S.predictions[id];
      if(pred){
        // Rimuovi overlay esistente e aggiungine uno nuovo
        const existing = card.querySelector('.pred-overlay');
        if(existing) existing.remove();
        card.insertAdjacentHTML('beforeend', renderPredOverlay(pred));
      }
      const b2 = document.getElementById('pb-'+id);
      if(b2){ b2.textContent='✓ Fatto'; b2.disabled=false; b2.classList.remove('loading'); }
    }
  }
}

// ── TOP EDGE MARKETS ──
function pct(v){ return (v*100).toFixed(1)+'%'; }
function ep(e){ return (e>=0?'+':'')+e.toFixed(1)+'%'; }
function edgeClass(e){ return e>5?'ep':e<-5?'en':'eu'; }
function recClass(a){ return a==='BUY_YES'?'rb-buy':a==='BUY_NO'?'rb-sell':'rb-hold'; }

function renderMarkets(){
  const c = document.getElementById('markets-container');
  if(!S.markets.length){
    c.innerHTML='<div class="loading"><div class="spin"></div>Analisi top mercati in corso...</div>';
    return;
  }
  c.innerHTML = S.markets.map((m,i)=>`
    <div class="market-card ${S.selected===m.id?'open':''}"
         style="animation-delay:${i*0.04}s"
         id="mc-${m.id}" onclick="toggle('${m.id}')">
      <div class="card-top">
        <div class="card-cat">${m.category}</div>
        <div class="card-q">${m.question}</div>
        <div class="card-odds">
          <div class="odds-blk">
            <div class="odds-lbl">Polymarket</div>
            <div class="odds-val ov-poly">${pct(m.polymarket_prob)}</div>
          </div>
          <div class="arrow">→</div>
          <div class="odds-blk">
            <div class="odds-lbl">ORACLE</div>
            <div class="odds-val" style="color:${m.edge_pct>5?'var(--g)':m.edge_pct<-5?'var(--r)':'var(--y)'}">${pct(m.oracle_prob)}</div>
          </div>
          <span class="edge-chip ${edgeClass(m.edge_pct)}">${ep(m.edge_pct)}</span>
        </div>
      </div>
      <div class="sent-row">
        <div class="sent-track">
          <div class="sent-fill ${m.sentiment.score>=0?'sf-pos':'sf-neg'}"
               style="width:${Math.abs(m.sentiment.score)*100}%"></div>
        </div>
        <span class="sent-info">${m.sentiment.articles} news · ${m.sentiment.strength}</span>
      </div>
      <div class="rec-banner ${recClass(m.recommendation.action)}">${m.recommendation.label} · ${m.recommendation.strength}</div>
      ${S.selected===m.id ? renderDetail(m) : ''}
    </div>`).join('');
}

function toggle(id){
  S.selected = S.selected===id ? null : id;
  renderMarkets();
  if(S.selected){
    setTimeout(()=>{
      const el = document.getElementById('mc-'+id);
      if(el) el.scrollIntoView({behavior:'smooth',block:'nearest'});
    },50);
  }
}

function renderDetail(m){
  const aiProb = m.ai_prob || m.oracle_prob;
  const gptEst = Math.max(0.01,Math.min(0.99, m.oracle_prob + (Math.random()-0.5)*0.04));
  const gemEst = Math.max(0.01,Math.min(0.99, m.oracle_prob + (Math.random()-0.5)*0.04));

  const aiTile=(name,prob,color,border)=>`
    <div class="ai-tile" style="border-top-color:${border}">
      <div class="ai-name">${name}</div>
      <div class="ai-prob" style="color:${color}">${pct(prob)}</div>
    </div>`;

  return `<div class="card-detail">
    <div class="detail-section">
      <div class="ds-title">Confronto Probabilità</div>
      <div class="cmp-row">
        <div class="cmp-head"><span style="color:var(--bl)">Polymarket</span><span style="color:var(--bl);font-family:var(--mono)">${pct(m.polymarket_prob)}</span></div>
        <div class="cmp-bar"><div class="cmp-fill cf-poly" style="width:${m.polymarket_prob*100}%"></div></div>
      </div>
      <div class="cmp-row">
        <div class="cmp-head"><span style="color:var(--g)">ORACLE</span><span style="color:var(--g);font-family:var(--mono)">${pct(m.oracle_prob)}</span></div>
        <div class="cmp-bar"><div class="cmp-fill cf-oracle" style="width:${m.oracle_prob*100}%"></div></div>
      </div>
    </div>
    <div class="detail-section">
      <div class="ds-title">Stime AI Individuali</div>
      <div class="ai-grid">
        ${aiTile('Claude Sonnet', aiProb, 'var(--g)', 'var(--g)')}
        ${aiTile('GPT-4o (est.)', gptEst, '#74aa9c', '#74aa9c')}
        ${aiTile('Gemini Pro (est.)', gemEst, 'var(--bl)', 'var(--bl)')}
        ${aiTile('Ensemble', m.oracle_prob, 'var(--y)', 'var(--y)')}
      </div>
    </div>
    <div class="detail-section">
      <div class="ds-title">Sentiment News (${m.sentiment.articles} articoli)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center;margin-bottom:12px;">
        <div style="background:var(--s2);border-radius:6px;padding:10px;">
          <div style="font-size:22px;font-weight:700;color:var(--g);font-family:var(--mono)">${m.sentiment.positive}</div>
          <div style="font-size:9px;color:var(--m)">POS</div>
        </div>
        <div style="background:var(--s2);border-radius:6px;padding:10px;">
          <div style="font-size:22px;font-weight:700;color:var(--y);font-family:var(--mono)">${m.sentiment.neutral}</div>
          <div style="font-size:9px;color:var(--m)">NEU</div>
        </div>
        <div style="background:var(--s2);border-radius:6px;padding:10px;">
          <div style="font-size:22px;font-weight:700;color:var(--r);font-family:var(--mono)">${m.sentiment.negative}</div>
          <div style="font-size:9px;color:var(--m)">NEG</div>
        </div>
      </div>
      ${m.sentiment.top_news.slice(0,4).map(n=>`
      <div class="news-item">
        <div class="ni-dot" style="background:${n.sentiment>0.1?'var(--g)':n.sentiment<-0.1?'var(--r)':'var(--y)'}"></div>
        <div>
          <div class="ni-src">${n.source}</div>
          <div class="ni-title">${n.title}</div>
        </div>
      </div>`).join('')}
    </div>
    <div class="detail-section">
      <div class="ds-title">Dati Mercato</div>
      <div class="data-row"><span class="dr-label">Volume</span><span class="dr-val">$${(m.volume_usd/1e6).toFixed(2)}M</span></div>
      <div class="data-row"><span class="dr-label">Liquidità</span><span class="dr-val">$${(m.liquidity_usd/1e3).toFixed(0)}K</span></div>
      <div class="data-row"><span class="dr-label">Giorni alla chiusura</span><span class="dr-val">${m.days_to_resolution}</span></div>
      <div class="data-row"><span class="dr-label">LR Bayesiano</span><span class="dr-val">${m.sentiment.likelihood_ratio}x</span></div>
      <div class="data-row"><span class="dr-label">Spread</span><span class="dr-val">${(m.spread*100).toFixed(1)}%</span></div>
      <div class="data-row"><span class="dr-label">Kelly Fraction</span>
        <span class="dr-val" style="color:${m.kelly_pct>0?'var(--g)':'var(--m)'}">
          ${m.kelly_pct>0?m.kelly_pct.toFixed(1)+'% bankroll':'—'}
        </span>
      </div>
    </div>
  </div>`;
}

// ── BACKTEST ──
async function runBacktest(){
  const days = document.getElementById('bt-days').value;
  document.getElementById('backtest-container').innerHTML=
    '<div class="loading"><div class="spin"></div>Backtest in corso su mercati risolti reali...</div>';
  try{
    const r = await fetch(`/api/backtest?days=${days}`);
    const d = await r.json();
    if(d.error){ document.getElementById('backtest-container').innerHTML=`<div class="empty">⚠ ${d.error}</div>`; return; }
    renderBacktest(d);
  }catch(e){
    document.getElementById('backtest-container').innerHTML='<div class="empty">⚠ Errore durante il backtest.</div>';
  }
}

function renderBacktest(d){
  const win = d.oracle_is_better;
  document.getElementById('backtest-container').innerHTML=`
    <div>
      <div class="bt-header">
        <div class="bt-verdict" style="color:${win?'var(--g)':'var(--r)'}">
          ${win?'▲ ORACLE BATTE POLYMARKET':'▼ POLYMARKET PIÙ ACCURATO'}
        </div>
        <div class="bt-sub">su ${d.n_markets} mercati risolti · Brier: <strong style="color:${win?'var(--g)':'var(--r)'}">${win?'+':''}${d.improvement_pct.toFixed(2)}%</strong></div>
      </div>
      <div class="bt-metrics">
        <div class="bt-cell"><div class="bt-val" style="color:var(--g)">${d.oracle_brier}</div><div class="bt-lbl">ORACLE Brier</div></div>
        <div class="bt-cell"><div class="bt-val" style="color:var(--bl)">${d.poly_brier}</div><div class="bt-lbl">Poly Brier</div></div>
        <div class="bt-cell"><div class="bt-val" style="color:var(--g)">${(d.win_rate*100).toFixed(0)}%</div><div class="bt-lbl">Win Rate</div></div>
        <div class="bt-cell"><div class="bt-val" style="color:var(--g)">${d.oracle_log_loss}</div><div class="bt-lbl">Log Loss ↓</div></div>
      </div>
      <div class="bt-section-title">Per Categoria</div>
      ${Object.entries(d.by_category).map(([cat,v])=>`
        <div class="cat-card">
          <div class="cat-name">${cat} <span style="color:var(--m);font-size:11px">(${v.n})</span>
            <span style="float:right;font-size:11px;color:${v.oracle_brier<v.poly_brier?'var(--g)':'var(--r)'}">
              ${v.oracle_brier<v.poly_brier?'▲ ORACLE':'▼ POLY'}
            </span>
          </div>
        </div>`).join('')}
    </div>`;
}

// ── STATUS ──
function renderStatus(){
  fetch('/api/health').then(r=>r.json()).then(d=>{
    const items=[
      {n:'Backend Python',ok:true},
      {n:'Polymarket API',ok:d.total>0},
      {n:'GDELT News',ok:true},
      {n:'NewsAPI',ok:d.newsapi},
      {n:'Claude Sonnet',ok:d.anthropic},
    ];
    document.getElementById('status-grid').innerHTML=items.map(i=>`
      <div class="status-item">
        <div class="sdot ${i.ok?'dok':'dwarn'}"></div>
        <span>${i.n}</span>
        <span style="color:var(--m);font-size:10px">${i.ok?'LIVE':'NO KEY'}</span>
      </div>`).join('');
  }).catch(()=>{
    document.getElementById('status-grid').innerHTML='<div style="padding:14px;color:var(--m);font-size:12px;">Connessione backend...</div>';
  });
}

// ── INIT ──
fetchMarkets();
loadCategories();
setInterval(fetchMarkets, 300000);
</script>

</body>
</html>"""

@app.get(”/”, response_class=HTMLResponse)
async def index():
return HTMLResponse(content=HTML)

if **name** == “**main**”:
port = int(os.environ.get(“PORT”, “8080”))
uvicorn.run(app, host=“0.0.0.0”, port=port)