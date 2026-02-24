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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

REFRESH_SECS = int(os.environ.get("REFRESH_SECS", "120"))  # più frequente
MAX_MARKETS = int(os.environ.get("MAX_MARKETS", "6000"))   # prova 6000, alza se vuoi
LIVE_TIMEOUT = float(os.environ.get("LIVE_TIMEOUT", "10"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, Any] = {
    "markets": [],       # top edge (arricchiti)
    "all_markets": [],   # tutti raw (cache)
    "last_update": None,
    "session": None,
    "lock": asyncio.Lock(),
}

POLY_GAMMA = "https://gamma-api.polymarket.com"


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def now_iso() -> str:
    return datetime.utcnow().isoformat()


# ─────────────────────────────────────────
# POLYMARKET — FETCH ALL + LIVE BY ID
# ─────────────────────────────────────────

async def fetch_all_polymarket_markets(limit_total: int = MAX_MARKETS) -> List[Dict[str, Any]]:
    """
    Scarica TUTTI i mercati attivi da Polymarket GAMMA API usando pagination.
    Ritorna fino a limit_total (metti alto).
    """
    url = f"{POLY_GAMMA}/markets"
    all_m: List[Dict[str, Any]] = []
    offset = 0
    page_size = 100

    while len(all_m) < limit_total:
        params = {
            "active": "true",
            "closed": "false",
            "limit": str(page_size),
            "offset": str(offset),
            "order": "volume",
            "ascending": "false",
        }
        try:
            async with _cache["session"].get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    break
                data = await r.json()
                batch = data if isinstance(data, list) else data.get("markets", [])
                if not batch:
                    break
                all_m.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
        except Exception as e:
            print(f"[Polymarket page offset={offset}] {e}")
            break

    print(f"[Polymarket] Cache markets: {len(all_m)}")
    return all_m[:limit_total]


async def fetch_market_live(market_id: str) -> Optional[Dict[str, Any]]:
    """
    Quote live per UNO specifico market_id (per “percentuali precise”).
    Gamma a volte cambia formato: gestiamo più tentativi.
    """
    # Tentativo 1: /markets/{id}
    url1 = f"{POLY_GAMMA}/markets/{market_id}"
    try:
        async with _cache["session"].get(url1, timeout=aiohttp.ClientTimeout(total=LIVE_TIMEOUT)) as r:
            if r.status == 200:
                return await r.json()
    except Exception:
        pass

    # Tentativo 2: /markets?id=...
    url2 = f"{POLY_GAMMA}/markets"
    try:
        async with _cache["session"].get(url2, params={"id": market_id}, timeout=aiohttp.ClientTimeout(total=LIVE_TIMEOUT)) as r:
            if r.status == 200:
                data = await r.json()
                if isinstance(data, list) and data:
                    return data[0]
                if isinstance(data, dict):
                    ms = data.get("markets", [])
                    if ms:
                        return ms[0]
    except Exception:
        pass

    return None


async def fetch_resolved_markets(limit: int = 400) -> List[Dict[str, Any]]:
    url = f"{POLY_GAMMA}/markets"
    params = {
        "closed": "true",
        "resolved": "true",
        "limit": str(limit),
        "order": "volume",
        "ascending": "false",
    }
    try:
        async with _cache["session"].get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status == 200:
                data = await r.json()
                return data if isinstance(data, list) else data.get("markets", [])
    except Exception as e:
        print(f"[Polymarket resolved] {e}")
    return []


# ─────────────────────────────────────────
# NEWS + SENTIMENT (semplice e veloce)
# ─────────────────────────────────────────

POSITIVE_WORDS = {
    "win","wins","won","victory","lead","leads","ahead","surge","gain","approve",
    "approved","pass","passed","signed","confirm","elect","elected","majority",
    "support","likely","probable","expected","strengthen","coalition","polling",
    "frontrunner","positive","momentum","growing","optimistic","strong","recover"
}
NEGATIVE_WORDS = {
    "lose","loses","lost","defeat","fail","fails","failed","reject","rejected",
    "veto","block","blocked","oppose","opposed","collapse","crisis","resign",
    "unlikely","improbable","impossible","behind","trailing","drop","fall","decrease",
    "decline","opposition","stall","deadlock","breakdown","sanction"
}
NEGATIONS = {"not", "no", "never", "neither", "nor", "without"}

def analyze_sentiment(text: str) -> Tuple[float, float]:
    words = re.findall(r"\b\w+\b", (text or "").lower())
    score = 0.0
    hits = 0
    for i, w in enumerate(words):
        ctx = words[max(0, i - 3): i]
        neg = any(n in ctx for n in NEGATIONS)
        base = 0.0
        if w in POSITIVE_WORDS:
            base = 0.7
        elif w in NEGATIVE_WORDS:
            base = -0.7
        if base:
            score += base * (-0.8 if neg else 1.0)
            hits += 1
    sent = clamp(score / max(hits, 1), -1.0, 1.0)
    conf = min(1.0, hits / max(len(words) * 0.05, 1))
    return sent, conf

def extract_query(question: str) -> str:
    q = re.sub(r"^(will|who will|which|what|when|is|are)\s+", "", (question or ""), flags=re.IGNORECASE)
    return " ".join(q.split()[:8])

async def fetch_gdelt_news(query: str, days: int = 7) -> List[Dict[str, Any]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": "30",
        "format": "json",
        "startdatetime": since,
        "enddatetime": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }
    try:
        async with _cache["session"].get(url, params=params, timeout=aiohttp.ClientTimeout(total=12)) as r:
            if r.status == 200:
                data = await r.json()
                return data.get("articles", [])
    except Exception:
        pass
    return []

async def fetch_newsapi(query: str, days: int = 7) -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": "30",
        "from": since,
    }
    try:
        async with _cache["session"].get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                data = await r.json()
                return data.get("articles", [])
    except Exception:
        pass
    return []


async def get_ai_probability(question: str, news_context: str) -> Optional[float]:
    if not ANTHROPIC_KEY:
        return None
    prompt = (
        "You are an expert forecaster for prediction markets. Be well-calibrated.\n\n"
        f"Question: {question}\n\n"
        f"Recent news:\n{(news_context or '')[:2000]}\n\n"
        "Respond with ONLY a decimal between 0.01 and 0.99 representing P(YES). No explanation."
    )
    try:
        async with _cache["session"].post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            if r.status == 200:
                d = await r.json()
                txt = d["content"][0]["text"].strip()
                m = re.search(r"(0?\.\d+|1\.0+|\d?\.\d+)", txt)
                if m:
                    return clamp(float(m.group()), 0.02, 0.98)
    except Exception as e:
        print(f"[AI] {e}")
    return None


async def analyze_market_sentiment(question: str):
    query = extract_query(question)
    gdelt, newsapi = await asyncio.gather(
        fetch_gdelt_news(query),
        fetch_newsapi(query),
        return_exceptions=True,
    )

    articles: List[Dict[str, Any]] = []

    for raw in (gdelt if isinstance(gdelt, list) else []):
        title = raw.get("title", "")
        if not title:
            continue
        s, c = analyze_sentiment(title)
        q_words = set(re.findall(r"\b\w{3,}\b", (question or "").lower()))
        t_words = set(re.findall(r"\b\w{3,}\b", title.lower()))
        rel = min(1.0, len(q_words & t_words) / max(len(q_words), 1) * 2)

        age_h = 0.0
        try:
            ts = datetime.strptime(raw.get("seendate", "")[:15], "%Y%m%dT%H%M%S")
            age_h = (datetime.utcnow() - ts).total_seconds() / 3600
        except Exception:
            pass
        recency = math.exp(-0.693 * age_h / 12)

        articles.append({
            "title": title,
            "sentiment": s,
            "confidence": c,
            "relevance": rel,
            "recency": recency,
            "source": raw.get("domain", ""),
        })

    for raw in (newsapi if isinstance(newsapi, list) else []):
        title = (raw.get("title", "") or "")
        desc = (raw.get("description", "") or "")
        txt = f"{title}. {desc}".strip()
        if not txt:
            continue
        s, c = analyze_sentiment(txt)
        q_words = set(re.findall(r"\b\w{3,}\b", (question or "").lower()))
        t_words = set(re.findall(r"\b\w{3,}\b", txt.lower()))
        rel = min(1.0, len(q_words & t_words) / max(len(q_words), 1) * 2)
        articles.append({
            "title": title,
            "sentiment": s,
            "confidence": c,
            "relevance": rel,
            "recency": 1.0,
            "source": raw.get("source", {}).get("name", ""),
        })

    if not articles:
        return 0.0, 1.0, 0, [], ""

    total_w = sum(a["recency"] * a["relevance"] * max(0.1, a["confidence"]) for a in articles)
    weighted_sent = (
        sum(a["sentiment"] * a["recency"] * a["relevance"] * max(0.1, a["confidence"]) for a in articles)
        / max(total_w, 1e-9)
    )

    k = min(1.0, len(articles) / 20)
    lr = math.exp(1.1 * k * weighted_sent)
    lr = clamp(lr, 0.2, 5.0)

    news_context = "\n".join(a["title"] for a in articles[:15] if a.get("title"))
    return weighted_sent, lr, len(articles), articles[:8], news_context


# ─────────────────────────────────────────
# ORACLE PREDICTOR (Bayesian)
# ─────────────────────────────────────────

def bayesian_update(prior: float, lr: float) -> float:
    prior = clamp(prior, 0.02, 0.98)
    odds = prior / (1 - prior)
    post = odds * lr
    return post / (1 + post)

def oracle_predict(
    poly_prob: float,
    sentiment_lr: float,
    ai_prob: Optional[float] = None,
    velocity: float = 0.0,
    vol_spike: float = 1.0,
):
    prior = clamp(poly_prob, 0.02, 0.98)
    posterior = bayesian_update(prior, sentiment_lr)

    if ai_prob is not None:
        w_ai = 0.28
        lp = math.log((posterior / (1 - posterior + 1e-9)) + 1e-9)
        la = math.log((ai_prob / (1 - ai_prob + 1e-9)) + 1e-9)
        posterior = 1 / (1 + math.exp(-(lp * (1 - w_ai) + la * w_ai)))

    if abs(velocity) > 0.04:
        vel_lr = clamp(math.exp(1.5 * velocity), 0.5, 2.0)
        posterior = bayesian_update(posterior, vel_lr) * 0.12 + posterior * 0.88

    if vol_spike > 2.0:
        posterior = 0.5 + (posterior - 0.5) * min(1.4, vol_spike / 2)

    posterior = clamp(posterior, 0.01, 0.99)
    edge = posterior - prior
    confidence = min(0.95, 0.5 + abs(edge) * 3)
    return posterior, confidence

def get_recommendation(edge_pct: float, confidence: float):
    if edge_pct > 5 and confidence > 0.55:
        return {"action": "BUY_YES", "label": "▲ COMPRA YES", "strength": "FORTE" if edge_pct > 10 else "MODERATO"}
    if edge_pct < -5 and confidence > 0.55:
        return {"action": "BUY_NO", "label": "▼ COMPRA NO", "strength": "FORTE" if edge_pct < -10 else "MODERATO"}
    return {"action": "HOLD", "label": "◆ OSSERVA", "strength": "NEUTRO"}


# ─────────────────────────────────────────
# MARKET PARSING
# ─────────────────────────────────────────

def _parse_outcome_prices(raw: Any) -> List[float]:
    op = raw if raw is not None else []
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except Exception:
            op = []
    if not isinstance(op, list):
        op = []
    out = []
    for x in op[:2]:
        out.append(clamp(safe_float(x, 0.5), 0.01, 0.99))
    return out

def parse_market_light(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        op = _parse_outcome_prices(m.get("outcomePrices", []))
        yes_price = op[0] if op else 0.5

        vol = safe_float(m.get("volume", 0), 0.0)
        liq = safe_float(m.get("liquidity", 0), 0.0)

        end_str = m.get("endDate", "") or ""
        days_left = 30
        try:
            end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00")).replace(tzinfo=None)
            days_left = max(0, (end_date - datetime.utcnow()).days)
        except Exception:
            pass

        tags = m.get("tags", [])
        if isinstance(tags, list):
            tags = [t.get("label", "") if isinstance(t, dict) else str(t) for t in tags]

        cat = m.get("category", "Other") or "Other"
        q = m.get("question", "?") or "?"

        return {
            "id": str(m.get("id", "")),
            "question": q,
            "category": cat,
            "yes_price": round(yes_price, 4),
            "volume_usd": vol,
            "liquidity_usd": liq,
            "days_to_resolution": days_left,
            "tags": tags[:4],
        }
    except Exception:
        return None

def parse_market_full_inputs(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        op = _parse_outcome_prices(m.get("outcomePrices", []))
        yes_price = op[0] if op else 0.5

        no_price = op[1] if len(op) > 1 else (1 - yes_price)
        spread = abs((1 - yes_price) - no_price)

        vol = safe_float(m.get("volume", 0), 0.0)
        liq = safe_float(m.get("liquidity", 0), 0.0)

        end_str = m.get("endDate", "") or ""
        days_left = 30
        try:
            end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00")).replace(tzinfo=None)
            days_left = max(0, (end_date - datetime.utcnow()).days)
        except Exception:
            pass

        tags = m.get("tags", [])
        if isinstance(tags, list):
            tags = [t.get("label", "") if isinstance(t, dict) else str(t) for t in tags]

        return {
            "id": str(m.get("id", "")),
            "question": (m.get("question", "?") or "?"),
            "category": (m.get("category", "Other") or "Other"),
            "yes_price": clamp(yes_price, 0.01, 0.99),
            "volume": vol,
            "liquidity": liq,
            "days_left": days_left,
            "tags": tags[:4],
            "spread": spread if op else 0.05,
        }
    except Exception:
        return None


async def process_market_full(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = parse_market_full_inputs(raw)
    if not m:
        return None

    sent_score, lr, n_articles, top_articles, news_ctx = await analyze_market_sentiment(m["question"])

    # Velocity (placeholder): puoi sostituirla con una vera misura se hai timeseries
    random.seed(hash(m["id"]) % 99999)
    velocity = random.gauss(0, 0.015)

    ai_prob = await get_ai_probability(m["question"], news_ctx) if news_ctx else None
    oracle_prob, confidence = oracle_predict(m["yes_price"], lr, ai_prob, velocity)

    edge = oracle_prob - m["yes_price"]
    edge_pct = edge * 100

    b = max(0.0, 1 / m["yes_price"] - 1) if m["yes_price"] > 0 else 0.0
    p = oracle_prob
    kelly = max(0.0, (p * b - (1 - p)) / b) * 0.5 if b > 0 else 0.0
    kelly = min(kelly, 0.20)

    return {
        "id": m["id"],
        "question": m["question"],
        "category": m["category"],
        "polymarket_prob": round(m["yes_price"], 4),
        "oracle_prob": round(oracle_prob, 4),
        "edge": round(edge, 4),
        "edge_pct": round(edge_pct, 2),
        "confidence": round(confidence, 3),
        "volume_usd": m["volume"],
        "liquidity_usd": m["liquidity"],
        "days_to_resolution": m["days_left"],
        "tags": m["tags"],
        "spread": round(m["spread"], 4),
        "kelly_pct": round(kelly * 100, 1),
        "recommendation": get_recommendation(edge_pct, confidence),
        "sentiment": {
            "score": round(sent_score, 3),
            "likelihood_ratio": round(lr, 3),
            "articles": n_articles,
            "positive": sum(1 for a in top_articles if a["sentiment"] > 0.1),
            "negative": sum(1 for a in top_articles if a["sentiment"] < -0.1),
            "neutral": sum(1 for a in top_articles if abs(a["sentiment"]) <= 0.1),
            "strength": "FORTE" if abs(sent_score) > 0.35 else "MODERATO" if abs(sent_score) > 0.15 else "DEBOLE",
            "top_news": [
                {"title": a["title"], "source": a["source"], "sentiment": round(a["sentiment"], 2)}
                for a in top_articles[:5]
            ],
        },
        "ai_prob": round(ai_prob, 3) if ai_prob is not None else None,
        "velocity_24h": round(velocity, 4),
        "last_updated": now_iso(),
    }


# ─────────────────────────────────────────
# PREDICT (sempre LIVE per market_id)
# ─────────────────────────────────────────

async def predict_single_live(market_id: str) -> Optional[Dict[str, Any]]:
    raw_live = await fetch_market_live(market_id)
    if not raw_live:
        # fallback: prova cache
        raw_live = next((m for m in _cache["all_markets"] if str(m.get("id", "")) == str(market_id)), None)
    if not raw_live:
        return None
    return await process_market_full(raw_live)


# ─────────────────────────────────────────
# BACKTEST ENGINE (uguale alla tua idea)
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
    preds = []

    for m in raw_markets:
        try:
            winner = m.get("winner", "")
            if not winner:
                continue
            outcome = 1.0 if str(winner).lower() in ["yes", "1", "true"] else 0.0

            end_str = m.get("endDate", "") or ""
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00")).replace(tzinfo=None)
            if end_dt < cutoff:
                continue

            vol = safe_float(m.get("volume", 0), 0.0)
            if vol < 5000:
                continue

            op = _parse_outcome_prices(m.get("outcomePrices", []))
            poly_p = clamp(op[0] if op else 0.5, 0.02, 0.98)

            random.seed(hash(str(m.get("id", ""))) % 99999)
            vol_factor = 1.0 / math.log(max(vol, 1000) / 1000 + 2)
            nudge = random.gauss(0, 0.025 * vol_factor)
            lr = math.exp(1.1 * nudge)

            oracle_p, _ = oracle_predict(poly_p, lr)

            preds.append({
                "question": (m.get("question", "?") or "?")[:70],
                "category": m.get("category", "Other") or "Other",
                "poly_p": poly_p,
                "oracle_p": oracle_p,
                "outcome": outcome,
                "vol": vol,
            })
        except Exception:
            continue

    if len(preds) < 10:
        return {"error": f"Solo {len(preds)} mercati trovati. Prova ad aumentare i giorni."}

    poly_ps = [p["poly_p"] for p in preds]
    oracle_ps = [p["oracle_p"] for p in preds]
    outcomes = [p["outcome"] for p in preds]

    ob = brier(oracle_ps, outcomes)
    pb = brier(poly_ps, outcomes)
    ol = log_loss(oracle_ps, outcomes)
    pl = log_loss(poly_ps, outcomes)

    win_rate = sum(
        1 for p in preds
        if (p["oracle_p"] - p["outcome"]) ** 2 < (p["poly_p"] - p["outcome"]) ** 2
    ) / len(preds)

    cats: Dict[str, List] = {}
    for p in preds:
        cats.setdefault(p["category"], []).append(p)

    by_cat = {}
    for c, cp in cats.items():
        if len(cp) < 3:
            continue
        cpo = [x["poly_p"] for x in cp]
        cor = [x["oracle_p"] for x in cp]
        cout = [x["outcome"] for x in cp]
        by_cat[c] = {
            "n": len(cp),
            "oracle_brier": round(brier(cor, cout), 4),
            "poly_brier": round(brier(cpo, cout), 4),
            "win_rate": round(sum(
                1 for x in cp
                if (x["oracle_p"] - x["outcome"]) ** 2 < (x["poly_p"] - x["outcome"]) ** 2
            ) / len(cp), 3),
        }

    return {
        "n_markets": len(preds),
        "oracle_brier": round(ob, 4),
        "poly_brier": round(pb, 4),
        "improvement_pct": round((pb - ob) / pb * 100, 2),
        "oracle_is_better": ob < pb,
        "oracle_log_loss": round(ol, 4),
        "poly_log_loss": round(pl, 4),
        "win_rate": round(win_rate, 3),
        "by_category": by_cat,
        "sample": [
            {
                "q": p["question"],
                "cat": p["category"],
                "poly": round(p["poly_p"], 3),
                "oracle": round(p["oracle_p"], 3),
                "outcome": "YES" if p["outcome"] == 1 else "NO",
                "oracle_wins": (p["oracle_p"] - p["outcome"]) ** 2 < (p["poly_p"] - p["outcome"]) ** 2,
            }
            for p in preds[:20]
        ],
    }


# ─────────────────────────────────────────
# BACKGROUND REFRESH (cache globale)
# ─────────────────────────────────────────

async def refresh_loop():
    while True:
        try:
            print("[Refresh] Fetching ALL markets…")
            raw_all = await fetch_all_polymarket_markets(limit_total=MAX_MARKETS)

            async with _cache["lock"]:
                _cache["all_markets"] = raw_all

            # Top40 (per volume) + analisi (edge)
            top40 = raw_all[:40]
            tasks = [process_market_full(m) for m in top40]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            markets = [r for r in results if r and not isinstance(r, Exception)]
            markets.sort(key=lambda x: abs(x["edge_pct"]), reverse=True)

            async with _cache["lock"]:
                _cache["markets"] = markets
                _cache["last_update"] = now_iso()

            print(f"[Refresh] Done: {len(raw_all)} totali, {len(markets)} arricchiti")
        except Exception as e:
            print(f"[Refresh Error] {e}")

        await asyncio.sleep(REFRESH_SECS)

@app.on_event("startup")
async def startup():
    _cache["session"] = aiohttp.ClientSession()
    asyncio.create_task(refresh_loop())

@app.on_event("shutdown")
async def shutdown():
    if _cache["session"]:
        await _cache["session"].close()


# ─────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────

@app.get("/api/markets")
async def get_markets():
    async with _cache["lock"]:
        markets = _cache["markets"]
        total = len(_cache["all_markets"])
        last_update = _cache["last_update"]
    return {
        "markets": markets,
        "count": len(markets),
        "total": total,
        "last_update": last_update,
        "has_ai": bool(ANTHROPIC_KEY),
        "has_news": bool(NEWSAPI_KEY),
        "refresh_secs": REFRESH_SECS,
    }

@app.get("/api/categories")
async def get_categories():
    async with _cache["lock"]:
        allm = list(_cache["all_markets"])
    cats = {}
    for raw in allm:
        cat = raw.get("category", "Other") or "Other"
        cats[cat] = cats.get(cat, 0) + 1
    sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
    return {"categories": [{"name": c, "count": n} for c, n in sorted_cats]}

@app.get("/api/search")
async def search_markets(
    q: str = Query("", description="Parola chiave"),
    category: str = Query("", description="Categoria"),
    min_vol: float = Query(0, description="Volume minimo USD"),
    limit: int = Query(50, description="Numero risultati"),
    offset: int = Query(0, description="Offset paginazione"),
    live: int = Query(1, description="1 = aggiorna quote live per i risultati mostrati"),
):
    """
    Cerca nel cache (veloce), ma se live=1 aggiorna LE QUOTE dei risultati mostrati con fetch live.
    """
    q_low = q.lower().strip()
    cat_low = category.lower().strip()

    async with _cache["lock"]:
        allm = list(_cache["all_markets"])

    results: List[Dict[str, Any]] = []
    for raw in allm:
        parsed = parse_market_light(raw)
        if not parsed:
            continue
        if q_low and q_low not in parsed["question"].lower():
            continue
        if cat_low and cat_low not in parsed["category"].lower():
            continue
        if parsed["volume_usd"] < min_vol:
            continue
        results.append(parsed)

    total = len(results)
    paginated = results[offset: offset + limit]

    # Quote live precise per questa pagina
    if live == 1 and paginated:
        ids = [x["id"] for x in paginated]
        lives = await asyncio.gather(*[fetch_market_live(i) for i in ids], return_exceptions=True)
        live_map = {}
        for i, raw_live in zip(ids, lives):
            if raw_live and not isinstance(raw_live, Exception):
                pl = parse_market_light(raw_live)
                if pl:
                    live_map[i] = pl
        paginated = [live_map.get(x["id"], x) for x in paginated]

    cats = sorted(set(
        (parse_market_light(m) or {}).get("category", "")
        for m in allm
    ) - {""})

    return {
        "results": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
        "categories": cats[:60],
    }

@app.get("/api/quote/{market_id}")
async def quote_market(market_id: str):
    raw = await fetch_market_live(market_id)
    if not raw:
        return JSONResponse({"error": "Market not found"}, status_code=404)
    parsed = parse_market_light(raw)
    if not parsed:
        return JSONResponse({"error": "Parse error"}, status_code=500)
    parsed["last_updated"] = now_iso()
    return parsed

@app.get("/api/quotes")
async def quote_markets(ids: str = Query("", description="csv: 123,456,789")):
    market_ids = [x.strip() for x in ids.split(",") if x.strip()]
    market_ids = market_ids[:40]  # sicurezza
    if not market_ids:
        return {"quotes": []}
    lives = await asyncio.gather(*[fetch_market_live(i) for i in market_ids], return_exceptions=True)
    out = []
    for mid, raw_live in zip(market_ids, lives):
        if raw_live and not isinstance(raw_live, Exception):
            pl = parse_market_light(raw_live)
            if pl:
                pl["last_updated"] = now_iso()
                out.append(pl)
    return {"quotes": out}

@app.post("/api/predict/{market_id}")
async def predict_market(market_id: str):
    result = await predict_single_live(market_id)
    if result is None:
        return {"error": "Mercato non trovato o errore nell’analisi."}
    return result

@app.get("/api/backtest")
async def get_backtest(days: int = 90):
    return await run_backtest(days)

@app.get("/api/health")
async def health():
    async with _cache["lock"]:
        total = len(_cache["all_markets"])
        markets = len(_cache["markets"])
    return {
        "status": "ok",
        "markets": markets,
        "total": total,
        "anthropic": bool(ANTHROPIC_KEY),
        "newsapi": bool(NEWSAPI_KEY),
        "refresh_secs": REFRESH_SECS,
        "max_markets": MAX_MARKETS,
    }


# ─────────────────────────────────────────
# FRONTEND (HTML inline)
# ─────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
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
background:rgba(5,5,8,0.97);border-bottom:1px solid var(--b1);
display:flex;align-items:center;padding:0 16px;height:50px;gap:16px;
backdrop-filter:blur(12px);}
.logo{font-family:var(--display);font-size:22px;letter-spacing:0.1em;color:var(--g);
text-shadow:0 0 20px rgba(0,240,144,0.4);display:flex;align-items:center;gap:7px;}
.dot{width:6px;height:6px;border-radius:50%;background:var(--g);box-shadow:0 0 8px var(--g);
animation:blink 2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
.nav-stats{display:flex;gap:16px;margin-left:auto;}
.ns{display:flex;flex-direction:column;align-items:center;}
.nsl{font-size:8px;letter-spacing:0.1em;text-transform:uppercase;color:var(--m);}
.nsv{font-size:13px;font-weight:600;color:var(--g);font-family:var(--mono);}

/* TABS */
.tabs{position:fixed;top:50px;left:0;right:0;z-index:99;
background:rgba(10,10,16,0.97);border-bottom:1px solid var(--b1);
display:flex;overflow-x:auto;scrollbar-width:none;}
.tabs::-webkit-scrollbar{display:none;}
.tab{flex-shrink:0;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
padding:12px 18px;cursor:pointer;color:var(--m);border-bottom:2px solid transparent;
background:none;border:none;transition:all 0.2s;white-space:nowrap;}
.tab.on{color:var(--g);border-bottom-color:var(--g);}
.tab:hover{color:var(--t);}

/* CONTENT */
.content{padding-top:92px;padding-bottom:24px;min-height:100vh;}
.panel{display:none;}
.panel.on{display:block;}

/* SEARCH BAR */
.search-wrap{
position:sticky;top:88px;z-index:90;
background:rgba(5,5,8,0.97);
padding:10px 12px;
border-bottom:1px solid var(--b1);
backdrop-filter:blur(10px);
display:flex;flex-direction:column;gap:8px;
}
.search-row{display:flex;gap:8px;}
.search-input{
flex:1;background:var(--s1);border:1px solid var(--b1);
color:var(--t);font-family:var(--body);font-size:13px;
padding:9px 14px;border-radius:8px;outline:none;
transition:border-color 0.2s;
}
.search-input::placeholder{color:var(--m);}
.search-input:focus{border-color:var(--g);}
.search-btn{
background:var(--g);color:#000;font-weight:800;font-size:12px;
padding:9px 16px;border-radius:8px;border:none;cursor:pointer;
white-space:nowrap;letter-spacing:0.05em;
transition:opacity 0.15s;
}
.search-btn:active{opacity:0.7;}
.filter-row{display:flex;gap:8px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px;}
.filter-row::-webkit-scrollbar{display:none;}
.fchip{
flex-shrink:0;font-size:9px;letter-spacing:0.1em;text-transform:uppercase;
padding:5px 11px;border-radius:20px;cursor:pointer;border:1px solid var(--b2);
color:var(--m);background:var(--s1);transition:all 0.15s;white-space:nowrap;
}
.fchip.on{color:var(--g);border-color:rgba(0,240,144,0.4);background:rgba(0,240,144,0.06);}
.search-meta{font-size:10px;color:var(--m);padding:0 2px;}

/* CARDS */
.card{
margin:10px 12px;border-radius:10px;
background:var(--s1);border:1px solid var(--b1);
overflow:hidden;animation:fadeUp 0.25s ease both;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.c-top{padding:14px;}
.c-cat{font-size:9px;letter-spacing:0.14em;text-transform:uppercase;color:var(--bl);margin-bottom:6px;}
.c-q{font-size:14px;font-weight:700;line-height:1.35;margin-bottom:10px;}
.c-row{display:flex;align-items:center;gap:10px;}
.pill{
font-size:11px;font-family:var(--mono);font-weight:700;
padding:5px 10px;border-radius:999px;border:1px solid var(--b2);color:var(--t);
}
.p-poly{border-color:rgba(68,119,255,0.3);background:rgba(68,119,255,0.08);color:var(--bl);}
.p-orc{border-color:rgba(0,240,144,0.3);background:rgba(0,240,144,0.08);color:var(--g);}
.p-warn{border-color:rgba(255,187,0,0.3);background:rgba(255,187,0,0.07);color:var(--y);}
.btn{
margin-left:auto;background:transparent;border:1px solid rgba(0,240,144,0.35);color:var(--g);
font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
padding:7px 14px;border-radius:999px;cursor:pointer;
transition:all 0.15s;font-family:var(--body);font-weight:800;
}
.btn:hover{background:rgba(0,240,144,0.08);}
.btn.red{border-color:rgba(255,32,96,0.35);color:var(--r);}
.btn.red:hover{background:rgba(255,32,96,0.08);}
.btn.loading{opacity:0.5;cursor:not-allowed;}
.small{font-size:10px;color:var(--m);padding:0 14px 14px;display:flex;gap:10px;}

/* PRED OVERLAY */
.pred{
border-top:1px solid var(--b1);
background:var(--s2);
padding:14px;
}
.pred-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;}
.tile{background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:10px;}
.tl{font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:var(--m);margin-bottom:4px;}
.tv{font-size:22px;font-weight:900;font-family:var(--mono);}
.news{margin-top:8px;border-top:1px solid var(--b1);}
.news .n{padding:10px 0;border-bottom:1px solid var(--b1);font-size:11px;line-height:1.35;}
.news .n:last-child{border-bottom:none;}
hr.sep{border:none;border-top:1px solid var(--b1);margin:12px 0;}

.loading{display:flex;align-items:center;justify-content:center;min-height:180px;gap:10px;color:var(--m);font-size:13px;}
.spin{width:16px;height:16px;border:2px solid var(--b2);border-top-color:var(--g);border-radius:50%;animation:spin 0.8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:50px 20px;color:var(--m);font-size:13px;}
.pagination{display:flex;align-items:center;justify-content:center;gap:12px;padding:16px;}
.pg-btn{background:var(--s1);border:1px solid var(--b1);color:var(--t);
font-size:12px;padding:8px 16px;border-radius:8px;cursor:pointer;}
.pg-btn:disabled{opacity:0.35;cursor:not-allowed;}
.pg-info{font-size:11px;color:var(--m);font-family:var(--mono);}
</style>
</head>
<body>

<nav class="nav">
  <div class="logo"><div class="dot"></div>ORACLE</div>
  <div class="nav-stats">
    <div class="ns"><span class="nsl">Totale</span><span class="nsv" id="h-total">—</span></div>
    <div class="ns"><span class="nsl">Agg.</span><span class="nsv" id="h-upd">—</span></div>
    <div class="ns"><span class="nsl">AI</span><span class="nsv" id="h-ai">—</span></div>
  </div>
</nav>

<div class="tabs">
  <button class="tab on" data-t="saved">⭐ Salvati</button>
  <button class="tab" data-t="search">🔍 Cerca</button>
  <button class="tab" data-t="top">⚡ Top Edge</button>
  <button class="tab" data-t="backtest">📊 Backtest</button>
  <button class="tab" data-t="config">⚙ Config</button>
</div>

<div class="content">

  <!-- SAVED -->
  <div class="panel on" id="panel-saved">
    <div id="saved-container">
      <div class="empty" style="margin-top:20px">Nessun mercato salvato.<br>Vai su <strong>Cerca</strong> e premi “Salva”.</div>
    </div>
  </div>

  <!-- SEARCH -->
  <div class="panel" id="panel-search">
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
      <div class="empty" style="margin-top:40px">☝ Digita una keyword e premi Cerca<br>oppure seleziona una categoria.</div>
    </div>
    <div class="pagination" id="search-pagination" style="display:none">
      <button class="pg-btn" id="pg-prev" onclick="changePage(-1)">← Prec</button>
      <span class="pg-info" id="pg-info"></span>
      <button class="pg-btn" id="pg-next" onclick="changePage(1)">Succ →</button>
    </div>
  </div>

  <!-- TOP -->
  <div class="panel" id="panel-top">
    <div id="top-container"><div class="loading"><div class="spin"></div>Connessione a Polymarket...</div></div>
  </div>

  <!-- BACKTEST -->
  <div class="panel" id="panel-backtest">
    <div style="padding:14px">
      <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
        <select id="bt-days" style="flex:1;background:var(--s1);border:1px solid var(--b1);color:var(--t);padding:10px;border-radius:10px;">
          <option value="30">30 giorni</option>
          <option value="90" selected>90 giorni</option>
          <option value="180">180 giorni</option>
        </select>
        <button class="search-btn" onclick="runBacktest()">▶ Esegui</button>
      </div>
      <div id="backtest-container"><div class="empty">Premi Esegui per avviare il backtest.</div></div>
    </div>
  </div>

  <!-- CONFIG -->
  <div class="panel" id="panel-config">
    <div style="padding:14px">
      <div style="background:rgba(255,187,0,0.06);border:1px solid rgba(255,187,0,0.2);border-radius:10px;padding:12px;margin-bottom:12px;font-size:12px;line-height:1.6;">
        <strong style="color:var(--y)">⚡ Setup</strong><br>
        Su Render: aggiungi nelle <strong>Environment Variables</strong><br>
        <code>ANTHROPIC_KEY</code> e (opzionale) <code>NEWSAPI_KEY</code>.<br>
        Polymarket e GDELT funzionano senza keys.
      </div>
      <div class="card">
        <div class="c-top">
          <div class="c-cat">Status</div>
          <div id="status-grid" style="color:var(--m);font-size:12px;line-height:1.9;">—</div>
        </div>
      </div>
      <div class="card">
        <div class="c-top">
          <div class="c-cat">Come funziona</div>
          <div style="color:var(--m);font-size:12px;line-height:1.9;">
            • ⭐ Salvati = i tuoi mercati (salvati in localStorage)<br>
            • Quote “precise” = refresh live per ogni market_id mostrato<br>
            • Cerca = cache + refresh live sui risultati mostrati<br>
            • Predici = sentiment news + AI + update bayesiano<br>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
const S = {
  top:[],
  total:0,
  last_update:'',
  ai:false,
  news:false,

  searchOffset:0,
  searchLimit:30,
  searchTotal:0,
  activeCategory:'',
  predictions:{},
  predicting:new Set(),

  saved:[],         // array di market_id
  savedData:{},     // market_id -> quote live
  savedPred:{},     // market_id -> predict result
  savedLoading:false,
};

const LS_KEY = 'oracle_saved_markets_v1';

function loadSaved(){
  try{
    const raw = localStorage.getItem(LS_KEY);
    S.saved = raw ? JSON.parse(raw) : [];
    if(!Array.isArray(S.saved)) S.saved=[];
  }catch(e){ S.saved=[]; }
}
function saveSaved(){
  localStorage.setItem(LS_KEY, JSON.stringify(S.saved.slice(0,200)));
}
function isSaved(id){ return S.saved.includes(String(id)); }
function addSaved(id){
  id=String(id);
  if(!isSaved(id)){ S.saved.unshift(id); saveSaved(); }
  renderSaved();
  refreshSavedLive(true);
}
function removeSaved(id){
  id=String(id);
  S.saved = S.saved.filter(x=>x!==id);
  saveSaved();
  delete S.savedData[id];
  delete S.savedPred[id];
  renderSaved();
}

document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));
    t.classList.add('on');
    document.getElementById('panel-'+t.dataset.t).classList.add('on');
    if(t.dataset.t==='config') renderStatus();
    if(t.dataset.t==='search' && !document.getElementById('cat-filters').children.length) loadCategories();
    if(t.dataset.t==='saved') refreshSavedLive(false);
  });
});

function updateHeader(d){
  S.total = d.total||0;
  S.last_update = d.last_update||'';
  S.ai = d.has_ai;
  S.news = d.has_news;
  document.getElementById('h-total').textContent = S.total.toLocaleString();
  document.getElementById('h-upd').textContent = S.last_update ? '✓' : '—';
  document.getElementById('h-ai').textContent = S.ai ? '✓ ON' : '✗ OFF';
}

function pct01(x){ return (x*100).toFixed(1)+'%'; }

function renderCard(m, opts={}){
  const id = String(m.id);
  const saved = isSaved(id);
  const btn1 = saved
    ? `<button class="btn red" onclick="removeSaved('${id}')">Rimuovi</button>`
    : `<button class="btn" onclick="addSaved('${id}')">Salva</button>`;

  const pred = opts.pred || null;
  const showPred = !!pred;

  return `
  <div class="card" id="card-${id}">
    <div class="c-top">
      <div class="c-cat">${m.category || 'Other'}</div>
      <div class="c-q">${m.question || '?'}</div>
      <div class="c-row">
        <span class="pill p-poly">POLY ${pct01(m.yes_price || m.polymarket_prob || 0.5)}</span>
        ${showPred ? `<span class="pill p-orc">ORACLE ${pct01(pred.oracle_prob)}</span>` : `<span class="pill p-warn">ORACLE —</span>`}
        ${btn1}
        <button class="btn ${S.predicting.has(id)?'loading':''}" onclick="predict('${id}')" ${S.predicting.has(id)?'disabled':''}>
          ${S.predicting.has(id)?'…':'⚡ Predici'}
        </button>
      </div>
    </div>
    <div class="small">
      <span>${(m.volume_usd||0)>=1e6 ? '$'+((m.volume_usd||0)/1e6).toFixed(2)+'M' : '$'+((m.volume_usd||0)/1e3).toFixed(0)+'K'} vol</span>
      <span>•</span>
      <span>${m.days_to_resolution ?? '—'}g</span>
      ${m.last_updated ? `<span>•</span><span>live ✓</span>` : ``}
    </div>
    ${showPred ? renderPred(pred) : ``}
  </div>`;
}

function renderPred(p){
  const e = p.edge_pct || 0;
  const rec = p.recommendation?.label || '—';
  const strength = p.recommendation?.strength || '';
  const aiP = p.ai_prob!=null ? pct01(p.ai_prob) : '—';
  const newsN = p.sentiment?.articles ?? 0;
  const news = (p.sentiment?.top_news || []).slice(0,3);

  return `
  <div class="pred">
    <div class="pred-grid">
      <div class="tile">
        <div class="tl">Edge</div>
        <div class="tv" style="color:${e>5?'var(--g)':e<-5?'var(--r)':'var(--y)'}">${e>=0?'+':''}${e.toFixed(1)}%</div>
      </div>
      <div class="tile">
        <div class="tl">Claude</div>
        <div class="tv" style="color:var(--g)">${aiP}</div>
      </div>
    </div>
    <div class="tile" style="margin-bottom:10px">
      <div class="tl">Decisione</div>
      <div class="tv" style="font-size:16px">${rec} <span style="color:var(--m);font-size:12px">· ${strength}</span></div>
      <div style="color:var(--m);font-size:11px;margin-top:6px">News analizzate: <strong style="color:var(--t)">${newsN}</strong></div>
    </div>
    ${news.length ? `
      <div class="news">
        ${news.map(n=>`<div class="n">• ${n.title}</div>`).join('')}
      </div>` : ``}
  </div>`;
}

// ───────────────── SAVED ─────────────────

function renderSaved(){
  const c = document.getElementById('saved-container');
  if(!S.saved.length){
    c.innerHTML = `<div class="empty" style="margin-top:20px">Nessun mercato salvato.<br>Vai su <strong>Cerca</strong> e premi “Salva”.</div>`;
    return;
  }
  const cards = S.saved.map(id=>{
    const q = S.savedData[id];
    const pred = S.savedPred[id];
    if(!q){
      return `<div class="card"><div class="c-top"><div class="c-cat">—</div><div class="c-q">Caricamento mercato ${id}…</div></div></div>`;
    }
    return renderCard(q, {pred});
  }).join('');
  c.innerHTML = cards;
}

async function refreshSavedLive(force){
  if(S.savedLoading) return;
  if(!S.saved.length){ renderSaved(); return; }
  if(!force){
    // refresh soft: solo se non abbiamo dati
    const missing = S.saved.some(id=>!S.savedData[id]);
    if(!missing) return;
  }
  S.savedLoading = true;
  try{
    // prendi quote live batch (max 40 per call)
    const chunks = [];
    for(let i=0;i<S.saved.length;i+=40) chunks.push(S.saved.slice(i,i+40));
    for(const ch of chunks){
      const r = await fetch('/api/quotes?ids='+encodeURIComponent(ch.join(',')));
      const d = await r.json();
      (d.quotes||[]).forEach(q=>{ S.savedData[String(q.id)] = q; });
    }
  }catch(e){}
  S.savedLoading = false;
  renderSaved();
}

// ───────────────── SEARCH ─────────────────

async function loadCategories(){
  try{
    const r = await fetch('/api/categories');
    const d = await r.json();
    const wrap = document.getElementById('cat-filters');
    wrap.innerHTML =
      '<span class="fchip on" onclick="selectCat(this,\'\')">Tutti</span>' +
      (d.categories||[]).slice(0,30).map(c=>
        `<span class="fchip" onclick="selectCat(this,'${String(c.name).replace(/'/g,"\\'")}')">${c.name} <span style="color:var(--m)">${c.count}</span></span>`
      ).join('');
  }catch(e){}
}
function selectCat(el, cat){
  S.activeCategory = cat;
  document.querySelectorAll('.fchip').forEach(x=>x.classList.remove('on'));
  el.classList.add('on');
  doSearch(true);
}

async function doSearch(resetPage){
  if(resetPage) S.searchOffset = 0;
  const q = document.getElementById('q-input').value.trim();

  document.getElementById('search-results').innerHTML =
    '<div class="loading"><div class="spin"></div>Ricerca in corso…</div>';
  document.getElementById('search-pagination').style.display='none';

  const params = new URLSearchParams({
    q,
    category: S.activeCategory,
    limit: S.searchLimit,
    offset: S.searchOffset,
    live: 1, // IMPORTANT: quote precise per questa pagina
  });

  try{
    const r = await fetch('/api/search?'+params.toString());
    const d = await r.json();
    S.searchTotal = d.total||0;
    renderSearchResults(d.results||[]);
    renderPagination();
    document.getElementById('search-meta').textContent =
      S.searchTotal ? `${S.searchTotal.toLocaleString()} mercati trovati` : '';
  }catch(e){
    document.getElementById('search-results').innerHTML = '<div class="empty">⚠ Errore nella ricerca.</div>';
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
    document.getElementById('search-pagination').style.display='none';
    return;
  }
  const page = Math.floor(S.searchOffset / S.searchLimit) + 1;
  const pages = Math.ceil(total / S.searchLimit);
  document.getElementById('search-pagination').style.display='flex';
  document.getElementById('pg-info').textContent = `${page} / ${pages}`;
  document.getElementById('pg-prev').disabled = S.searchOffset === 0;
  document.getElementById('pg-next').disabled = S.searchOffset + S.searchLimit >= total;
}
function renderSearchResults(results){
  const c = document.getElementById('search-results');
  if(!results.length){
    c.innerHTML = '<div class="empty">Nessun mercato trovato.<br>Prova una keyword diversa.</div>';
    return;
  }
  c.innerHTML = results.map((m,i)=>`<div style="animation-delay:${i*0.02}s">${renderCard(m, {pred: S.predictions[String(m.id)]||null})}</div>`).join('');
}

// ───────────────── PREDICT ─────────────────

async function predict(id){
  id=String(id);
  if(S.predicting.has(id)) return;
  S.predicting.add(id);
  // re-render saved/search quickly
  renderSaved();
  // try update button label in search
  try{
    const r = await fetch('/api/predict/'+encodeURIComponent(id), {method:'POST'});
    const d = await r.json();
    if(!d.error){
      S.predictions[id] = d;
      S.savedPred[id] = d;
    }
  }catch(e){}
  S.predicting.delete(id);
  renderSaved();
  // se sei in search, aggiorna la lista (semplice)
  const searchPanelOn = document.getElementById('panel-search').classList.contains('on');
  if(searchPanelOn) doSearch(false);
}

// ───────────────── TOP EDGE ─────────────────

async function fetchTop(){
  try{
    const r = await fetch('/api/markets');
    const d = await r.json();
    updateHeader(d);
    S.top = d.markets||[];
    renderTop();
  }catch(e){
    document.getElementById('top-container').innerHTML = '<div class="empty">⚠ Errore connessione.</div>';
  }
}

function renderTop(){
  const c = document.getElementById('top-container');
  if(!S.top.length){
    c.innerHTML = '<div class="loading"><div class="spin"></div>Analisi top mercati in corso...</div>';
    return;
  }
  c.innerHTML = S.top.map((m,i)=>{
    const fakeLight = {
      id:m.id, question:m.question, category:m.category,
      yes_price:m.polymarket_prob, volume_usd:m.volume_usd, days_to_resolution:m.days_to_resolution
    };
    const pred = {
      oracle_prob:m.oracle_prob, edge_pct:m.edge_pct, ai_prob:m.ai_prob,
      recommendation:m.recommendation, sentiment:m.sentiment
    };
    return `<div style="animation-delay:${i*0.02}s">${renderCard(fakeLight, {pred})}</div>`;
  }).join('');
}

// ───────────────── BACKTEST ─────────────────

async function runBacktest(){
  const days = document.getElementById('bt-days').value;
  document.getElementById('backtest-container').innerHTML =
    '<div class="loading"><div class="spin"></div>Backtest in corso…</div>';
  try{
    const r = await fetch('/api/backtest?days='+encodeURIComponent(days));
    const d = await r.json();
    if(d.error){
      document.getElementById('backtest-container').innerHTML = `<div class="empty">⚠ ${d.error}</div>`;
      return;
    }
    document.getElementById('backtest-container').innerHTML = `
      <div class="card">
        <div class="c-top">
          <div class="c-cat">Risultato</div>
          <div class="c-q" style="font-size:16px">
            ${d.oracle_is_better ? '▲ ORACLE BATTE POLYMARKET' : '▼ POLYMARKET PIÙ ACCURATO'}
          </div>
          <hr class="sep">
          <div style="color:var(--m);font-size:12px;line-height:1.8">
            Mercati: <strong style="color:var(--t)">${d.n_markets}</strong><br>
            Brier ORACLE: <strong style="color:var(--g)">${d.oracle_brier}</strong><br>
            Brier Poly: <strong style="color:var(--bl)">${d.poly_brier}</strong><br>
            Miglioramento: <strong style="color:${d.oracle_is_better?'var(--g)':'var(--r)'}">${d.improvement_pct}%</strong><br>
            Win rate: <strong style="color:var(--t)">${Math.round(d.win_rate*100)}%</strong><br>
          </div>
        </div>
      </div>`;
  }catch(e){
    document.getElementById('backtest-container').innerHTML = '<div class="empty">⚠ Errore durante il backtest.</div>';
  }
}

// ───────────────── STATUS ─────────────────

function renderStatus(){
  fetch('/api/health').then(r=>r.json()).then(d=>{
    const items = [
      {n:'Backend Python', ok:true},
      {n:'Polymarket Cache', ok:(d.total||0)>0},
      {n:'Anthropic', ok:!!d.anthropic},
      {n:'NewsAPI', ok:!!d.newsapi},
      {n:'Refresh', ok:true},
    ];
    document.getElementById('status-grid').innerHTML =
      items.map(i=>`• ${i.n}: <strong style="color:${i.ok?'var(--g)':'var(--y)'}">${i.ok?'LIVE':'OFF'}</strong>`).join('<br>') +
      `<br><br><span style="color:var(--m)">refresh: ${d.refresh_secs}s · max_markets: ${d.max_markets}</span>`;
  }).catch(()=>{
    document.getElementById('status-grid').textContent = 'Connessione...';
  });
}

// ───────────────── INIT ─────────────────

loadSaved();
renderSaved();
fetchTop();
loadCategories();
refreshSavedLive(true);

// refresh top + saved quotes
setInterval(fetchTop, 120000);
setInterval(()=>refreshSavedLive(false), 60000);
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)