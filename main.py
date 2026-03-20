"""
DevLabz Search — Self-hosted Perplexity-like AI search engine.
Single-file FastAPI application with embedded cyberpunk frontend.
"""

import os
import json
import asyncio
import logging
import locale
from datetime import datetime
from typing import Optional

try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:
    pass  # Fallback to default locale in container

import httpx
import markdown
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://10.8.0.101:5456")
RERANKER_URL = os.getenv("RERANKER_URL", "http://10.8.0.101:8401")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://10.8.0.101:11235")
LLM_URL = os.getenv("LLM_URL", "http://10.8.0.2:8080")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B")

SYSTEM_PROMPT_TEMPLATE = """\
Tu es DevLabz Search, un agent de recherche expert de classe mondiale.
Date et heure actuelles : {datetime}.

## Identité
Tu es un analyste de recherche senior. Tu ne résumes pas — tu ANALYSES, tu SYNTHÉTISES, tu CONFRONTES les sources entre elles. Tu produis des réponses dignes d'un rapport de recherche professionnel : précises, structurées, exhaustives, avec un regard critique.

## Langue
Réponds TOUJOURS en français, quelle que soit la langue des sources.

## Méthodologie de recherche
1. **Analyse croisée** : ne te contente jamais d'une seule source. Compare les informations entre sources, identifie les convergences et les contradictions.
2. **Esprit critique** : signale quand les sources se contredisent, quand une info semble datée ou douteuse, quand un consensus n'existe pas.
3. **Données concrètes** : privilégie les chiffres, dates, noms, faits vérifiables. Évite les généralités creuses.
4. **Nuance** : distingue fait avéré, consensus d'experts, opinion minoritaire, et spéculation. Ne présente pas une opinion comme un fait.
5. **Exhaustivité** : couvre tous les angles pertinents du sujet. Si un aspect important manque dans les sources, signale-le explicitement.

## Format de réponse
- Commence par UNE PHRASE de définition/réponse directe (la réponse à la question en une ligne).
- Puis structure le reste avec des **### sous-titres** et des **listes à puces** sous chaque sous-titre.
- NE JAMAIS écrire un paragraphe de plus de 2 lignes. Dès que tu as 3+ infos, utilise une liste à puces.
- Cite tes sources avec [1], [2], etc.
- Pas de préambule, pas de conclusion, pas de récapitulatif.
- Exemple de structure idéale :

Phrase de définition directe [1].
### Sous-titre 1
- Point clé [2]
- Point clé [3]
### Sous-titre 2
- Point clé [4]
- Point clé [5]

## Gestion de la conversation
RÈGLES DE PRIORITÉ :
1. Si la question fait référence à la conversation précédente (ex: "ma question", "tu as dit", "développe", "explique", "c'était quoi", "et pour", "pareil mais"), réponds en te basant sur l'HISTORIQUE DE CONVERSATION. Ignore les résultats de recherche s'ils ne sont pas pertinents à la conversation.
2. Si la question est une nouvelle recherche, base-toi sur les sources fournies.
3. Combine les deux quand c'est pertinent : enrichis le contexte conversationnel avec de nouvelles données.

## Qualité
- Ne dis JAMAIS "je n'ai pas accès à internet" ou "en tant qu'IA" — tu ES un moteur de recherche, tu AS les sources devant toi.
- Si les sources ne couvrent pas le sujet, dis-le clairement et indique ce qu'il faudrait chercher de plus.
- CONCISION ABSOLUE. 150 mots max. L'utilisateur demandera plus s'il veut.
- UNE LIGNE par point de liste. Pas de paragraphe dans un bullet point.
- ZÉRO redondance. Si une info est dite, elle n'est pas redite autrement.
- INTERDITS : "Points à retenir", "Résumé", "En conclusion", "En résumé", "Résumé analytique", "Il est important de noter", "Il convient de souligner", "Sources : Wikipédia, etc.", tout titre de type récapitulatif.
- Pas de titre "Résumé analytique" ou "Synthèse" en début de réponse non plus. Commence par l'information directement.
"""

log = logging.getLogger("devlabz")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DevLabz Search")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def search_searxng(query: str, client: httpx.AsyncClient) -> list[dict]:
    """Query SearXNG and return up to 30 results."""
    try:
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "pageno": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:30]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in results
        ]
    except Exception as e:
        log.error("SearXNG error: %s", e)
        return []


async def rerank(query: str, results: list[dict], client: httpx.AsyncClient) -> list[dict]:
    """Rerank results using TEI reranker."""
    if not results:
        return results
    texts = [f"{r['title']}. {r['snippet']}" for r in results]
    try:
        resp = await client.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "texts": texts},
            timeout=30,
        )
        resp.raise_for_status()
        ranked = resp.json()
        # TEI returns list of {index, score}
        if isinstance(ranked, list):
            scored = ranked
        elif isinstance(ranked, dict):
            scored = ranked.get("results", ranked.get("rankings", []))
        else:
            scored = []
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        reranked = []
        for item in scored:
            idx = item.get("index", 0)
            if 0 <= idx < len(results):
                entry = results[idx].copy()
                entry["score"] = round(item.get("score", 0), 4)
                reranked.append(entry)
        return reranked
    except Exception as e:
        log.error("Reranker error: %s", e)
        # Fallback: return originals with fake scores
        for i, r in enumerate(results):
            r["score"] = round(1.0 - i * 0.03, 4)
        return results


async def crawl_pages(urls: list[str], client: httpx.AsyncClient) -> dict[str, str]:
    """Crawl pages via Crawl4AI and return {url: markdown_content}."""
    contents: dict[str, str] = {}
    if not urls:
        return contents
    try:
        # Try batch crawl
        resp = await client.post(
            f"{CRAWL4AI_URL}/crawl",
            json={"urls": urls, "word_count_threshold": 50},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Handle various response formats
        if isinstance(data, list):
            for item in data:
                url = item.get("url", "")
                text = item.get("markdown", item.get("text", item.get("content", "")))
                if url and text:
                    contents[url] = text[:3000]
        elif isinstance(data, dict):
            result = data.get("result", data.get("results", data))
            if isinstance(result, list):
                for item in result:
                    url = item.get("url", "")
                    text = item.get("markdown", item.get("text", item.get("content", "")))
                    if url and text:
                        contents[url] = text[:3000]
            elif isinstance(result, dict):
                for url_key, item in result.items():
                    text = item if isinstance(item, str) else item.get("markdown", item.get("text", ""))
                    if text:
                        contents[url_key] = text[:3000]
    except Exception as e:
        log.error("Crawl4AI error: %s", e)
    return contents


async def stream_llm(messages: list[dict], client: httpx.AsyncClient):
    """Yield streamed tokens from the LLM."""
    try:
        async with client.stream(
            "POST",
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": True,
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=120,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as e:
        log.error("LLM stream error: %s", e)
        yield f"\n\n[Erreur LLM: {e}]"


async def generate_related(query: str, answer: str, client: httpx.AsyncClient) -> list[str]:
    """Generate 3 related questions."""
    prompt = (
        f"L'utilisateur a cherché : \"{query}\"\n"
        f"Voici un extrait de la réponse : {answer[:500]}\n\n"
        "Génère exactement 3 questions de recherche liées et complémentaires, en français. "
        "Réponds UNIQUEMENT avec un JSON array de 3 strings, sans autre texte. Exemple: [\"q1\",\"q2\",\"q3\"]"
    )
    full = ""
    async for token in stream_llm(
        [{"role": "user", "content": prompt}],
        client,
    ):
        full += token
    # Parse JSON from response
    try:
        # Find JSON array in response
        start = full.find("[")
        end = full.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(full[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Search endpoint (SSE)
# ---------------------------------------------------------------------------

@app.post("/api/search")
async def api_search(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    mode = body.get("mode", "quick")  # quick | deep
    history = body.get("history", [])  # list of {role, content} from previous turns
    async def event_generator():
        async with httpx.AsyncClient() as client:
            now = datetime.now().strftime("%A %d %B %Y, %H:%M:%S")
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(datetime=now)

            # 1 — Search
            yield {"event": "status", "data": "Recherche en cours..."}
            results = await search_searxng(query, client)
            if not results:
                yield {"event": "status", "data": "Aucun résultat trouvé."}
                yield {"event": "sources", "data": json.dumps([])}
                yield {"event": "answer", "data": "Aucun résultat trouvé pour cette requête."}
                yield {"event": "related", "data": json.dumps([])}
                yield {"event": "done", "data": ""}
                return

            # 2 — Rerank
            yield {"event": "status", "data": "Classement par pertinence..."}
            ranked = await rerank(query, results, client)
            yield {"event": "sources", "data": json.dumps(ranked[:15], ensure_ascii=False)}

            # 3 — Deep mode: crawl top pages
            crawled: dict[str, str] = {}
            if mode == "deep":
                yield {"event": "status", "data": "Lecture des pages..."}
                top_urls = [r["url"] for r in ranked[:5]]
                crawled = await crawl_pages(top_urls, client)

            # 4 — Build context
            context_parts = []
            for i, r in enumerate(ranked[:10], 1):
                url = r["url"]
                if url in crawled and crawled[url].strip():
                    context_parts.append(
                        f"[{i}] {r['title']} ({url})\n{crawled[url][:2000]}"
                    )
                else:
                    context_parts.append(
                        f"[{i}] {r['title']} ({url})\n{r['snippet']}"
                    )
            context = "\n\n---\n\n".join(context_parts)

            # 5 — Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]

            # Inject previous turns (trimmed to last 10 exchanges max)
            for turn in history[-20:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

            messages.append({
                "role": "user",
                "content": f"Question : {query}\n\nSources :\n\n{context}",
            })

            # 6 — Stream answer
            yield {"event": "status", "data": "Génération de la réponse..."}
            full_answer = ""
            async for token in stream_llm(messages, client):
                full_answer += token
                yield {"event": "answer", "data": json.dumps(token, ensure_ascii=False)}

            # 7 — Related questions
            yield {"event": "status", "data": "Questions liées..."}
            related = await generate_related(query, full_answer, client)
            yield {"event": "related", "data": json.dumps(related, ensure_ascii=False)}

            yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Whisper + Kokoro proxy routes
# ---------------------------------------------------------------------------

WHISPER_URL = os.getenv("WHISPER_URL", "http://10.8.0.101:8300")
KOKORO_URL_ENV = os.getenv("KOKORO_URL", "http://10.8.0.101:8880")


@app.post("/api/whisper")
async def proxy_whisper(request: Request):
    """Proxy audio to Whisper for transcription."""
    from fastapi.responses import JSONResponse
    body = await request.body()
    ct = request.headers.get("content-type", "")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{WHISPER_URL}/v1/audio/transcriptions",
            content=body,
            headers={"content-type": ct},
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/api/tts")
async def proxy_tts(request: Request):
    """Proxy text to Kokoro for TTS."""
    from fastapi.responses import Response
    body = await request.json()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{KOKORO_URL_ENV}/v1/audio/speech",
            json=body,
        )
        return Response(
            content=resp.content,
            media_type=resp.headers.get("content-type", "audio/mpeg"),
            status_code=resp.status_code,
        )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DevLabz Search</title>
<style>
/* ================================================================
   RESET & VARIABLES
   ================================================================ */
*,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0a0a1a;
  --cyan: #00f0ff;
  --red: #ff1744;
  --magenta: #e040fb;
  --text: #e0e0e0;
  --text2: #888;
  --card: rgba(255,255,255,0.04);
  --border: rgba(0,240,255,0.1);
}

body {
  font-family: system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

a { color: var(--cyan); text-decoration: none; transition: color 0.2s; }
a:hover { text-decoration: underline; }

/* ================================================================
   STARFIELD CANVAS BACKGROUND
   ================================================================ */
#starfield {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
}
@keyframes gridDrift {
  from { transform: translate(0,0); }
  to   { transform: translate(60px,60px); }
}

/* ================================================================
   SCROLLBAR
   ================================================================ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,240,255,0.35); }

/* ================================================================
   KEYFRAME ANIMATIONS
   ================================================================ */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes logoGlow {
  0%,100% { filter: drop-shadow(0 0 15px rgba(0,240,255,0.3)); }
  50%     { filter: drop-shadow(0 0 30px rgba(0,240,255,0.6)); }
}
@keyframes pulseRecord {
  0%,100% { box-shadow: 0 0 0 0 rgba(255,23,68,0.5); }
  50%     { box-shadow: 0 0 0 12px rgba(255,23,68,0); }
}
@keyframes pulseDot {
  0%,100% { opacity: 1; }
  50%     { opacity: 0.3; }
}
@keyframes cursorBlink {
  0%,100% { opacity: 1; }
  50%     { opacity: 0; }
}
@keyframes glowPulse {
  0%,100% { box-shadow: 0 0 15px rgba(0,240,255,0.1); }
  50%     { box-shadow: 0 0 30px rgba(0,240,255,0.25); }
}

/* ================================================================
   HEADER
   ================================================================ */
.header {
  text-align: center;
  padding: 70px 20px 14px;
  animation: fadeSlideUp 0.6s ease;
  transition: padding 0.3s ease;
}
.header.compact { padding: 20px 20px 8px; }

.logo {
  font-size: 2.5rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--cyan), var(--magenta));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: logoGlow 3s ease-in-out infinite;
  transition: font-size 0.3s ease;
}
.header.compact .logo { font-size: 1.4rem; }

.tagline {
  color: var(--text2);
  font-size: 0.82rem;
  margin-top: 6px;
  transition: opacity 0.3s, max-height 0.3s;
}
.header.compact .tagline {
  opacity: 0;
  max-height: 0;
  overflow: hidden;
  margin: 0;
}

/* ================================================================
   SEARCH BAR
   ================================================================ */
.search-wrap {
  display: flex;
  justify-content: center;
  padding: 14px 20px;
  position: sticky;
  top: 0;
  z-index: 100;
  background: linear-gradient(var(--bg) 80%, transparent);
}

.search-box {
  display: flex;
  align-items: center;
  max-width: 900px;
  width: 100%;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  transition: box-shadow 0.3s ease, border-color 0.3s ease;
  backdrop-filter: blur(12px);
}
.search-box:focus-within {
  border-color: var(--cyan);
  animation: glowPulse 2s ease-in-out infinite;
}

.search-box input {
  flex: 1;
  padding: 16px 20px;
  font-size: 1rem;
  background: transparent;
  border: none;
  color: var(--text);
  outline: none;
  min-width: 0;
}
.search-box input::placeholder { color: var(--text2); }

/* ================================================================
   MIC BUTTON
   ================================================================ */
.mic-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: rgba(255,255,255,0.05);
  color: var(--text2);
  cursor: pointer;
  transition: all 0.3s ease;
  flex-shrink: 0;
  margin-right: 4px;
  font-size: 1.1rem;
}
.mic-btn:hover { background: rgba(255,255,255,0.1); color: var(--text); }
.mic-btn.recording {
  background: var(--red);
  color: #fff;
  animation: pulseRecord 1.5s ease-in-out infinite;
}

/* ================================================================
   MODE TOGGLE
   ================================================================ */
.mode-toggle {
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 4px;
  flex-shrink: 0;
}
.mode-btn {
  padding: 8px 14px;
  font-size: 0.75rem;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.25s ease;
  background: transparent;
  color: var(--text2);
}
.mode-btn:hover { color: var(--text); }
.mode-btn.active { background: rgba(0,240,255,0.12); color: var(--cyan); }

/* ================================================================
   GO BUTTON
   ================================================================ */
.search-box button.go {
  padding: 14px 28px;
  background: var(--cyan);
  border: none;
  color: #0a0a1a;
  font-weight: 700;
  font-size: 0.9rem;
  cursor: pointer;
  transition: filter 0.2s, transform 0.2s;
  flex-shrink: 0;
}
.search-box button.go:hover { filter: brightness(1.15); transform: scale(1.02); }
.search-box button.go:active { transform: scale(0.98); }

/* ================================================================
   STATUS BAR
   ================================================================ */
.status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px;
  font-size: 0.82rem;
  color: var(--cyan);
  min-height: 32px;
  opacity: 0;
  transition: opacity 0.3s ease;
}
.status.on { opacity: 1; }

.status .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--cyan);
  display: none;
  animation: pulseDot 1s ease-in-out infinite;
}
.status.on .dot { display: block; }

/* ================================================================
   MAIN CONTAINER
   ================================================================ */
.main {
  max-width: 1600px;
  margin: 0 auto;
  padding: 0 30px 80px;
}

/* ================================================================
   ANSWER CARD
   ================================================================ */
.answer-wrap { position: relative; }

.answer {
  display: none;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px 32px;
  margin-bottom: 16px;
  backdrop-filter: blur(12px);
  line-height: 1.55;
  font-size: 1rem;
  letter-spacing: 0.01em;
  position: relative;
}
.answer.on {
  display: block;
  animation: fadeSlideUp 0.4s ease;
}

.answer h1, .answer h2, .answer h3 { color: var(--cyan); margin: 10px 0 4px; }
.answer h1 { font-size: 1.25rem; }
.answer h2 { font-size: 1.12rem; }
.answer h3 { font-size: 1.02rem; }
.answer p { margin: 2px 0; }
.answer ul, .answer ol { margin: 2px 0 2px 18px; }
.answer li { margin: 1px 0; line-height: 1.45; }
.answer br + br { display: none; }
.answer ul + p:empty, .answer ol + p:empty { display: none; }
.answer li > br:first-child { display: none; }
.answer code {
  background: rgba(0,240,255,0.08);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.87em;
  color: var(--cyan);
}
.answer pre {
  background: rgba(0,0,0,0.5);
  padding: 16px;
  border-radius: 10px;
  overflow-x: auto;
  margin: 12px 0;
  border: 1px solid rgba(0,240,255,0.05);
}
.answer pre code { background: none; padding: 0; color: var(--text); }
.answer blockquote {
  border-left: 3px solid var(--cyan);
  padding-left: 16px;
  color: var(--text2);
  margin: 12px 0;
}
.answer a.cite {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,240,255,0.12);
  color: var(--cyan);
  font-size: 0.68rem;
  font-weight: 700;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  vertical-align: super;
  margin: 0 2px;
  text-decoration: none;
  transition: background 0.2s;
}
.answer a.cite:hover { background: rgba(0,240,255,0.3); }

/* ================================================================
   TTS BUTTON
   ================================================================ */
.tts-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  background: rgba(0,240,255,0.1);
  border: 1px solid var(--border);
  color: var(--cyan);
  font-size: 1.1rem;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.2s;
  display: none;
  align-items: center;
  justify-content: center;
}
.answer-wrap:has(.answer.on) .tts-btn {
  display: flex;
}
.tts-btn:hover { background: rgba(0,240,255,0.2); border-color: var(--cyan); }

/* ================================================================
   TYPING CURSOR
   ================================================================ */
.typing-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background: var(--cyan);
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: cursorBlink 0.8s step-end infinite;
}

/* ================================================================
   RELATED QUESTIONS
   ================================================================ */
.related {
  display: none;
  margin-top: 24px;
  margin-bottom: 28px;
}
.related.on {
  display: block;
  animation: fadeSlideUp 0.4s ease 0.2s both;
}
.related-title {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 10px;
}
.related-chips { display: flex; flex-wrap: wrap; gap: 10px; }
.related-chip {
  padding: 10px 18px;
  font-size: 0.82rem;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 24px;
  color: var(--text);
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(8px);
}
.related-chip:hover {
  border-color: var(--magenta);
  color: var(--magenta);
  transform: translateY(-1px);
}

/* ================================================================
   SOURCES
   ================================================================ */
.sources-section {
  display: none;
  margin-top: 8px;
}
.sources-section.on {
  display: block;
  animation: fadeSlideUp 0.4s ease 0.1s both;
}
.sources-title {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 14px;
}
.sources-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.src-card {
  display: flex;
  align-items: flex-start;
  gap: 20px;
  padding: 24px 32px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(8px);
}
.src-card:hover {
  border-color: var(--cyan);
  background: rgba(0,240,255,0.03);
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0,240,255,0.1);
}
.src-card .num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 28px;
  border-radius: 50%;
  background: rgba(0,240,255,0.12);
  color: var(--cyan);
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
  margin-top: 2px;
}
.src-card .info { flex: 1; min-width: 0; }
.src-card .info h4 {
  font-size: 1.05rem;
  color: var(--text);
  margin-bottom: 6px;
  line-height: 1.4;
  word-wrap: break-word;
  font-weight: 600;
}
.src-card .info .meta {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 8px;
}
.src-card .info .domain { font-size: 0.82rem; color: var(--cyan); opacity: 0.7; }
.src-card .info .score {
  font-size: 0.72rem;
  color: var(--red);
  background: rgba(255,23,68,0.1);
  padding: 2px 8px;
  border-radius: 8px;
}
.src-card .info .snippet {
  font-size: 0.9rem;
  color: var(--text2);
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ================================================================
   RESPONSIVE
   ================================================================ */
/* ================================================================
   CONVERSATION HISTORY
   ================================================================ */
.conv-history {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-bottom: 24px;
}

.conv-turn {
  animation: fadeSlideUp 0.3s ease;
}

.conv-query {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}

.conv-query-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: rgba(224,64,251,0.15);
  border: 1px solid rgba(224,64,251,0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  flex-shrink: 0;
  color: var(--magenta);
}

.conv-query-text {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  padding-top: 5px;
}

.conv-answer {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 28px;
  backdrop-filter: blur(12px);
  line-height: 1.55;
  font-size: 1rem;
  opacity: 0.75;
  transition: opacity 0.2s;
  position: relative;
}

.conv-answer:hover { opacity: 1; }

.conv-answer h1, .conv-answer h2, .conv-answer h3 { color: var(--cyan); margin: 10px 0 4px; }
.conv-answer h1 { font-size: 1.25rem; }
.conv-answer h2 { font-size: 1.12rem; }
.conv-answer h3 { font-size: 1.02rem; }
.conv-answer p { margin: 2px 0; }
.conv-answer ul, .conv-answer ol { margin: 2px 0 2px 18px; }
.conv-answer li { margin: 1px 0; line-height: 1.45; }
.conv-answer br + br { display: none; }
.conv-answer ul + p:empty, .conv-answer ol + p:empty { display: none; }
.conv-answer li > br:first-child { display: none; }
.conv-answer code {
  background: rgba(0,240,255,0.08);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.87em;
  color: var(--cyan);
}
.conv-answer pre {
  background: rgba(0,0,0,0.5);
  padding: 16px;
  border-radius: 10px;
  overflow-x: auto;
  margin: 12px 0;
  border: 1px solid rgba(0,240,255,0.05);
}
.conv-answer pre code { background: none; padding: 0; color: var(--text); }
.conv-answer a.cite {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,240,255,0.12);
  color: var(--cyan);
  font-size: 0.68rem;
  font-weight: 700;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  vertical-align: super;
  margin: 0 2px;
  text-decoration: none;
}

.conv-sources-toggle {
  font-size: 0.75rem;
  color: var(--text2);
  cursor: pointer;
  margin-top: 10px;
  padding: 6px 12px;
  background: rgba(0,240,255,0.04);
  border: 1px solid rgba(0,240,255,0.08);
  border-radius: 8px;
  display: inline-block;
  transition: all 0.2s;
}
.conv-sources-toggle:hover {
  border-color: var(--cyan);
  color: var(--cyan);
}

.conv-sources-list {
  display: none;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
}
.conv-sources-list.open { display: flex; }

.conv-separator {
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,240,255,0.15), transparent);
  margin: 8px 0;
}

/* New conversation button */
.new-conv-btn {
  position: fixed;
  bottom: 24px;
  right: 24px;
  padding: 12px 20px;
  background: rgba(224,64,251,0.12);
  border: 1px solid rgba(224,64,251,0.3);
  border-radius: 12px;
  color: var(--magenta);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  backdrop-filter: blur(12px);
  z-index: 200;
  display: none;
}
.new-conv-btn:hover {
  background: rgba(224,64,251,0.25);
  border-color: var(--magenta);
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(224,64,251,0.2);
}
.new-conv-btn.on { display: block; }

/* Follow-up input at bottom of conversation */
.followup-wrap {
  display: none;
  position: sticky;
  bottom: 0;
  z-index: 100;
  padding: 16px 0;
  background: linear-gradient(transparent, var(--bg) 30%);
}
.followup-wrap.on { display: block; }

.followup-box {
  display: flex;
  align-items: center;
  max-width: 900px;
  margin: 0 auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  backdrop-filter: blur(12px);
  transition: box-shadow 0.3s, border-color 0.3s;
}
.followup-box:focus-within {
  border-color: var(--cyan);
  animation: glowPulse 2s ease-in-out infinite;
}

.followup-box input {
  flex: 1;
  padding: 14px 18px;
  font-size: 0.95rem;
  background: transparent;
  border: none;
  color: var(--text);
  outline: none;
  min-width: 0;
}
.followup-box input::placeholder { color: var(--text2); }

.followup-box button {
  padding: 14px 22px;
  background: var(--cyan);
  border: none;
  color: #0a0a1a;
  font-weight: 700;
  font-size: 0.85rem;
  cursor: pointer;
  transition: filter 0.2s;
  flex-shrink: 0;
}
.followup-box button:hover { filter: brightness(1.15); }

/* Turn counter badge */
.turn-count {
  font-family: monospace;
  font-size: 0.65rem;
  color: var(--text2);
  position: absolute;
  top: 12px;
  right: 14px;
  opacity: 0.5;
}

@media (max-width: 700px) {
  .main { padding: 0 14px 50px; }
  .answer { padding: 22px 20px; }
  .conv-answer { padding: 18px 16px; }
  .search-box { flex-wrap: wrap; }
  .mode-toggle { width: 100%; justify-content: center; padding: 6px; }
  .mic-btn { margin: 0 4px; }
  .src-card { padding: 14px 16px; }
  .header { padding: 40px 16px 10px; }
  .logo { font-size: 1.8rem; }
  .new-conv-btn { bottom: 14px; right: 14px; padding: 10px 16px; }
}
</style>
</head>
<body>

<!-- ============================================================
     HEADER
     ============================================================ -->
<canvas id="starfield"></canvas>
<div class="header" id="hdr">
  <div class="logo">DevLabz Search</div>
  <div class="tagline">Recherche IA augment&eacute;e &mdash; sources v&eacute;rifi&eacute;es, r&eacute;ponses instantan&eacute;es</div>
</div>

<!-- ============================================================
     SEARCH BAR
     ============================================================ -->
<div class="search-wrap">
  <div class="search-box" id="searchBox">
    <input id="q" placeholder="Rechercher..." autocomplete="off" autofocus />
    <button class="mic-btn" id="micBtn" title="Recherche vocale">
      <span class="mic-icon">&#127908;</span>
    </button>
    <div class="mode-toggle">
      <button class="mode-btn active" data-mode="quick">Rapide</button>
      <button class="mode-btn" data-mode="deep">Approfondi</button>
    </div>
    <button class="go" id="goBtn">Rechercher</button>
  </div>
</div>

<!-- ============================================================
     STATUS
     ============================================================ -->
<div class="status" id="status">
  <span class="dot"></span>
  <span class="status-text" id="statusText"></span>
</div>

<!-- ============================================================
     MAIN CONTENT
     ============================================================ -->
<div class="main">
  <div class="conv-history" id="convHistory"></div>

  <div class="answer-wrap">
    <div class="answer" id="answer"></div>
    <button class="tts-btn" id="ttsBtn" title="Lire la r&eacute;ponse">&#128264;</button>
  </div>

  <div class="related" id="related">
    <div class="related-title">Questions li&eacute;es</div>
    <div class="related-chips" id="relChips"></div>
  </div>

  <div class="followup-wrap" id="followupWrap">
    <div class="followup-box">
      <input id="followupInput" placeholder="Continuer la conversation..." autocomplete="off" />
      <button id="followupBtn">Envoyer</button>
    </div>
  </div>
</div>

<button class="new-conv-btn" id="newConvBtn">&#10227; Nouvelle conversation</button>

<!-- ============================================================
     JAVASCRIPT
     ============================================================ -->
<script>
(function() {
  "use strict";

  /* ---- State ---- */
  var mode = "quick";
  var sources = [];
  var abortCtrl = null;
  var streaming = false;

  /* ---- Conversation state ---- */
  var conversation = [];  /* [{query, answer, sources}] */
  var llmHistory = [];    /* [{role, content}] sent to backend */

  /* ---- DOM refs ---- */
  var qInput      = document.getElementById("q");
  var hdr         = document.getElementById("hdr");
  var statusEl    = document.getElementById("status");
  var statusText  = document.getElementById("statusText");
  var answerEl    = document.getElementById("answer");
  var relEl       = document.getElementById("related");
  var relChips    = document.getElementById("relChips");
  /* Sources section removed from UI — sources parsed silently for citations */
  var micBtn      = document.getElementById("micBtn");
  var ttsBtn      = document.getElementById("ttsBtn");
  var goBtn       = document.getElementById("goBtn");
  var convHistory   = document.getElementById("convHistory");
  var newConvBtn    = document.getElementById("newConvBtn");
  var followupWrap  = document.getElementById("followupWrap");
  var followupInput = document.getElementById("followupInput");
  var followupBtn   = document.getElementById("followupBtn");

  /* ==============================================================
     UTILITY
     ============================================================== */
  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s || "";
    return d.innerHTML;
  }

  function setStatus(text, on) {
    if (on) {
      statusEl.classList.add("on");
      statusText.textContent = text;
    } else {
      statusEl.classList.remove("on");
      statusText.textContent = "";
    }
  }

  /* ==============================================================
     SIMPLE MARKDOWN RENDERER
     ============================================================== */
  function renderMd(t) {
    /* Pre-process: group list items by removing blank lines between them */
    t = t.replace(/(^[\\-\\*] .+$)\\n{2,}(?=[\\-\\*] )/gm, "$1\\n");
    t = t.replace(/(^\\d+\\. .+$)\\n{2,}(?=\\d+\\. )/gm, "$1\\n");

    var r = t
      .replace(/```(\\w*)\\n([\\s\\S]*?)```/g, "<pre><code>$2</code></pre>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/^## (.+)$/gm, "<h2>$1</h2>")
      .replace(/^# (.+)$/gm, "<h1>$1</h1>")
      .replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>")
      .replace(/\\*(.+?)\\*/g, "<em>$1</em>")
      .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
      .replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>')
      .replace(/^[\\-\\*] (.+)$/gm, "<li>$1</li>")
      .replace(/^\\d+\\. (.+)$/gm, "<li>$1</li>");

    /* Wrap consecutive <li> into <ul>, removing any \\n between them */
    r = r.replace(/((?:<li>.*<\\/li>(?:\\n|\\s)*)+)/g, function(block) {
      var clean = block.replace(/\\n/g, "").replace(/\\s*(<li>)/g, "$1");
      return "<ul>" + clean + "</ul>";
    });

    r = r
      /* Remove blank lines around block elements */
      .replace(/\\n*(<\\/?(?:ul|h[123]|pre|blockquote)(?:[^>]*)>)\\n*/g, "$1")
      /* Remaining double newlines = paragraph breaks */
      .replace(/\\n\\n+/g, "</p><p>")
      .replace(/\\n/g, "<br>")
      /* Wrap in <p> */
      .replace(/^/, "<p>")
      .replace(/$/, "</p>")
      /* Cleanup */
      .replace(/<p>\\s*<\\/p>/g, "")
      .replace(/<p><br><\\/p>/g, "")
      .replace(/<p>\\s*(<(?:ul|h[123]|pre|blockquote))/g, "$1")
      .replace(/(<\\/(?:ul|h[123]|pre|blockquote)>)\\s*<\\/p>/g, "$1")
      .replace(/<br>\\s*(<(?:ul|h[123]))/g, "$1")
      .replace(/(<\\/(?:ul|h[123])>)\\s*<br>/g, "$1");

    return r;
  }

  /* Sources rendering removed — sources parsed silently for [N] citation links */

  /* ==============================================================
     RENDER: ANSWER (current turn)
     ============================================================== */
  function renderAnswer(md, isStreaming) {
    var processed = md.replace(/\\[(\\d+)\\]/g, function(m, n) {
      var idx = parseInt(n) - 1;
      if (idx >= 0 && idx < sources.length) {
        return '<a class="cite" href="' + sources[idx].url + '" target="_blank" title="' + esc(sources[idx].title) + '">' + n + "</a>";
      }
      return m;
    });
    answerEl.innerHTML = renderMd(processed) + (isStreaming ? '<span class="typing-cursor"></span>' : "");
    if (!answerEl.classList.contains("on")) {
      answerEl.classList.add("on");
    }
    /* Auto-scroll to bottom during streaming */
    if (isStreaming) {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }
  }

  /* ==============================================================
     RENDER: RELATED QUESTIONS
     ============================================================== */
  function renderRelated(qs) {
    relChips.innerHTML = "";
    if (!qs || !qs.length) return;
    relEl.classList.add("on");
    qs.forEach(function(q, i) {
      var chip = document.createElement("div");
      chip.className = "related-chip";
      chip.style.animationDelay = (0.1 * i) + "s";
      chip.textContent = q;
      chip.onclick = function() { go(q); };
      relChips.appendChild(chip);
    });
  }

  /* ==============================================================
     RENDER: CONVERSATION HISTORY
     ============================================================== */
  function renderConvHistory() {
    convHistory.innerHTML = "";
    if (!conversation.length) return;

    conversation.forEach(function(turn, idx) {
      var turnEl = document.createElement("div");
      turnEl.className = "conv-turn";

      /* Query bubble */
      var queryEl = document.createElement("div");
      queryEl.className = "conv-query";
      queryEl.innerHTML =
        '<div class="conv-query-icon">Q</div>' +
        '<div class="conv-query-text">' + esc(turn.query) + '</div>';
      turnEl.appendChild(queryEl);

      /* Answer card */
      var ansEl = document.createElement("div");
      ansEl.className = "conv-answer";

      var processed = turn.answer.replace(/\\[(\\d+)\\]/g, function(m, n) {
        var sidx = parseInt(n) - 1;
        if (turn.sources && sidx >= 0 && sidx < turn.sources.length) {
          return '<a class="cite" href="' + turn.sources[sidx].url + '" target="_blank" title="' + esc(turn.sources[sidx].title) + '">' + n + "</a>";
        }
        return m;
      });
      ansEl.innerHTML = renderMd(processed) + '<span class="turn-count">#' + (idx + 1) + '</span>';
      turnEl.appendChild(ansEl);

      /* Collapsible sources — all sources with full details */
      if (turn.sources && turn.sources.length) {
        var toggleEl = document.createElement("div");
        toggleEl.className = "conv-sources-toggle";
        toggleEl.textContent = turn.sources.length + " source(s)";
        var srcList = document.createElement("div");
        srcList.className = "conv-sources-list";
        turn.sources.forEach(function(s, si) {
          var srcItem = document.createElement("div");
          srcItem.className = "src-card";
          var domain;
          try { domain = new URL(s.url).hostname.replace("www.", ""); }
          catch(e) { domain = s.url; }
          srcItem.onclick = function() { window.open(s.url, "_blank"); };
          srcItem.innerHTML =
            '<span class="num">' + (si + 1) + "</span>" +
            '<div class="info">' +
              "<h4>" + esc(s.title) + "</h4>" +
              '<div class="meta">' +
                '<span class="domain">' + esc(domain) + "</span>" +
                '<a href="' + esc(s.url) + '" target="_blank" style="font-size:0.72rem;color:var(--text2);word-break:break-all;margin-left:8px">' + esc(s.url) + "</a>" +
                (s.score !== undefined ? '<span class="score">' + (s.score * 100).toFixed(0) + "%</span>" : "") +
              "</div>" +
              (s.snippet ? '<div class="snippet">' + esc(s.snippet) + "</div>" : "") +
            "</div>";
          srcList.appendChild(srcItem);
        });
        toggleEl.addEventListener("click", function() {
          srcList.classList.toggle("open");
          toggleEl.textContent = srcList.classList.contains("open")
            ? "Masquer les sources"
            : turn.sources.length + " source(s)";
        });
        turnEl.appendChild(toggleEl);
        turnEl.appendChild(srcList);
      }

      /* Separator */
      var sep = document.createElement("div");
      sep.className = "conv-separator";
      turnEl.appendChild(sep);

      convHistory.appendChild(turnEl);
    });
  }

  /* ==============================================================
     ARCHIVE CURRENT TURN INTO HISTORY
     ============================================================== */
  function archiveCurrentTurn(query, answer, turnSources) {
    conversation.push({
      query: query,
      answer: answer,
      sources: turnSources ? turnSources.slice() : []
    });

    /* Add to LLM history for context */
    llmHistory.push({ role: "user", content: query });
    llmHistory.push({ role: "assistant", content: answer });

    renderConvHistory();
    newConvBtn.classList.add("on");
    followupWrap.classList.add("on");
    followupInput.focus();
  }

  /* ==============================================================
     NEW CONVERSATION
     ============================================================== */
  function resetConversation() {
    conversation = [];
    llmHistory = [];
    convHistory.innerHTML = "";
    answerEl.innerHTML = "";
    answerEl.classList.remove("on");
    relEl.classList.remove("on");
    relChips.innerHTML = "";
    sources = [];
    qInput.value = "";
    followupInput.value = "";
    hdr.classList.remove("compact");
    newConvBtn.classList.remove("on");
    followupWrap.classList.remove("on");
    setStatus("", false);
    qInput.focus();
  }

  newConvBtn.addEventListener("click", resetConversation);

  /* ==============================================================
     MAIN SEARCH FUNCTION
     ============================================================== */
  function go(override) {
    var query = override || qInput.value.trim();
    if (!query) return;
    if (override) qInput.value = query;
    if (abortCtrl) abortCtrl.abort();
    abortCtrl = new AbortController();
    streaming = true;

    /* Compact header */
    hdr.classList.add("compact");

    /* Reset current turn UI */
    answerEl.innerHTML = "";
    answerEl.classList.remove("on");
    relEl.classList.remove("on");
    relChips.innerHTML = "";
    sources = [];
    qInput.value = "";
    followupInput.value = "";
    setStatus("Recherche en cours...", true);

    var raw = "";

    fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        mode: mode,
        history: llmHistory
      }),
      signal: abortCtrl.signal
    })
    .then(function(resp) {
      var reader = resp.body.getReader();
      var dec = new TextDecoder();
      var buf = "";

      function pump() {
        return reader.read().then(function(result) {
          if (result.done) {
            streaming = false;
            setStatus("", false);
            if (raw) renderAnswer(raw, false);
            return;
          }
          buf += dec.decode(result.value, { stream: true });
          var lines = buf.split("\\n");
          buf = lines.pop();
          var evt = "";
          for (var i = 0; i < lines.length; i++) {
            var ln = lines[i];
            if (ln.indexOf("event:") === 0) {
              evt = ln.slice(6).trim();
            } else if (ln.indexOf("data:") === 0) {
              var d = ln.slice(5);
              if (evt === "status") {
                setStatus(d, true);
              } else if (evt === "sources") {
                /* Parse sources silently (used for citation links, not displayed) */
                try { sources = JSON.parse(d); } catch(e) {}
              } else if (evt === "answer") {
                try { raw += JSON.parse(d); } catch(e) { raw += d; }
                renderAnswer(raw, true);
              } else if (evt === "related") {
                try { renderRelated(JSON.parse(d)); } catch(e) {}
              } else if (evt === "done") {
                streaming = false;
                setStatus("", false);
                renderAnswer(raw, false);
                /* Archive into conversation history */
                archiveCurrentTurn(query, raw, sources);
                setTimeout(function() {
                  answerEl.innerHTML = "";
                  answerEl.classList.remove("on");
                  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
                  followupInput.focus();
                }, 600);
              }
              evt = "";
            }
          }
          return pump();
        });
      }
      return pump();
    })
    .catch(function(e) {
      if (e.name !== "AbortError") {
        setStatus("Erreur", true);
        console.error(e);
      }
      streaming = false;
    });
  }

  /* ==============================================================
     MODE TOGGLE
     ============================================================== */
  document.querySelectorAll(".mode-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      mode = this.getAttribute("data-mode");
      document.querySelectorAll(".mode-btn").forEach(function(b) { b.classList.remove("active"); });
      this.classList.add("active");
    });
  });

  /* ==============================================================
     ENTER KEY + GO BUTTON
     ============================================================== */
  qInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") go();
  });
  goBtn.addEventListener("click", function() { go(); });

  /* Follow-up input */
  followupInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") {
      var q = followupInput.value.trim();
      if (q) { followupInput.value = ""; go(q); }
    }
  });
  followupBtn.addEventListener("click", function() {
    var q = followupInput.value.trim();
    if (q) { followupInput.value = ""; go(q); }
  });

  /* ==============================================================
     VOICE: WHISPER STT
     ============================================================== */
  var mediaRec = null;
  var recording = false;
  var audioChunks = [];

  micBtn.addEventListener("click", function() {
    if (recording) {
      mediaRec.stop();
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(function(stream) {
        mediaRec = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
        audioChunks = [];

        mediaRec.ondataavailable = function(e) {
          if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRec.onstop = function() {
          stream.getTracks().forEach(function(t) { t.stop(); });
          recording = false;
          micBtn.classList.remove("recording");
          micBtn.innerHTML = '<span class="mic-icon">&#127908;</span>';

          var blob = new Blob(audioChunks, { type: "audio/webm" });
          setStatus("Transcription en cours...", true);

          var fd = new FormData();
          fd.append("file", blob, "audio.webm");
          fd.append("language", "fr");

          fetch("/api/whisper", { method: "POST", body: fd })
            .then(function(r) { return r.json(); })
            .then(function(d) {
              var txt = d.text || d.transcription || "";
              if (txt.trim()) {
                qInput.value = txt.trim();
                setStatus("", false);
                go();
              } else {
                setStatus("Rien entendu", true);
                setTimeout(function() { setStatus("", false); }, 2000);
              }
            })
            .catch(function(e) {
              console.error("Whisper error:", e);
              setStatus("Erreur Whisper", true);
              setTimeout(function() { setStatus("", false); }, 2000);
            });
        };

        mediaRec.start();
        recording = true;
        micBtn.classList.add("recording");
        micBtn.innerHTML = "&#9632;";
      })
      .catch(function(e) {
        console.error("Mic error:", e);
        alert("Micro non disponible");
      });
  });

  /* ==============================================================
     TTS: KOKORO
     ============================================================== */
  var ttsAudio = null;

  ttsBtn.addEventListener("click", function() {
    /* Read last answer from history or current */
    var text = answerEl.innerText;
    if ((!text || text.length < 10) && conversation.length) {
      text = conversation[conversation.length - 1].answer;
    }
    if (!text || text.length < 10) return;

    if (ttsAudio && !ttsAudio.paused) {
      ttsAudio.pause();
      ttsAudio = null;
      ttsBtn.innerHTML = "&#128264;";
      return;
    }

    ttsBtn.innerHTML = "...";
    fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: text.substring(0, 2000),
        voice: "ff_siwis",
        model: "kokoro",
        response_format: "mp3"
      })
    })
    .then(function(r) { return r.blob(); })
    .then(function(blob) {
      ttsAudio = new Audio(URL.createObjectURL(blob));
      ttsAudio.onended = function() { ttsBtn.innerHTML = "&#128264;"; };
      ttsAudio.play();
      ttsBtn.innerHTML = "&#9209;";
    })
    .catch(function(e) {
      console.error("TTS error:", e);
      ttsBtn.innerHTML = "&#128264;";
    });
  });

  /* Expose go() globally for related chips */
  window.go = go;
})();
</script>
<script>
/* ── Starlink Mesh Network ── */
(function(){
  const c=document.getElementById('starfield');
  const ctx=c.getContext('2d');
  let w,h,nodes=[],mouse={x:-1000,y:-1000};
  const NUM=120;
  const LINK_DIST=150;
  const MOUSE_DIST=200;

  function resize(){w=c.width=window.innerWidth;h=c.height=window.innerHeight}
  window.addEventListener('resize',resize);
  document.addEventListener('mousemove',function(e){mouse.x=e.clientX;mouse.y=e.clientY});
  resize();

  for(let i=0;i<NUM;i++){
    nodes.push({
      x:Math.random()*w,
      y:Math.random()*h,
      vx:(Math.random()-0.5)*0.4,
      vy:(Math.random()-0.5)*0.4,
      r:Math.random()*1.5+0.8,
      o:Math.random()*0.4+0.2
    });
  }

  function draw(){
    ctx.clearRect(0,0,w,h);

    /* update positions */
    for(let n of nodes){
      n.x+=n.vx;
      n.y+=n.vy;
      if(n.x<0||n.x>w)n.vx*=-1;
      if(n.y<0||n.y>h)n.vy*=-1;
      /* slight attraction to mouse */
      const mdx=mouse.x-n.x,mdy=mouse.y-n.y;
      const md=Math.sqrt(mdx*mdx+mdy*mdy);
      if(md<MOUSE_DIST&&md>0){
        n.vx+=mdx/md*0.01;
        n.vy+=mdy/md*0.01;
      }
      /* damping */
      n.vx*=0.999;
      n.vy*=0.999;
    }

    /* draw links */
    for(let i=0;i<nodes.length;i++){
      for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x;
        const dy=nodes[i].y-nodes[j].y;
        const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<LINK_DIST){
          const alpha=0.15*(1-dist/LINK_DIST);
          ctx.beginPath();
          ctx.moveTo(nodes[i].x,nodes[i].y);
          ctx.lineTo(nodes[j].x,nodes[j].y);
          ctx.strokeStyle='rgba(0,240,255,'+alpha+')';
          ctx.lineWidth=0.6;
          ctx.stroke();
        }
      }
    }

    /* draw nodes */
    for(let n of nodes){
      /* glow */
      const g=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r*4);
      g.addColorStop(0,'rgba(0,240,255,'+(n.o*0.6)+')');
      g.addColorStop(1,'rgba(0,240,255,0)');
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r*4,0,Math.PI*2);
      ctx.fillStyle=g;
      ctx.fill();
      /* core */
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,240,255,'+n.o+')';
      ctx.fill();
    }

    /* mouse node connections */
    if(mouse.x>0){
      for(let n of nodes){
        const dx=mouse.x-n.x,dy=mouse.y-n.y;
        const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<MOUSE_DIST){
          const alpha=0.2*(1-dist/MOUSE_DIST);
          ctx.beginPath();
          ctx.moveTo(mouse.x,mouse.y);
          ctx.lineTo(n.x,n.y);
          ctx.strokeStyle='rgba(224,64,251,'+alpha+')';
          ctx.lineWidth=0.8;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE
