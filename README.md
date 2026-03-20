# DevLabz Search

Self-hosted AI-powered search engine — a Perplexity alternative that runs entirely on your own infrastructure.

## What is it?

DevLabz Search is a single-file FastAPI application that chains together open-source tools to deliver AI-synthesized answers with source citations:

1. **SearXNG** — meta-search engine (web results)
2. **TEI Reranker** — semantic reranking of results (bge-reranker-v2-m3)
3. **Crawl4AI** — scrapes top pages for full content
4. **LLM** — generates a cited, structured answer via OpenAI-compatible API
5. **Whisper** — voice input (speech-to-text)
6. **Kokoro** — voice output (text-to-speech)

## Features

- **Conversational search** — follow-up questions with full conversation history
- **Two search modes** — Quick (snippets only) or Deep (full page scraping)
- **Voice input/output** — speak your query, listen to the answer
- **Cyberpunk UI** — animated Starlink mesh background, glassmorphism cards
- **Source citations** — every claim linked to its source with `[1]`, `[2]`, etc.
- **Related questions** — auto-generated follow-up suggestions
- **Zero dependencies on cloud AI** — runs on local LLMs (llama.cpp, Ollama, vLLM, etc.)
- **Single file** — everything (backend + frontend) in one `main.py`

## Architecture

```
User query
    │
    ▼
SearXNG ──► Reranker ──► Crawl4AI ──► LLM ──► Streamed answer
  (search)    (rank)      (scrape)    (generate)
```

All services communicate via HTTP APIs. DevLabz Search is the orchestrator.

## Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/sk7n4k3d/devlabz-search.git
cd devlabz-search
```

Edit `docker-compose.yml` to point to your service URLs, then:

```bash
docker compose up -d
```

Access at `http://localhost:8889`.

### Manual

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8889
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARXNG_URL` | `http://10.8.0.101:5456` | SearXNG instance URL |
| `RERANKER_URL` | `http://10.8.0.101:8401` | TEI reranker URL |
| `CRAWL4AI_URL` | `http://10.8.0.101:11235` | Crawl4AI instance URL |
| `LLM_URL` | `http://10.8.0.2:8080` | OpenAI-compatible LLM API |
| `LLM_MODEL` | `Qwen3.5-122B-A10B` | Model name |
| `LLM_API_KEY` | (empty) | API key if required |
| `WHISPER_URL` | `http://10.8.0.101:8300` | Whisper STT API |
| `KOKORO_URL` | `http://10.8.0.101:8880` | Kokoro TTS API |

## Prerequisites

You need these services running somewhere on your network:

- [SearXNG](https://github.com/searxng/searxng) — meta-search
- [TEI](https://github.com/huggingface/text-embeddings-inference) — reranking model
- [Crawl4AI](https://github.com/unclecode/crawl4ai) — web scraper
- Any OpenAI-compatible LLM API (llama.cpp, Ollama, vLLM, etc.)
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) (optional, for voice input)
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (optional, for voice output)

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend |
| `/api/search` | POST | Search (SSE stream) |
| `/api/whisper` | POST | Proxy to Whisper STT |
| `/api/tts` | POST | Proxy to Kokoro TTS |

### `/api/search` payload

```json
{
  "query": "your search query",
  "mode": "quick",
  "history": []
}
```

- `mode`: `"quick"` (snippets) or `"deep"` (scrape pages)
- `history`: array of `{role, content}` for conversation context

### SSE events

| Event | Data | Description |
|-------|------|-------------|
| `status` | string | Progress update |
| `sources` | JSON array | Ranked search results |
| `answer` | JSON string | Token-by-token answer |
| `related` | JSON array | Related questions |
| `done` | empty | Stream complete |

## License

MIT
