# DevLabz Search

Moteur de recherche IA self-hosted — une alternative à Perplexity qui tourne entièrement sur votre propre infrastructure.

## C'est quoi ?

DevLabz Search est une application FastAPI en un seul fichier qui enchaîne des outils open-source pour fournir des réponses synthétisées par IA avec citations de sources :

1. **SearXNG** — méta-moteur de recherche (résultats web)
2. **TEI Reranker** — reclassement sémantique des résultats (bge-reranker-v2-m3)
3. **Crawl4AI** — scrape les meilleures pages pour le contenu complet
4. **LLM** — génère une réponse structurée et citée via API compatible OpenAI
5. **Whisper** — entrée vocale (speech-to-text)
6. **Kokoro** — sortie vocale (text-to-speech)

## Fonctionnalités

- **Recherche conversationnelle** — questions de suivi avec historique complet
- **Deux modes** — Rapide (snippets) ou Approfondi (scraping des pages)
- **Entrée/sortie vocale** — parlez votre requête, écoutez la réponse
- **Interface cyberpunk** — fond animé Starlink mesh, cartes glassmorphism
- **Citations de sources** — chaque affirmation liée à sa source `[1]`, `[2]`, etc.
- **Questions liées** — suggestions de suivi auto-générées
- **Zéro dépendance cloud** — tourne sur des LLM locaux (llama.cpp, Ollama, vLLM, etc.)
- **Fichier unique** — tout (backend + frontend) dans un seul `main.py`

## Architecture

```
Requête utilisateur
    │
    ▼
SearXNG ──► Reranker ──► Crawl4AI ──► LLM ──► Réponse streamée
(recherche)  (classement)  (scraping)  (génération)
```

Tous les services communiquent via APIs HTTP. DevLabz Search est l'orchestrateur.

## Démarrage rapide

### Docker Compose (recommandé)

```bash
git clone https://github.com/sk7n4k3d/devlabz-search.git
cd devlabz-search
```

Modifiez `docker-compose.yml` pour pointer vers vos URLs de services, puis :

```bash
docker compose up -d
```

Accessible sur `http://localhost:8889`.

### Manuel

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8889
```

## Configuration

Toute la configuration se fait via variables d'environnement :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `SEARXNG_URL` | `http://10.8.0.101:5456` | URL de l'instance SearXNG |
| `RERANKER_URL` | `http://10.8.0.101:8401` | URL du reranker TEI |
| `CRAWL4AI_URL` | `http://10.8.0.101:11235` | URL de Crawl4AI |
| `LLM_URL` | `http://10.8.0.2:8080` | API LLM compatible OpenAI |
| `LLM_MODEL` | `Qwen3.5-122B-A10B` | Nom du modèle |
| `LLM_API_KEY` | (vide) | Clé API si nécessaire |
| `WHISPER_URL` | `http://10.8.0.101:8300` | API Whisper STT |
| `KOKORO_URL` | `http://10.8.0.101:8880` | API Kokoro TTS |

## Prérequis

Ces services doivent tourner quelque part sur votre réseau :

- [SearXNG](https://github.com/searxng/searxng) — méta-recherche
- [TEI](https://github.com/huggingface/text-embeddings-inference) — modèle de reranking
- [Crawl4AI](https://github.com/unclecode/crawl4ai) — scraper web
- N'importe quelle API LLM compatible OpenAI (llama.cpp, Ollama, vLLM, etc.)
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) (optionnel, entrée vocale)
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (optionnel, sortie vocale)

## API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web |
| `/api/search` | POST | Recherche (flux SSE) |
| `/api/whisper` | POST | Proxy vers Whisper STT |
| `/api/tts` | POST | Proxy vers Kokoro TTS |

### Payload `/api/search`

```json
{
  "query": "votre recherche",
  "mode": "quick",
  "history": []
}
```

- `mode` : `"quick"` (snippets) ou `"deep"` (scraping des pages)
- `history` : tableau de `{role, content}` pour le contexte conversationnel

### Événements SSE

| Événement | Données | Description |
|-----------|---------|-------------|
| `status` | string | Mise à jour de progression |
| `sources` | tableau JSON | Résultats classés |
| `answer` | string JSON | Réponse token par token |
| `related` | tableau JSON | Questions liées |
| `done` | vide | Flux terminé |

## Licence

MIT
