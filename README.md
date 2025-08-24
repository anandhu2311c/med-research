# Smart Medical Literature Assistant

A full-stack application for automated literature search across PubMed, arXiv, and ClinicalTrials.gov with AI-powered summarization and evidence extraction.

## Architecture

- **Backend**: FastAPI + LangServe + LangGraph
- **Frontend**: React + Vite + TypeScript
- **LLM**: Groq API (Llama-3 family)
- **Vector DB**: Pinecone (Starter tier)
- **Embeddings**: Hugging Face sentence-transformers (local)
- **Observability**: Langfuse
- **Database**: SQLite + optional Supabase

## Features

- Automated literature search across multiple sources
- Deduplication and ranking of papers
- Evidence-based summaries with citations
- Interactive UI with filters and downloadable reports
- Full observability with traces and metrics

## Quick Start

1. Clone and install dependencies
2. Set up environment variables
3. Run backend: `uvicorn app.main:app --reload`
4. Run frontend: `cd frontend && npm run dev`

## API Setup

You'll need free API keys from:

1. **Groq** (LLM): [console.groq.com](https://console.groq.com) - 14,400 requests/day free
2. **Pinecone** (Vector DB): [pinecone.io](https://pinecone.io) - 100K vectors free
3. **Langfuse** (Observability): [cloud.langfuse.com](https://cloud.langfuse.com) - 50K observations/month free

**Detailed setup instructions**: See [docs/api-setup.md](docs/api-setup.md)

## Quick Setup

```bash
# 1. Copy environment template
copy .env.example .env

# 2. Edit .env with your API keys (see docs/api-setup.md)

# 3. Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# 4. Verify setup
python scripts/verify_setup.py

# 5. Run the application
uvicorn app.main:app --reload
# In another terminal: cd frontend && npm run dev
```