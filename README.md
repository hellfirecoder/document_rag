
# Document RAG (Offline-First)

A fully local Retrieval Augmented Generation system for asking questions against your PDF documents — no cloud, no API costs, no data leaving your machine.

---

## What This Does

I built this because I kept running into the same problem: cloud LLMs are powerful, but they're a bad fit when the documents you're working with are sensitive, when you want predictable costs, or when you just don't want to depend on an external API that can go down or rate-limit you at the worst moment.

This project lets you upload a PDF, chunk and embed it locally using Ollama, store the vectors in a local Qdrant instance, and then ask natural language questions against it — all through a clean Streamlit UI. Everything runs on your machine.

---

## Features

**Privacy-first by design.** Your PDFs, embeddings, and vectors never leave your environment. The LLM runs locally through Ollama, so there's no third-party involved at any point.

**Event-driven ingestion pipeline.** Uploading a PDF triggers a `rag/inngest_pdf` event, which kicks off an Inngest function that loads the document, chunks the text, generates embeddings via Ollama, and upserts everything into Qdrant — all asynchronously and without blocking the UI.

**Hybrid retrieval.** You can switch between dense vector search and hybrid mode, which fuses dense results with sparse lexical search for better coverage across documents where keyword matching matters as much as semantic similarity.

**Reranking.** Before the final answer is generated, an optional reranking stage reorders the retrieved chunks by relevance. You can toggle this per query — useful when you want higher precision without increasing Top-K.

**Configurable chunking.** You control chunk size and overlap at ingestion time. Defaults work well out of the box, but you can tune them if your documents have unusual structure.

**Model flexibility.** Any Ollama-compatible embedding or chat model works. I've tested with `qwen3-embedding:0.6b` for embeddings and `llama2:latest` for chat, but you can swap in whatever you have pulled locally. The app automatically avoids passing your question to an embedding-only model by mistake, and models can be refreshed directly from the UI.

**Retrieval evaluation benchmarks.** A benchmark script lets you measure retrieval quality against a labeled JSONL dataset. It outputs hit rate, MRR, and recall@k — handy if you're comparing embedding models or tuning retrieval settings.

**Persistent conversation history.** Every question and answer is saved locally in SQLite, so you can review prior Q&A sessions without re-running queries.

**Source transparency.** Answers come with the source chunks that were used to generate them, so you can verify what the model actually saw.

**One-command startup.** A Docker Compose file spins up Qdrant, the FastAPI server, Streamlit, and Inngest together, so you don't have to manage four terminal windows just to get started.

---

## Architecture

```
Streamlit UI
    │
    ├── Upload PDF → rag/inngest_pdf event
    │       └── Load PDF → Chunk text → Embed (Ollama) → Upsert (Qdrant)
    │
    └── Ask question → rag/query_pdf_ai event
            └── Embed query → Search Qdrant (dense or hybrid)
                    └── Optional rerank → Chat (Ollama) → Answer + sources
```

Main files:
- `streamlit_app.py` — UI for upload, query, and history
- `main.py` — FastAPI app hosting the Inngest functions
- `data_loader.py` — PDF loading, chunking, embedding
- `vector_db.py` — Qdrant collection management, upsert, and search
- `reranker.py` — Lexical reranking utility
- `eval/benchmark_retrieval.py` — Retrieval benchmark script

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running
- Qdrant running locally at `http://localhost:6333`
- Inngest dev server

Recommended models:
- Embedding: `qwen3-embedding:0.6b`
- Chat: `llama2:latest`

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd document_rag
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -e .
```

If the editable install fails:

```bash
pip install fastapi inngest llama-index-core llama-index-readers-file ollama python-dotenv qdrant-client streamlit uvicorn
```

### 4. Start Ollama and pull models

```bash
ollama serve
```

In a separate terminal:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull llama2:latest
```

### 5. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 6. Start the Inngest dev server

```bash
inngest dev
```

This exposes the local event runtime at `http://127.0.0.1:8288/v1`.

### 6b. One-command startup with Docker Compose (optional)

If you'd rather not manage services manually, this starts everything together:

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

### 7. Start the FastAPI server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 8. Start the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Using It

1. Upload a PDF from the UI
2. Set your chunk size and overlap (defaults are fine to start)
3. Pick an embedding model and click **Ingest PDF**
4. Once ingestion finishes, type your question
5. Choose Top-K, a chat model, retrieval mode (`dense` or `hybrid`), and rerank settings
6. Click **Ask**
7. Review the answer and the source chunks it came from
8. Use the history toggle to revisit previous Q&A

---

## Running Retrieval Benchmarks

Prepare a JSONL file where each line has a `question` and `expected_source` field, then run:

```bash
python eval/benchmark_retrieval.py \
  --dataset path/to/your_eval_dataset.jsonl \
  --retrieval-mode hybrid \
  --rerank-enabled \
  --top-k 5
```

Outputs: `hit_rate`, `mrr`, and `recall_at_k`.

---

## Troubleshooting

**`Connection refused` to Ollama** — Make sure `ollama serve` is running on `http://localhost:11434`.

**No results from retrieval** — Confirm ingestion completed successfully, and that you're using the same embedding model for querying that you used during ingestion.

**Qdrant errors** — Make sure Qdrant is reachable at `http://localhost:6333`.

**Inngest polling errors** — Check that both `inngest dev` and the FastAPI server are running and reachable.

---

## Security and Privacy

This is designed for local use. PDF content, embeddings, and LLM calls stay entirely on your machine.

---

