import logging
import time
from typing import Any

from fastapi import FastAPI
import inngest
import inngest.fast_api
import uuid
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError, ReadTimeout, RequestException
from data_loader import load_and_chunk_pdf, embed_texts
from reranker import rerank_contexts
from vector_db import QdrantVectorDB



inngest_client=inngest.Inngest(
    app_id="rag",
    logger=logging.getLogger("uvicorn"),
    is_production=False
)


def _collection_name_for_model(embed_model: str | None) -> str:
    if not embed_model:
        return "my_collection"
    safe = (
        embed_model.lower()
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    return f"rag_{safe}"

@inngest_client.create_function(
    fn_id="RAG:Inngest",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)

async def rag_pdf(ctx: inngest.Context):
    def _ingest() -> dict:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunk_size = int(ctx.event.data.get("chunk_size", 512))
        chunk_overlap = int(ctx.event.data.get("chunk_overlap", 100))
        chunks = load_and_chunk_pdf(
            pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embed_model = ctx.event.data.get("embed_model")
        vecs = embed_texts(chunks, model=embed_model)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        collection_name = _collection_name_for_model(embed_model)
        QdrantVectorDB(collection_name=collection_name, dim=len(vecs[0])).upsert(ids, vecs, payloads)
        return {"ingested": len(chunks)}

    ingested = await ctx.step.run("ingest-pdf", _ingest)
    return ingested


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5, retrieval_mode: str = "dense") -> dict:
        embed_model = ctx.event.data.get("embed_model")
        query_vec = embed_texts([question], model=embed_model)[0]
        collection_name = _collection_name_for_model(embed_model)
        store = QdrantVectorDB(collection_name=collection_name, dim=len(query_vec))
        mode = retrieval_mode if retrieval_mode in {"dense", "hybrid"} else "dense"

        if mode == "hybrid":
            found = store.hybrid_search(question, query_vec, top_k=top_k, sparse_k=max(20, top_k * 4))
        else:
            found = store.search(query_vec, top_k)

        rerank_enabled = bool(ctx.event.data.get("rerank_enabled", True))
        rerank_top_n = int(ctx.event.data.get("rerank_top_n", top_k))
        contexts = found.get("contents", [])
        if rerank_enabled:
            contexts = rerank_contexts(question, contexts, top_n=rerank_top_n)

        return {"contexts": contexts, "sources": found["sources"]}

    def _get_ollama_models() -> list[str]:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def _pick_default_chat_model() -> str:
        models = _get_ollama_models()
        non_embed = [m for m in models if "embed" not in m.lower()]
        if non_embed:
            return non_embed[0]
        if models:
            return models[0]
        raise RuntimeError(
            "No Ollama models detected. Pull a model first (for example: 'ollama pull llama3.2')."
        )

    def ollama_chat(
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        read_timeout_seconds: int = 240,
        retries: int = 1,
    ) -> str:
        chosen_model = (model or _pick_default_chat_model()).strip()
        if "embed" in chosen_model.lower():
            models = _get_ollama_models()
            non_embed = [m for m in models if "embed" not in m.lower()]
            if non_embed:
                chosen_model = non_embed[0]
            else:
                raise RuntimeError(
                    f"Selected model '{chosen_model}' is an embedding model. Choose a chat-capable model."
                )

        url = "http://localhost:11434/api/chat"
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        response = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=(10, read_timeout_seconds))
                break
            except ReadTimeout as exc:
                if attempt < retries:
                    time.sleep(1.5)
                    continue
                raise RuntimeError(
                    f"Ollama timed out after {read_timeout_seconds}s while generating with model '{chosen_model}'. "
                    "Try reducing max_tokens or using a faster model."
                ) from exc
            except RequestsConnectionError as exc:
                raise RuntimeError(
                    "Cannot connect to Ollama at http://localhost:11434. "
                    "Start Ollama first (for example: 'ollama serve')."
                ) from exc
            except RequestException as exc:
                raise RuntimeError(f"Ollama request failed: {exc}") from exc

        if response is None:
            raise RuntimeError("Ollama request did not return a response.")

        if response.status_code >= 400:
            detail = response.text.strip()
            raise RuntimeError(f"Ollama chat failed ({response.status_code}) for model '{chosen_model}': {detail}")

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc
        if "message" in data:
            return data["message"]["content"].strip()
        elif "response" in data:
            return data["response"].strip()
        return str(data)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    retrieval_mode = str(ctx.event.data.get("retrieval_mode", "dense")).lower()

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k, retrieval_mode=retrieval_mode))

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    messages = [
        {"role": "system", "content": "You answer questions using only the provided context."},
        {"role": "user", "content": user_content}
    ]

    llm_model = ctx.event.data.get("llm_model")
    read_timeout_seconds = int(ctx.event.data.get("llm_timeout_seconds", 240))

    answer = await ctx.step.run(
        "llm-answer",
        lambda: ollama_chat(
            messages,
            model=llm_model,
            max_tokens=1024,
            temperature=0.2,
            read_timeout_seconds=read_timeout_seconds,
            retries=1,
        )
    )

    return {"answer": answer, "sources": found["sources"], "num_contexts": len(found["contexts"])}




app=FastAPI()


inngest.fast_api.serve(app,inngest_client,[rag_pdf, rag_query_pdf_ai])