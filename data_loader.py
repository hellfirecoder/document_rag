import concurrent.futures
from typing import Optional

import requests
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader


OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_DIM = 768


def get_ollama_embedding(text: str, model: Optional[str] = None) -> list[float]:
    embed_model = model or OLLAMA_EMBED_MODEL
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": embed_model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def load_and_chunk_pdf(path: str, chunk_size: int = 512, chunk_overlap: int = 100) -> list[str]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def embed_texts(
    texts: list[str],
    model: Optional[str] = None,
    max_workers: int = 4,
) -> list[list[float]]:
    if not texts:
        return []

    def safe_embed(text: str) -> Optional[list[float]]:
        try:
            return get_ollama_embedding(text, model=model)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(safe_embed, texts))

    embeddings = [emb for emb in results if emb is not None]
    if not embeddings:
        raise RuntimeError("Embedding failed for all input texts")
    return embeddings
