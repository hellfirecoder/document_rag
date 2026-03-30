from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from data_loader import embed_texts
from reranker import rerank_contexts
from vector_db import QdrantVectorDB


@dataclass
class EvalRow:
    question: str
    expected_source: str


def collection_name_for_model(embed_model: str | None) -> str:
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


def load_eval_rows(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        question = str(payload.get("question", "")).strip()
        expected_source = str(payload.get("expected_source", "")).strip()
        if question and expected_source:
            rows.append(EvalRow(question=question, expected_source=expected_source))
    return rows


def first_relevant_rank(sources: list[str], expected_source: str) -> int | None:
    for idx, source in enumerate(sources, start=1):
        if source == expected_source:
            return idx
    return None


def evaluate(
    dataset: Path,
    embed_model: str,
    top_k: int,
    retrieval_mode: str,
    rerank_enabled: bool,
) -> dict:
    rows = load_eval_rows(dataset)
    if not rows:
        raise RuntimeError("No valid rows found. Each JSONL row must include question and expected_source.")

    collection = collection_name_for_model(embed_model)
    first_vec = embed_texts(["health-check"], model=embed_model)[0]
    store = QdrantVectorDB(collection_name=collection, dim=len(first_vec))

    hits = 0
    rr_total = 0.0
    recall_at_k = 0

    for row in rows:
        query_vec = embed_texts([row.question], model=embed_model)[0]

        if retrieval_mode == "hybrid":
            found = store.hybrid_search(row.question, query_vec, top_k=top_k, sparse_k=max(20, top_k * 4))
        else:
            found = store.search(query_vec, top_k=top_k)

        contexts = found.get("contents", [])
        if rerank_enabled:
            contexts = rerank_contexts(row.question, contexts, top_n=top_k)

        sources = found.get("sources", [])
        rank = first_relevant_rank(sources, row.expected_source)

        if rank is not None:
            hits += 1
            rr_total += 1.0 / rank
            if rank <= top_k:
                recall_at_k += 1

    n = len(rows)
    return {
        "num_queries": n,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "rerank_enabled": rerank_enabled,
        "hit_rate": hits / n,
        "mrr": rr_total / n,
        "recall_at_k": recall_at_k / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for local RAG")
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.sample.jsonl"))
    parser.add_argument("--embed-model", type=str, default="qwen3-embedding:0.6b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", type=str, choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-enabled", action="store_true")
    parser.add_argument("--save-run-path", type=Path, default=Path("eval/benchmark_runs.jsonl"))
    parser.add_argument("--no-save-run", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(
        dataset=args.dataset,
        embed_model=args.embed_model,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        rerank_enabled=args.rerank_enabled,
    )
    metrics["embed_model"] = args.embed_model
    metrics["dataset"] = str(args.dataset)
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if not args.no_save_run:
        args.save_run_path.parent.mkdir(parents=True, exist_ok=True)
        with args.save_run_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
