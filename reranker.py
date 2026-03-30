from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def rerank_contexts(question: str, contexts: list[str], top_n: int = 5) -> list[str]:
    """Lightweight lexical reranker to prioritize context relevance before generation."""
    if not contexts:
        return []

    q_tokens = _tokenize(question)
    if not q_tokens:
        return contexts[:top_n]

    q_tf = Counter(q_tokens)
    scored: list[tuple[float, str]] = []

    for ctx in contexts:
        c_tokens = _tokenize(ctx)
        if not c_tokens:
            continue
        c_tf = Counter(c_tokens)

        overlap = sum(min(q_tf[t], c_tf.get(t, 0)) for t in q_tf)
        coverage = overlap / max(1, len(set(q_tokens)))
        density = overlap / max(1, len(c_tokens))
        score = (coverage * 0.8) + (density * 0.2)
        scored.append((score, ctx))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [ctx for _, ctx in scored]

    seen: set[str] = set()
    unique: list[str] = []
    for item in reranked:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)

    return unique[:top_n]
