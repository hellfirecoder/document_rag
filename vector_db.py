from __future__ import annotations

import math
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

class QdrantVectorDB:
    def __init__(self, url="http://localhost:6333", collection_name="my_collection" ,dim=768):
        self.client = QdrantClient(url=url,timeout=30)
        self.collection_name = collection_name
        if not self._collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def _collection_exists(self, collection_name: str) -> bool:
        # Compatible across qdrant-client versions where has_collection may not exist.
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def upsert(self,id,vectors, payloads):
        points=[PointStruct(id=id[i], vector=vectors[i], payload=payloads[i]) for i in range(len(id))]
        self.client.upsert(collection_name=self.collection_name, points=points)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())

    def _dense_hits(self, query_vector: list[float], top_k: int) -> list:
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        query_out = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return getattr(query_out, "points", query_out)

    def _all_payload_docs(self, limit: int = 4000) -> list[dict]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        docs: list[dict] = []
        for p in points:
            payload = getattr(p, "payload", None) or {}
            text = payload.get("text", "")
            if not text:
                continue
            docs.append(
                {
                    "id": str(getattr(p, "id", "")),
                    "text": text,
                    "source": payload.get("source", ""),
                }
            )
        return docs

    def _sparse_search(self, query_text: str, top_k: int) -> list[dict]:
        docs = self._all_payload_docs()
        if not docs:
            return []

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        tokenized_docs = [self._tokenize(doc["text"]) for doc in docs]
        lengths = [len(toks) for toks in tokenized_docs]
        avg_doc_len = (sum(lengths) / len(lengths)) if lengths else 0.0

        df: Counter[str] = Counter()
        for toks in tokenized_docs:
            df.update(set(toks))

        k1 = 1.5
        b = 0.75
        scores: list[tuple[int, float]] = []
        n_docs = len(tokenized_docs)

        for i, toks in enumerate(tokenized_docs):
            tf = Counter(toks)
            score = 0.0
            for term in query_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                term_df = df.get(term, 0)
                idf = math.log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
                denom = f + k1 * (1 - b + b * (len(toks) / avg_doc_len if avg_doc_len else 1.0))
                score += idf * (f * (k1 + 1) / denom)

            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        out: list[dict] = []
        for idx, score in scores[:top_k]:
            doc = docs[idx]
            out.append(
                {
                    "id": doc["id"],
                    "score": score,
                    "payload": {"text": doc["text"], "source": doc["source"]},
                }
            )
        return out

    @staticmethod
    def _rrf_fuse(
        dense: list,
        sparse: list[dict],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[dict]:
        score_map: dict[str, float] = {}
        payload_map: dict[str, dict] = {}

        for rank, hit in enumerate(dense, start=1):
            hit_id = str(getattr(hit, "id", ""))
            if not hit_id:
                continue
            score_map[hit_id] = score_map.get(hit_id, 0.0) + 1.0 / (rrf_k + rank)
            payload_map[hit_id] = getattr(hit, "payload", None) or {}

        for rank, hit in enumerate(sparse, start=1):
            hit_id = str(hit.get("id", ""))
            if not hit_id:
                continue
            score_map[hit_id] = score_map.get(hit_id, 0.0) + 1.0 / (rrf_k + rank)
            payload_map[hit_id] = hit.get("payload", {})

        ranked_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)[:top_k]
        return [{"id": rid, "payload": payload_map.get(rid, {}), "score": score_map[rid]} for rid in ranked_ids]
    
    def search(self, query_vector, top_k=5):
        search_result = self._dense_hits(query_vector, top_k)
        
        contents = []
        sources = []
        seen_sources = set()

        for result in search_result:
            payload = getattr(result, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contents.append(text)
            if source and source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)

        return{"contents": contents, "sources":sources}

    def hybrid_search(self, query_text: str, query_vector: list[float], top_k: int = 5, sparse_k: int = 20):
        dense_hits = self._dense_hits(query_vector, max(top_k, sparse_k))
        sparse_hits = self._sparse_search(query_text, sparse_k)
        fused_hits = self._rrf_fuse(dense_hits, sparse_hits, top_k=top_k)

        contents: list[str] = []
        sources: list[str] = []
        seen_sources: set[str] = set()
        for hit in fused_hits:
            payload = hit.get("payload", {})
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contents.append(text)
            if source and source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)

        return {"contents": contents, "sources": sources}
