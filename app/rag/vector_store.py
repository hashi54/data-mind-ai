import re
import math
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.logging import logger


class EmbeddingEngine:
    """Computes vector representations for document chunks and user search queries."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000,
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        return embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Embedding engine has not been fitted with corpus yet.")
        return self.vectorizer.transform(texts).toarray()


class VectorStore:
    """In-memory semantic vector store with metadata filtering and similarity ranking."""

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])
        self.embedding_engine = EmbeddingEngine()

    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        self.chunks.extend(chunks)
        all_texts = [c["content"] for c in self.chunks]
        self.embeddings = self.embedding_engine.fit_transform(all_texts)
        logger.info(f"Vector store indexed {len(self.chunks)} total chunks.")

    def search(
        self,
        query: str,
        top_k: int = 3,
        department_filter: str = None,
    ) -> List[Dict[str, Any]]:
        if not self.chunks or len(self.embeddings) == 0:
            return []

        query_vec = self.embedding_engine.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings)[0]

        ranked_indices = np.argsort(scores)[::-1]
        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            dept = chunk["metadata"].get("department", "")

            # Apply department filter if specified
            if department_filter and department_filter.lower() not in dept.lower():
                continue

            similarity = float(scores[idx])
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["metadata"].get("document_name", "Unknown"),
                "department": dept,
                "page_number": chunk["metadata"].get("chunk_index", 1),
                "similarity_score": round(similarity, 4),
                "content": chunk["content"],
            })

            if len(results) >= top_k:
                break

        return results

    def get_document_count(self) -> int:
        return len(self.chunks)


vector_store = VectorStore()
