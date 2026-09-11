import pytest
from app.rag.chunking import DocumentChunker
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import vector_store


def test_document_chunking():
    """Tests chunking of markdown text with metadata preservation."""
    sample_doc = """# Refund Policy\n\n## Section 1\nRefunds are allowed within 14 days.\n\n## Section 2\nElectronics have a 10 day window."""
    meta = {"document_name": "test_refund.md", "department": "Operations"}
    chunks = DocumentChunker.chunk_markdown(sample_doc, meta, max_chunk_size=100)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["document_name"] == "test_refund.md"


def test_rag_query_retrieval():
    """Tests semantic vector search against indexed knowledge base."""
    # Query refund policy
    res = RAGPipeline.query("What is the refund window for electronics?", top_k=2)
    assert res["chunks_found"] > 0
    assert len(res["retrieved_chunks"]) > 0
    top_chunk = res["retrieved_chunks"][0]
    assert "document_name" in top_chunk
    assert "similarity_score" in top_chunk
