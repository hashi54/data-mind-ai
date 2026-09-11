from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger
from app.rag.chunking import DocumentChunker
from app.rag.vector_store import vector_store


class RAGPipeline:
    """End-to-end RAG system for document ingestion, indexing, and semantic retrieval."""

    @staticmethod
    def index_knowledge_base(docs_dir: str = None) -> int:
        kb_path = Path(docs_dir or settings.DOCUMENTS_DIR)
        if not kb_path.exists():
            logger.warning(f"Knowledge base path does not exist: {kb_path}")
            return 0

        all_chunks = []
        files = list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))

        dept_lookup = {
            "refund_policy": "Operations & Customer Experience",
            "employee_policy": "Human Resources & Legal",
            "product_catalog": "Product & Merchandising",
            "sales_strategy": "Sales & Business Development",
            "customer_support_policy": "Customer Support & Experience",
            "marketing_strategy": "Marketing & Growth",
        }

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                stem = file_path.stem
                dept = dept_lookup.get(stem, "Corporate")

                metadata = {
                    "document_name": file_path.name,
                    "department": dept,
                    "file_path": str(file_path),
                }

                chunks = DocumentChunker.chunk_markdown(content, metadata)
                all_chunks.extend(chunks)
                logger.info(f"Ingested '{file_path.name}': {len(chunks)} chunks.")

            except Exception as e:
                logger.error(f"Failed to index document {file_path}: {e}")

        vector_store.add_documents(all_chunks)
        return len(all_chunks)

    @staticmethod
    def query(
        question: str,
        department: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Performs semantic search across indexed company documents."""
        # Ensure documents are indexed
        if vector_store.get_document_count() == 0:
            RAGPipeline.index_knowledge_base()

        results = vector_store.search(question, top_k=top_k, department_filter=department)

        # Build context summary
        context_snippets = []
        for r in results:
            context_snippets.append(f"[{r['document_name']} | {r['department']} (p.{r['page_number']})]:\n{r['content']}")

        combined_context = "\n\n---\n\n".join(context_snippets)

        return {
            "question": question,
            "retrieved_chunks": results,
            "combined_context": combined_context,
            "chunks_found": len(results),
        }


if __name__ == "__main__":
    RAGPipeline.index_knowledge_base()
