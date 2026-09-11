from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.requests import DocumentQueryRequest
from app.schemas.responses import DocumentQueryResponse
from app.rag.pipeline import RAGPipeline
from app.rag.chunking import DocumentChunker
from app.rag.vector_store import vector_store
from app.core.config import settings
from app.core.security import sanitize_filename
from app.core.logging import logger

router = APIRouter(prefix="/documents", tags=["Knowledge Base & RAG"])


@router.post("/query", response_model=DocumentQueryResponse)
def query_knowledge_base(req: DocumentQueryRequest):
    """Performs semantic search and policy question answering over indexed company documents."""
    try:
        rag_res = RAGPipeline.query(
            question=req.question,
            department=req.department,
            top_k=req.top_k or 3,
        )
        top_chunks = rag_res["retrieved_chunks"]
        
        if top_chunks:
            answer = f"**Policy Knowledge Insight:**\n\n{top_chunks[0]['content']}"
        else:
            answer = "No matching documentation found for your query in the knowledge base."

        return DocumentQueryResponse(
            question=req.question,
            answer=answer,
            retrieved_chunks=top_chunks,
        )
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...)):
    """Uploads a markdown, text, or PDF policy document to the RAG knowledge base."""
    safe_name = sanitize_filename(file.filename)
    contents = await file.read()
    
    target_path = Path(settings.DOCUMENTS_DIR) / safe_name
    with open(target_path, "wb") as f:
        f.write(contents)

    # Re-index
    total_indexed = RAGPipeline.index_knowledge_base()
    return {
        "status": "success",
        "message": f"Successfully uploaded and indexed {safe_name}",
        "total_vector_chunks": total_indexed,
    }


@router.get("/list", response_model=List[Dict[str, Any]])
def list_knowledge_base_documents():
    """Lists all active documents in the knowledge base."""
    docs_dir = Path(settings.DOCUMENTS_DIR)
    files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
    
    docs_info = []
    for f in files:
        docs_info.append({
            "document_name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "last_modified": f.stat().st_mtime,
        })
    return docs_info
