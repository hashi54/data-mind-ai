from fastapi import APIRouter, HTTPException, status
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse
from app.llm.analyst_agent import BusinessAnalystOrchestrator
from app.core.logging import logger

router = APIRouter(prefix="/chat", tags=["AI Business Analyst Chat"])


@router.post("", response_model=ChatResponse)
def chat_with_analyst(request: ChatRequest):
    """
    Unified AI Analyst chat endpoint.
    Accepts questions like:
    - 'Why did sales decrease in August?'
    - 'What were our sales in Kerala last month?'
    - 'Which customers should our sales team contact this week?'
    - 'According to our refund policy, how much revenue did we lose through refunds?'
    """
    try:
        result = BusinessAnalystOrchestrator.answer_question(
            question=request.question,
            session_id=request.session_id or "default_session",
        )
        return ChatResponse(
            answer=result["answer"],
            intent=result.get("intent", "GENERAL_CHAT"),
            sources=result.get("sources", []),
            sql=result.get("sql"),
            data_preview=result.get("data_preview"),
            chart_spec=result.get("chart_spec"),
            confidence=result.get("confidence", 0.95),
            limitations=result.get("limitations"),
        )
    except Exception as e:
        logger.error(f"Error handling chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process inquiry: {str(e)}",
        )
