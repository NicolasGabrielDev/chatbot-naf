import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response, status

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.llm_service import LLMServiceError
from app.services.rag_service import RagService, RagServiceError

router = APIRouter()
logger = logging.getLogger(__name__)

CHAT_ERROR_MESSAGE = "Não foi possível processar sua pergunta agora. Tente novamente mais tarde."


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, response: Response) -> ChatResponse:
    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id
    started_at = perf_counter()
    logger.info(
        "chat.request.started request_id=%s question_length=%s",
        request_id,
        len(payload.question),
    )

    try:
        settings = get_settings()
        logger.info(
            "chat.config.loaded request_id=%s provider=%s model=%s top_k=%s context_file=%s",
            request_id,
            settings.llm_provider,
            settings.llm_model,
            settings.top_k,
            settings.context_file_path,
        )

        service = RagService(settings)
        logger.info("chat.rag.initialized request_id=%s", request_id)

        answer, sources, model_used = service.answer(payload.question)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "chat.request.completed request_id=%s model=%s sources=%s elapsed_ms=%s",
            request_id,
            model_used,
            len(sources),
            elapsed_ms,
        )
        return ChatResponse(answer=answer, sources=sources, model_used=model_used)
    except FileNotFoundError as exc:
        logger.exception("chat.request.failed request_id=%s error_type=file_not_found", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=CHAT_ERROR_MESSAGE,
            headers={"X-Request-ID": request_id},
        ) from exc
    except ValueError as exc:
        logger.exception("chat.request.failed request_id=%s error_type=config", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=CHAT_ERROR_MESSAGE,
            headers={"X-Request-ID": request_id},
        ) from exc
    except LLMServiceError as exc:
        logger.exception("chat.request.failed request_id=%s error_type=llm", request_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=CHAT_ERROR_MESSAGE,
            headers={"X-Request-ID": request_id},
        ) from exc
    except RagServiceError as exc:
        logger.exception("chat.request.failed request_id=%s error_type=rag", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=CHAT_ERROR_MESSAGE,
            headers={"X-Request-ID": request_id},
        ) from exc
    except Exception as exc:
        logger.exception("chat.request.failed request_id=%s error_type=unexpected", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=CHAT_ERROR_MESSAGE,
            headers={"X-Request-ID": request_id},
        ) from exc
