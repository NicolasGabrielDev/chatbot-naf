import logging
from pathlib import Path
from uuid import uuid4

import chromadb

from app.core.config import Settings
from app.schemas.chat import Source
from app.services.context_loader import load_context_sections
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RagServiceError(Exception):
    pass


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm_service = LLMService(settings)
        logger.info(
            "rag.vectorstore.connecting path=%s collection=%s",
            settings.vectorstore_path,
            settings.chroma_collection_name,
        )
        self.client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection_name)
        logger.info("rag.vectorstore.ready collection=%s", settings.chroma_collection_name)

    def index_context(self) -> int:
        try:
            logger.info("rag.index.started context_file=%s", self.settings.context_file_path)
            self.settings.validate_runtime()
            sections = load_context_sections(self.settings.context_file_path)
            chunks_with_metadata = split_sections(sections, self.settings.chunk_size, self.settings.chunk_overlap)
            chunks = [item["text"] for item in chunks_with_metadata]
            logger.info(
                "rag.index.chunked chunks=%s sections=%s chunk_size=%s chunk_overlap=%s",
                len(chunks),
                len(sections),
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            if not chunks:
                raise RagServiceError("Arquivo de contexto sem conteúdo indexável.")

            logger.info("rag.index.embedding.started chunks=%s provider=%s", len(chunks), self.settings.llm_provider)
            embeddings = self.llm_service.embed_texts(chunks)
            logger.info("rag.index.embedding.completed embeddings=%s", len(embeddings))
            self.client.delete_collection(self.settings.chroma_collection_name)
            self.collection = self.client.get_or_create_collection(name=self.settings.chroma_collection_name)
            self.collection.add(
                ids=[str(uuid4()) for _ in chunks],
                documents=chunks,
                embeddings=embeddings,
                metadatas=[
                    {
                        **item["metadata"],
                        "chunk": index,
                    }
                    for index, item in enumerate(chunks_with_metadata)
                ],
            )
            logger.info("rag.index.completed chunks=%s", len(chunks))
            return len(chunks)
        except Exception as exc:
            logger.exception("rag.index.failed")
            if isinstance(exc, RagServiceError):
                raise
            raise RagServiceError(f"Falha na indexação do contexto: {exc}") from exc

    def search(self, question: str) -> list[Source]:
        try:
            logger.info("rag.search.embedding_query.started question_length=%s", len(question))
            query_embedding = self.llm_service.embed_query(question)
            logger.info("rag.search.embedding_query.completed dimensions=%s", len(query_embedding))
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.settings.top_k,
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                Source(content=document, metadata=metadata or {}, distance=distance)
                for document, metadata, distance in zip(documents, metadatas, distances)
            ]
        except Exception as exc:
            logger.exception("rag.search.failed")
            raise RagServiceError(f"Falha na busca semântica: {exc}") from exc

    def answer(self, question: str) -> tuple[str, list[Source], str]:
        logger.info("rag.answer.started")
        self.settings.validate_runtime()
        indexed_count = self.collection.count()
        logger.info("rag.answer.collection_count count=%s", indexed_count)
        if indexed_count == 0:
            logger.info("rag.answer.no_indexed_context")
            return "Não foi possível localizar uma resposta segura na base consultada.", [], self.llm_service.model_used

        sources = self.search(question)
        logger.info("rag.answer.sources_recovered count=%s", len(sources))
        if not sources:
            return "Não foi possível localizar uma resposta segura na base consultada.", [], self.llm_service.model_used

        context = "\n\n---\n\n".join(source.content for source in sources)
        system_prompt = Path(self.settings.system_prompt_path).read_text(encoding="utf-8")
        logger.info(
            "rag.answer.llm.started context_length=%s system_prompt_length=%s",
            len(context),
            len(system_prompt),
        )
        answer = self.llm_service.generate_answer(question, context, system_prompt)
        logger.info("rag.answer.completed answer_length=%s", len(answer))
        return answer, sources, self.llm_service.model_used


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_sections(sections: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks = []
    for section in sections:
        if _is_low_value_section(section["text"]):
            continue
        for chunk in split_text(section["text"], chunk_size, chunk_overlap):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": dict(section["metadata"]),
                }
            )
    return chunks


def _is_low_value_section(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    lowered = text.lower()
    if "sumário" in lowered or "índice" in lowered:
        return True

    short_lines = sum(1 for line in lines if len(line) <= 70)
    numeric_lines = sum(1 for line in lines if line.isdigit())
    if len(lines) >= 25 and short_lines / len(lines) >= 0.8 and numeric_lines >= 8:
        return True

    return False
