import logging
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

import chromadb

from app.core.config import Settings
from app.schemas.chat import Source
from app.services.context_loader import load_context_sections
from app.services.llm_service import LLMService, LLMServiceError
from app.services.topic_catalog import load_topic_catalog, pages_for_topics

logger = logging.getLogger(__name__)
SEARCH_CANDIDATE_MULTIPLIER = 10
MAX_SEARCH_CANDIDATES = 50
SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
SEARCH_STOP_WORDS = {
    "como",
    "das",
    "declaracao",
    "dos",
    "ela",
    "ele",
    "esta",
    "este",
    "imposto",
    "para",
    "pela",
    "pelo",
    "que",
    "renda",
    "ser",
    "uma",
}


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
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.exception("rag.index.failed")
            if isinstance(exc, RagServiceError):
                raise
            raise RagServiceError(f"Falha na indexação do contexto: {exc}") from exc

    def search(self, question: str) -> list[Source]:
        try:
            topics = load_topic_catalog(self.settings.topic_index_path)
            topic_ids = self.llm_service.classify_topics(question, topics)
            related_pages = pages_for_topics(topics, topic_ids)
            page_filter = {"page": {"$in": related_pages}}
            logger.info("rag.search.topics_selected ids=%s pages=%s", topic_ids, related_pages)
            indexed_documents = self.collection.get(
                where=page_filter,
                include=["documents", "metadatas"],
            )
            lexical_candidates = [
                Source(content=document, metadata=metadata or {})
                for document, metadata in zip(
                    indexed_documents.get("documents", []),
                    indexed_documents.get("metadatas", []),
                )
            ]
            if not lexical_candidates:
                raise RagServiceError("Nenhum trecho indexado foi encontrado nas páginas dos temas selecionados.")

            logger.info("rag.search.embedding_query.started question_length=%s", len(question))
            query_embedding = self.llm_service.embed_query(question)
            logger.info("rag.search.embedding_query.completed dimensions=%s", len(query_embedding))
            candidate_count = min(
                len(lexical_candidates),
                max(self.settings.top_k * SEARCH_CANDIDATE_MULTIPLIER, self.settings.top_k),
                MAX_SEARCH_CANDIDATES,
            )
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_count,
                where=page_filter,
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            candidates = [
                Source(content=document, metadata=metadata or {}, distance=distance)
                for document, metadata, distance in zip(documents, metadatas, distances)
            ]
            return _merge_search_results(question, lexical_candidates, candidates, self.settings.top_k)
        except LLMServiceError:
            raise
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

    normalized_title = unicodedata.normalize("NFKD", lines[0].lower())
    title_without_accents = "".join(
        character for character in normalized_title if not unicodedata.combining(character)
    )
    if title_without_accents in {"sumario", "indice"}:
        return True

    short_lines = sum(1 for line in lines if len(line) <= 70)
    numeric_lines = sum(1 for line in lines if line.isdigit())
    if len(lines) >= 25 and short_lines / len(lines) >= 0.8 and numeric_lines >= 8:
        return True

    return False


def _rerank_sources(question: str, sources: list[Source]) -> list[Source]:
    query_tokens = set(_search_tokens(question)) - SEARCH_STOP_WORDS
    if not query_tokens:
        return sources

    ranked_sources = []
    for vector_position, source in enumerate(sources):
        lexical_score = _lexical_score(query_tokens, source.content)
        ranked_sources.append((lexical_score, vector_position, source))

    ranked_sources.sort(key=lambda item: (-item[0], item[1]))
    return [source for _, _, source in ranked_sources]


def _merge_search_results(
    question: str,
    lexical_candidates: list[Source],
    vector_candidates: list[Source],
    result_limit: int,
) -> list[Source]:
    query_tokens = set(_search_tokens(question)) - SEARCH_STOP_WORDS
    lexical_results = [
        source
        for source in _rerank_sources(question, lexical_candidates)
        if _lexical_score(query_tokens, source.content) > 0
    ]

    merged_results = []
    seen_contents = set()
    for source in lexical_results + vector_candidates:
        if source.content in seen_contents:
            continue
        seen_contents.add(source.content)
        merged_results.append(source)
        if len(merged_results) == result_limit:
            break
    return merged_results


def _lexical_score(query_tokens: set[str], document: str) -> int:
    document_tokens = _search_tokens(document)
    document_token_set = set(document_tokens)
    opening_token_set = set(_search_tokens(document[:400]))
    score = 0
    for query_token in query_tokens:
        exact_count = document_tokens.count(query_token)
        if exact_count:
            score += 2 + min(exact_count, 3)
            if query_token in opening_token_set:
                score += 1
            continue
        if len(query_token) >= 5 and any(
            len(document_token) >= 5 and query_token[:5] == document_token[:5]
            for document_token in document_token_set
        ):
            score += 1
            if any(
                len(opening_token) >= 5 and query_token[:5] == opening_token[:5]
                for opening_token in opening_token_set
            ):
                score += 1
    return score


def _search_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    return SEARCH_TOKEN_PATTERN.findall(without_accents)
