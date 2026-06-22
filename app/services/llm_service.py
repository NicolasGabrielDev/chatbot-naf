import logging
import hashlib
import json
import math
import re

from openai import OpenAI
import google.generativeai as genai

from app.core.config import Settings
from app.services.topic_catalog import simplify_question

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ0-9]{3,}")


class LLMServiceError(Exception):
    pass


class LLMRateLimitError(LLMServiceError):
    pass


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model_used(self) -> str:
        return self.settings.llm_model

    def generate_answer(self, question: str, context: str, system_prompt: str) -> str:
        if not context.strip():
            logger.info("llm.generate.skipped_empty_context")
            return _fallback_answer()

        try:
            logger.info(
                "llm.generate.started provider=%s model=%s question_length=%s context_length=%s",
                self.settings.llm_provider,
                self.settings.llm_model,
                len(question),
                len(context),
            )
            if self.settings.llm_provider == "openai":
                answer = self._generate_openai(question, context, system_prompt)
            else:
                answer = self._generate_gemini(question, context, system_prompt)
            logger.info("llm.generate.completed provider=%s answer_length=%s", self.settings.llm_provider, len(answer))
            return answer
        except Exception as exc:
            logger.exception("llm.generate.failed provider=%s model=%s", self.settings.llm_provider, self.settings.llm_model)
            if _is_rate_limit_error(exc):
                raise LLMRateLimitError("Limite temporário do provedor de IA atingido.") from exc
            raise LLMServiceError(f"Falha na chamada da LLM: {exc}") from exc

    def classify_topics(self, question: str, topics: list[dict]) -> list[str]:
        simplified_question = simplify_question(question)
        catalog = [
            {
                "id": topic["id"],
                "section": topic.get("section", ""),
                "title": topic["title"],
            }
            for topic in topics
        ]
        system_prompt = (
            "Classifique uma dúvida sobre Imposto de Renda usando somente o catálogo fornecido. "
            "Retorne JSON no formato {\"topic_ids\": [\"001\"]}. "
            "Selecione de um a três temas diretamente relacionados, priorizando o tema mais geral quando "
            "a pergunta pedir um conceito. Nunca invente IDs."
        )
        user_prompt = (
            f"Pergunta simplificada: {simplified_question}\n\n"
            f"Catálogo de temas: {json.dumps(catalog, ensure_ascii=False)}"
        )

        try:
            if self.settings.llm_provider == "openai":
                response = self._generate_openai_raw(system_prompt, user_prompt, json_response=True)
            else:
                response = self._generate_gemini_raw(system_prompt, user_prompt, json_response=True)
            topic_ids = _parse_topic_ids(response)
            valid_ids = {topic["id"] for topic in topics}
            if not topic_ids or len(topic_ids) > 3 or any(topic_id not in valid_ids for topic_id in topic_ids):
                raise LLMServiceError("O modelo não identificou temas válidos para a pergunta.")
            return topic_ids
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.exception("llm.topic_classification.failed provider=%s", self.settings.llm_provider)
            if _is_rate_limit_error(exc):
                raise LLMRateLimitError("Limite temporário do provedor de IA atingido.") from exc
            raise LLMServiceError(f"Falha na classificação do tema: {exc}") from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.info(
            "llm.embeddings.documents.started provider=%s model=%s count=%s",
            self.settings.embedding_provider,
            self._embedding_model_used(),
            len(texts),
        )
        if self.settings.embedding_provider == "local":
            embeddings = [_local_embedding(text, self.settings.local_embedding_dimensions) for text in texts]
            logger.info("llm.embeddings.documents.completed provider=local count=%s", len(embeddings))
            return embeddings

        if self.settings.llm_provider == "openai":
            client = OpenAI(api_key=self.settings.openai_api_key)
            response = client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            logger.info("llm.embeddings.documents.completed provider=openai count=%s", len(embeddings))
            return embeddings

        genai.configure(api_key=self.settings.gemini_api_key)
        embeddings = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[start : start + self.settings.embedding_batch_size]
            result = genai.embed_content(
                model=self.settings.gemini_embedding_model,
                content=batch,
                task_type="retrieval_document",
            )
            embeddings.extend(_extract_embeddings(result))
        logger.info("llm.embeddings.documents.completed provider=gemini count=%s", len(embeddings))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        logger.info(
            "llm.embeddings.query.started provider=%s model=%s text_length=%s",
            self.settings.embedding_provider,
            self._embedding_model_used(),
            len(text),
        )
        if self.settings.embedding_provider == "local":
            embedding = _local_embedding(text, self.settings.local_embedding_dimensions)
            logger.info("llm.embeddings.query.completed provider=local dimensions=%s", len(embedding))
            return embedding

        if self.settings.llm_provider == "openai":
            embedding = self.embed_texts([text])[0]
            logger.info("llm.embeddings.query.completed provider=openai dimensions=%s", len(embedding))
            return embedding

        genai.configure(api_key=self.settings.gemini_api_key)
        result = genai.embed_content(
            model=self.settings.gemini_embedding_model,
            content=text,
            task_type="retrieval_query",
        )
        embedding = _extract_embeddings(result)[0]
        logger.info("llm.embeddings.query.completed provider=gemini dimensions=%s", len(embedding))
        return embedding

    def _generate_openai(self, question: str, context: str, system_prompt: str) -> str:
        return self._generate_openai_raw(system_prompt, _build_user_prompt(question, context))

    def _generate_openai_raw(self, system_prompt: str, user_prompt: str, json_response: bool = False) -> str:
        client = OpenAI(api_key=self.settings.openai_api_key)
        request = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if json_response:
            request["response_format"] = {"type": "json_object"}
            request["temperature"] = 0
        response = client.chat.completions.create(
            **request,
        )
        return response.choices[0].message.content or _fallback_answer()

    def _generate_gemini(self, question: str, context: str, system_prompt: str) -> str:
        return self._generate_gemini_raw(system_prompt, _build_user_prompt(question, context))

    def _generate_gemini_raw(self, system_prompt: str, user_prompt: str, json_response: bool = False) -> str:
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=self.settings.llm_model,
            system_instruction=system_prompt,
        )
        if json_response:
            response = model.generate_content(
                user_prompt,
                generation_config={"response_mime_type": "application/json"},
            )
        else:
            response = model.generate_content(user_prompt)
        return response.text or _fallback_answer()

    def _embedding_model_used(self) -> str:
        if self.settings.embedding_provider == "local":
            return f"local-hash-{self.settings.local_embedding_dimensions}"
        if self.settings.llm_provider == "openai":
            return self.settings.openai_embedding_model
        return self.settings.gemini_embedding_model


def _build_user_prompt(question: str, context: str) -> str:
    return (
        "Contexto recuperado da base oficial:\n"
        f"{context}\n\n"
        "Pergunta do cidadão:\n"
        f"{question}\n\n"
        "Responda somente com base no contexto recuperado."
    )


def _fallback_answer() -> str:
    return "Não foi possível localizar uma resposta segura na base consultada."


def _parse_topic_ids(response: str) -> list[str]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise LLMServiceError("O modelo retornou uma classificação de tema inválida.") from exc
    topic_ids = payload.get("topic_ids")
    if not isinstance(topic_ids, list) or not all(isinstance(topic_id, str) for topic_id in topic_ids):
        raise LLMServiceError("O modelo retornou uma classificação de tema inválida.")
    return list(dict.fromkeys(topic_ids))


def _is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return "429" in error_text or "resource_exhausted" in error_text or "quota exceeded" in error_text


def _extract_embeddings(result) -> list[list[float]]:
    embeddings = result["embedding"]
    if not embeddings:
        return []
    if isinstance(embeddings[0], int | float):
        return [embeddings]
    return embeddings


def _local_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return vector

    features = tokens + [f"{first}_{second}" for first, second in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw_index = int.from_bytes(digest[:4], "big")
        raw_sign = int.from_bytes(digest[4:], "big")
        index = raw_index % dimensions
        sign = 1.0 if raw_sign % 2 == 0 else -1.0
        vector[index] += sign

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]
