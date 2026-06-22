import logging
import hashlib
import math
import re

from openai import OpenAI
import google.generativeai as genai

from app.core.config import Settings

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ0-9]{3,}")


class LLMServiceError(Exception):
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
            raise LLMServiceError(f"Falha na chamada da LLM: {exc}") from exc

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

    def _generate_openai_raw(self, system_prompt: str, user_prompt: str) -> str:
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or _fallback_answer()

    def _generate_gemini(self, question: str, context: str, system_prompt: str) -> str:
        return self._generate_gemini_raw(system_prompt, _build_user_prompt(question, context))

    def _generate_gemini_raw(self, system_prompt: str, user_prompt: str) -> str:
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=self.settings.llm_model,
            system_instruction=system_prompt,
        )
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
