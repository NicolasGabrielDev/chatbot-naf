import json
from pathlib import Path
import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
STOP_WORDS = {
    "como",
    "das",
    "dos",
    "ela",
    "ele",
    "essa",
    "esse",
    "esta",
    "este",
    "isso",
    "para",
    "pela",
    "pelo",
    "por",
    "que",
    "ser",
    "uma",
}


class TopicCatalogError(Exception):
    pass


def load_topic_catalog(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicCatalogError(f"Falha ao carregar o índice de temas: {exc}") from exc

    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise TopicCatalogError("O índice de temas não possui temas válidos.")
    return topics


def simplify_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKD", question.lower())
    without_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    tokens = TOKEN_PATTERN.findall(without_accents)
    relevant_tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(dict.fromkeys(relevant_tokens))


def pages_for_topics(topics: list[dict], topic_ids: list[str]) -> list[int]:
    topics_by_id = {topic["id"]: topic for topic in topics}
    invalid_ids = [topic_id for topic_id in topic_ids if topic_id not in topics_by_id]
    if invalid_ids:
        raise TopicCatalogError(f"Temas inexistentes retornados pelo modelo: {', '.join(invalid_ids)}")

    pages = {
        page
        for topic_id in topic_ids
        for page in topics_by_id[topic_id].get("pages", [])
        if isinstance(page, int)
    }
    if not pages:
        raise TopicCatalogError("Os temas selecionados não possuem páginas relacionadas.")
    return sorted(pages)
