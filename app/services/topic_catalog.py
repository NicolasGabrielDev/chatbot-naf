import json
from pathlib import Path
import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
KNOWN_FISCAL_ACRONYMS = {"cpf", "cnpj", "mei", "irpf", "irpj", "pix"}
QUERY_EXPANSIONS = {
    "academia": ["ginastica", "esporte", "despesas", "instrucao"],
    "automovel": ["veiculo", "automotor", "bens", "direitos"],
    "carro": ["veiculo", "automotor", "bens", "direitos"],
    "financiado": ["financiamento", "alienacao", "fiduciaria"],
    "financiamento": ["financiado", "alienacao", "fiduciaria"],
    "heranca": ["transferencias", "patrimoniais", "doacoes", "herancas", "bens", "direitos"],
    "herdei": ["heranca", "transferencias", "patrimoniais", "doacoes", "herancas"],
    "isento": ["rendimentos", "isentos", "nao", "tributaveis"],
    "izento": ["isento", "rendimentos", "isentos", "nao", "tributaveis"],
    "pix": ["restituicao", "chave", "cpf"],
    "veiculo": ["automotor", "bens", "direitos"],
}
STOP_WORDS = {
    "caso",
    "como",
    "das",
    "dos",
    "ela",
    "ele",
    "entra",
    "entrar",
    "entre",
    "essa",
    "esse",
    "esta",
    "este",
    "fazer",
    "feito",
    "isso",
    "meu",
    "meus",
    "minha",
    "minhas",
    "onde",
    "para",
    "pela",
    "pelo",
    "pode",
    "podem",
    "por",
    "preciso",
    "qual",
    "quais",
    "quando",
    "quem",
    "que",
    "sobre",
    "ser",
    "tenha",
    "tenho",
    "tipo",
    "valor",
    "valores",
    "devo",
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
    relevant_tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS and (len(token) >= 4 or token in KNOWN_FISCAL_ACRONYMS)
    ]
    expanded_tokens = []
    for token in relevant_tokens:
        expanded_tokens.append(token)
        expanded_tokens.extend(QUERY_EXPANSIONS.get(token, []))
    return " ".join(dict.fromkeys(expanded_tokens))


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
