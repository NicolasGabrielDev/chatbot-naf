import re
import unicodedata


SHORT_MESSAGE_FILLERS = {
    "a",
    "as",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "o",
    "os",
    "sobre",
}
SHORT_DECLARATION_PATTERNS = (
    "preciso declarar",
    "tenho que declarar",
    "tenho declarar",
    "sou obrigado declarar",
    "sou obrigado fazer ir",
    "devo declarar",
)
GREETING_PATTERN = re.compile(r"^(oi|ola|olá|bom dia|boa tarde|boa noite|tudo bem|e ai|e aí)[!.? ]*$")
ACCOUNTANT_QUESTION_PATTERN = re.compile(
    r"\b(substitu[io]\w*|troca\w*|dispensa\w*)\b.*\b(contador\w*|contabilista\w*)\b"
    r"|\b(contador\w*|contabilista\w*)\b.*\b(substitu[io]\w*|troca\w*|dispensa\w*)\b"
)
ACCOUNTANT_RECOMMENDATION_PATTERN = re.compile(
    r"\b(recomenda\w*|indica\w*|sugere\w*)\b.*\b(contador\w*|contabilista\w*)\b"
    r"|\b(contador\w*|contabilista\w*)\b.*\b(recomenda\w*|indica\w*|sugere\w*)\b"
)


def answer_direct_question(question: str) -> str | None:
    normalized_question = _normalize(question)

    if ACCOUNTANT_QUESTION_PATTERN.search(normalized_question):
        return (
            "Não. Eu não substituo um contador.\n\n"
            "Posso ajudar com informações gerais sobre Imposto de Renda, explicar conceitos e orientar "
            "o que você deve conferir. Para decisões específicas, casos complexos ou validação final da "
            "declaração, o ideal é consultar um contador ou a Receita Federal."
        )

    if ACCOUNTANT_RECOMMENDATION_PATTERN.search(normalized_question):
        return (
            "Não posso recomendar um contador específico.\n\n"
            "Posso te ajudar com dúvidas gerais sobre Imposto de Renda. Para escolher um profissional, "
            "confira se ele é habilitado, peça referências e avalie se ele tem experiência com o tipo de "
            "declaração que você precisa fazer."
        )

    return None


def should_answer_without_rag(question: str) -> bool:
    normalized_question = _normalize(question)
    return bool(GREETING_PATTERN.match(normalized_question)) or _is_short_unclear_message(normalized_question)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", without_accents).strip()


def _is_short_unclear_message(normalized_question: str) -> bool:
    if any(pattern in normalized_question for pattern in SHORT_DECLARATION_PATTERNS):
        return False

    tokens = re.findall(r"[a-z0-9]+", normalized_question)
    meaningful_tokens = [token for token in tokens if token not in SHORT_MESSAGE_FILLERS]
    return len(normalized_question) <= 30 and len(meaningful_tokens) <= 1
