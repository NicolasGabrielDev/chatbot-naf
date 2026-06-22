import argparse
import json
from pathlib import Path
import re

MIN_EXTRACTED_CHARS = 1000
DEFAULT_INPUT_PATH = Path("./data/context.pdf")
DEFAULT_OUTPUT_PATH = Path("./data/contexto.md")
DEFAULT_INDEX_PATH = Path("./data/context_index.json")
QUESTION_PATTERN = re.compile(r"^(\d{3})\s+[—-]\s+(.+)$")
DERIVED_TOPICS = [
    {
        "id": "conceito_restituicao",
        "section": "RESTITUIÇÃO / COMPENSAÇÃO DO IR",
        "title": "Conceito de restituição do Imposto de Renda: devolução de imposto pago acima do valor devido",
        "pages": [30, 48],
    }
]


class ExtractionError(Exception):
    pass


def extract_with_pymupdf(pdf_path: Path) -> list[dict]:
    try:
        import fitz
    except ImportError as exc:
        raise ExtractionError("PyMuPDF não está instalado. Instale as dependências do requirements.txt.") from exc

    pages = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text")
            pages.append({"page": index, "text": normalize_text(text)})
    return pages


def extract_with_pdfplumber(pdf_path: Path) -> list[dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ExtractionError("pdfplumber não está instalado para fallback de extração.") from exc

    pages = []
    with pdfplumber.open(pdf_path) as document:
        for index, page in enumerate(document.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": index, "text": normalize_text(text)})
    return pages


def extract_pdf(pdf_path: Path, min_chars: int) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    pages = extract_with_pymupdf(pdf_path)
    total_chars = count_chars(pages)
    if total_chars >= min_chars:
        return pages

    print("PyMuPDF extraiu pouco texto. Tentando fallback com pdfplumber...")
    pages = extract_with_pdfplumber(pdf_path)
    total_chars = count_chars(pages)
    if total_chars < min_chars:
        raise ExtractionError(
            "Pouco texto foi extraído do PDF. O arquivo pode ser escaneado e precisa de OCR antes da indexação."
        )
    return pages


def save_markdown(pages: list[dict], output_path: Path, source_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"<!-- source: {source_path} -->",
        "",
    ]

    for page in pages:
        if not page["text"]:
            continue
        lines.extend(
            [
                f"<!-- page: {page['page']} -->",
                "",
                f"## Página {page['page']}",
                "",
                page["text"],
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_topic_index(pages: list[dict]) -> list[dict]:
    topics = []
    current_topic = None
    collecting_title = False
    pending_headings = []

    for page in pages:
        page_number = page["page"]
        if current_topic and page_number not in current_topic["pages"]:
            current_topic["pages"].append(page_number)

        for line in page["text"].splitlines():
            if _is_topic_heading(line):
                pending_headings.append(line.strip())
                pending_headings = pending_headings[-2:]
                continue

            question_match = QUESTION_PATTERN.match(line)
            if question_match:
                if current_topic:
                    topics.append(current_topic)
                current_topic = {
                    "id": question_match.group(1),
                    "section": " / ".join(pending_headings),
                    "title": question_match.group(2).strip(),
                    "pages": [page_number],
                }
                pending_headings = []
                collecting_title = "?" not in current_topic["title"]
                continue

            if current_topic and collecting_title:
                current_topic["title"] = f"{current_topic['title']} {line.strip()}".strip()
                collecting_title = "?" not in line

    if current_topic:
        topics.append(current_topic)

    unique_topics = {}
    for topic in topics:
        unique_topics[topic["id"]] = topic
    return list(unique_topics.values()) + [dict(topic) for topic in DERIVED_TOPICS]


def _is_topic_heading(line: str) -> bool:
    normalized = line.strip()
    letters = [character for character in normalized if character.isalpha()]
    return bool(letters) and normalized == normalized.upper() and len(normalized) >= 4


def save_topic_index(topics: list[dict], index_path: Path, source_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source_path),
        "topics": topics,
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def count_chars(pages: list[dict]) -> int:
    return sum(len(page["text"]) for page in pages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai texto de um PDF e salva em Markdown para indexação.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--min-chars", type=int, default=MIN_EXTRACTED_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = extract_pdf(args.input, args.min_chars)
    save_markdown(pages, args.output, args.input)
    topics = build_topic_index(pages)
    save_topic_index(topics, args.index_output, args.input)
    print(f"Contexto extraído com sucesso: {args.output}")
    print(f"Índice de temas gerado com sucesso: {args.index_output}")
    print(f"Páginas processadas: {len(pages)}")
    print(f"Temas identificados: {len(topics)}")
    print(f"Caracteres extraídos: {count_chars(pages)}")


if __name__ == "__main__":
    main()
