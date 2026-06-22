import argparse
from pathlib import Path

MIN_EXTRACTED_CHARS = 1000
DEFAULT_INPUT_PATH = Path("./data/context.pdf")
DEFAULT_OUTPUT_PATH = Path("./data/contexto.md")


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


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def count_chars(pages: list[dict]) -> int:
    return sum(len(page["text"]) for page in pages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai texto de um PDF e salva em Markdown para indexação.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--min-chars", type=int, default=MIN_EXTRACTED_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = extract_pdf(args.input, args.min_chars)
    save_markdown(pages, args.output, args.input)
    print(f"Contexto extraído com sucesso: {args.output}")
    print(f"Páginas processadas: {len(pages)}")
    print(f"Caracteres extraídos: {count_chars(pages)}")


if __name__ == "__main__":
    main()
