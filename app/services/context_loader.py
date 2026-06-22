from pathlib import Path
import re

from bs4 import BeautifulSoup
from markdown import markdown

PAGE_MARKER_PATTERN = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->\s*$", re.IGNORECASE)
SOURCE_MARKER_PATTERN = re.compile(r"^<!--\s*source:\s*(.*?)\s*-->\s*$", re.IGNORECASE)


class ContextLoaderError(Exception):
    pass


def load_context_sections(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de contexto não encontrado: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt"}:
        text = _normalize_text(path.read_text(encoding="utf-8"))
        return [{"text": text, "metadata": {"source": str(path)}}] if text else []
    if suffix in {".md", ".markdown"}:
        return _load_markdown_sections(path)

    raise ContextLoaderError("Formato de contexto não suportado para indexação. Use Markdown ou TXT extraído.")


def _load_markdown_sections(path: Path) -> list[dict]:
    source = str(path)
    current_page = None
    current_lines = []
    sections = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        source_match = SOURCE_MARKER_PATTERN.match(raw_line.strip())
        if source_match:
            source = source_match.group(1) or str(path)
            continue

        page_match = PAGE_MARKER_PATTERN.match(raw_line.strip())
        if page_match:
            _append_section(sections, current_lines, source, current_page)
            current_page = int(page_match.group(1))
            current_lines = []
            continue

        current_lines.append(raw_line)

    _append_section(sections, current_lines, source, current_page)

    if sections:
        return sections

    html = markdown(path.read_text(encoding="utf-8"))
    text = _normalize_text(BeautifulSoup(html, "html.parser").get_text("\n"))
    return [{"text": text, "metadata": {"source": str(path)}}] if text else []


def _append_section(sections: list[dict], lines: list[str], source: str, page: int | None) -> None:
    text = _markdown_to_text("\n".join(lines))
    if not text:
        return

    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    sections.append({"text": text, "metadata": metadata})


def _markdown_to_text(text: str) -> str:
    html = markdown(text)
    return _normalize_text(BeautifulSoup(html, "html.parser").get_text("\n"))


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    compact_lines = [line for line in lines if line]
    return "\n".join(compact_lines)
