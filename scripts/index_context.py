from pathlib import Path

from app.core.config import get_settings
from app.services.rag_service import RagService

INDEX_CONTEXT_PATH = Path("./data/contexto.md")


def main() -> None:
    settings = get_settings()
    settings.context_file_path = INDEX_CONTEXT_PATH
    service = RagService(settings)
    total_chunks = service.index_context()
    print(f"Contexto indexado com sucesso: {total_chunks} chunks.")


if __name__ == "__main__":
    main()
