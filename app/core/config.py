from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    context_file_path: Path = Field(default=Path("./data/contexto.md"), alias="CONTEXT_FILE_PATH")
    topic_index_path: Path = Field(default=Path("./data/context_index.json"), alias="TOPIC_INDEX_PATH")
    system_prompt_path: Path = Field(default=Path("./app/prompts/system_prompt.md"), alias="SYSTEM_PROMPT_PATH")
    vectorstore_path: Path = Field(default=Path("./app/vectorstore"), alias="VECTORSTORE_PATH")
    chroma_collection_name: str = Field(default="ir_context", alias="CHROMA_COLLECTION_NAME")

    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP", ge=0)
    top_k: int = Field(default=4, alias="TOP_K", ge=1, le=20)
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_batch_size: int = Field(default=50, alias="EMBEDDING_BATCH_SIZE", ge=1, le=100)
    local_embedding_dimensions: int = Field(default=768, alias="LOCAL_EMBEDDING_DIMENSIONS", ge=128, le=4096)

    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    gemini_embedding_model: str = Field(default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        provider = value.lower().strip()
        if provider not in {"openai", "gemini"}:
            raise ValueError("LLM_PROVIDER deve ser 'openai' ou 'gemini'.")
        return provider

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        provider = value.lower().strip()
        if provider not in {"llm", "local"}:
            raise ValueError("EMBEDDING_PROVIDER deve ser 'llm' ou 'local'.")
        return provider

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, values) -> int:
        chunk_size = values.data.get("chunk_size")
        if chunk_size and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE.")
        return value

    def validate_runtime(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY não foi configurada.")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY não foi configurada.")
        if not self.context_file_path.exists():
            raise FileNotFoundError(f"Arquivo de contexto não encontrado: {self.context_file_path}")
        if not self.topic_index_path.exists():
            raise FileNotFoundError(f"Índice de temas não encontrado: {self.topic_index_path}")
        if not self.system_prompt_path.exists():
            raise FileNotFoundError(f"Prompt de sistema não encontrado: {self.system_prompt_path}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
