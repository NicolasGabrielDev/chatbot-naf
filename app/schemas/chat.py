from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("A pergunta não pode estar vazia.")
        return question


class Source(BaseModel):
    content: str
    metadata: dict
    distance: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str


class HealthResponse(BaseModel):
    status: str
