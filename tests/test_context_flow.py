import tempfile
import unittest
from pathlib import Path

from app.services.rag_service import RagService


class FakeSettings:
    top_k = 4
    system_prompt_path: Path

    def __init__(self, system_prompt_path: Path) -> None:
        self.system_prompt_path = system_prompt_path

    def validate_runtime(self) -> None:
        return None


class FakeCollection:
    def count(self) -> int:
        return 1

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [["O contribuinte deve declarar bens e direitos, incluindo imóveis."]],
            "metadatas": [[{"source": "contexto-teste.md", "page": 12, "chunk": 0}]],
            "distances": [[0.12]],
        }


class FakeLLMService:
    model_used = "modelo-teste"

    def __init__(self) -> None:
        self.context_received = ""
        self.system_prompt_received = ""

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def generate_answer(self, question: str, context: str, system_prompt: str) -> str:
        self.context_received = context
        self.system_prompt_received = system_prompt
        return "Resposta baseada no contexto recuperado."


class ContextFlowTest(unittest.TestCase):
    def test_answer_uses_recovered_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "system_prompt.md"
            prompt_path.write_text("Use apenas o contexto recuperado.", encoding="utf-8")

            service = RagService.__new__(RagService)
            service.settings = FakeSettings(prompt_path)
            service.collection = FakeCollection()
            service.llm_service = FakeLLMService()

            answer, sources, model_used = service.answer("Como declarar imóvel?")

            self.assertEqual(answer, "Resposta baseada no contexto recuperado.")
            self.assertEqual(model_used, "modelo-teste")
            self.assertEqual(len(sources), 1)
            self.assertIn("imóveis", sources[0].content)
            self.assertEqual(sources[0].metadata["page"], 12)
            self.assertIn("imóveis", service.llm_service.context_received)
            self.assertEqual(service.llm_service.system_prompt_received, "Use apenas o contexto recuperado.")


if __name__ == "__main__":
    unittest.main()
