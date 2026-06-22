import json
import tempfile
import unittest
from pathlib import Path

from app.schemas.chat import Source
from app.services.llm_service import LLMRateLimitError
from app.services.rag_service import RagService, _is_low_value_section, _merge_search_results, _rerank_sources


class FakeSettings:
    top_k = 4
    system_prompt_path: Path
    topic_index_path: Path

    def __init__(self, system_prompt_path: Path, topic_index_path: Path) -> None:
        self.system_prompt_path = system_prompt_path
        self.topic_index_path = topic_index_path

    def validate_runtime(self) -> None:
        return None


class FakeCollection:
    def __init__(self) -> None:
        self.where = None

    def count(self) -> int:
        return 1

    def get(self, where, include):
        self.where = where
        return {
            "documents": ["O contribuinte deve declarar bens e direitos, incluindo imóveis."],
            "metadatas": [{"source": "contexto-teste.md", "page": 12, "chunk": 0}],
        }

    def query(self, query_embeddings, n_results, where, include):
        self.where = where
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

    def classify_topics(self, question: str, topics: list[dict]) -> list[str]:
        return ["001"]

    def generate_answer(self, question: str, context: str, system_prompt: str) -> str:
        self.context_received = context
        self.system_prompt_received = system_prompt
        return "Resposta baseada no contexto recuperado."


class RateLimitedLLMService(FakeLLMService):
    def classify_topics(self, question: str, topics: list[dict]) -> list[str]:
        raise LLMRateLimitError("Limite temporário do provedor de IA atingido.")


class ContextFlowTest(unittest.TestCase):
    def test_answer_uses_recovered_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "system_prompt.md"
            prompt_path.write_text("Use apenas o contexto recuperado.", encoding="utf-8")
            topic_index_path = Path(temp_dir) / "context_index.json"
            topic_index_path.write_text(
                json.dumps(
                    {
                        "topics": [
                            {
                                "id": "001",
                                "section": "BENS E DIREITOS",
                                "title": "Como declarar imóvel?",
                                "pages": [12],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            service = RagService.__new__(RagService)
            service.settings = FakeSettings(prompt_path, topic_index_path)
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
            self.assertEqual(service.collection.where, {"page": {"$in": [12]}})

    def test_search_preserves_provider_rate_limit_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            topic_index_path = Path(temp_dir) / "context_index.json"
            topic_index_path.write_text(
                json.dumps({"topics": [{"id": "001", "title": "Tema", "pages": [12]}]}),
                encoding="utf-8",
            )
            service = RagService.__new__(RagService)
            service.settings = FakeSettings(Path(temp_dir) / "prompt.md", topic_index_path)
            service.collection = FakeCollection()
            service.llm_service = RateLimitedLLMService()

            with self.assertRaises(LLMRateLimitError):
                service.search("Pergunta")

    def test_rerank_prioritizes_declaration_requirement(self) -> None:
        sources = [
            Source(content="Como preencher e transmitir a declaração.", metadata={}, distance=0.1),
            Source(
                content="Está obrigada a apresentar a declaração a pessoa que recebeu rendimentos acima do limite.",
                metadata={},
                distance=0.4,
            ),
        ]

        reranked = _rerank_sources("Quem é obrigado a declarar imposto de renda?", sources)

        self.assertIn("obrigada", reranked[0].content)

    def test_rerank_prioritizes_income_tax_refund(self) -> None:
        sources = [
            Source(content="Informações gerais sobre o imposto de renda.", metadata={}, distance=0.1),
            Source(
                content="A restituição devolve ao contribuinte o imposto pago em valor maior que o devido.",
                metadata={},
                distance=0.4,
            ),
        ]

        reranked = _rerank_sources("O que é restituição?", sources)

        self.assertIn("restituição", reranked[0].content)

    def test_merge_includes_lexical_result_missing_from_vector_search(self) -> None:
        lexical_source = Source(
            content="Está obrigada a declarar a pessoa que recebeu rendimentos acima do limite.",
            metadata={},
        )
        vector_source = Source(
            content="Informações sobre imposto pago no exterior.",
            metadata={},
            distance=0.1,
        )

        results = _merge_search_results(
            "Quem é obrigado a declarar imposto de renda?",
            [lexical_source],
            [vector_source],
            result_limit=2,
        )

        self.assertEqual(results[0], lexical_source)

    def test_section_with_return_to_summary_is_indexable(self) -> None:
        section = "RESTITUIÇÃO\nO imposto pago a maior pode ser restituído.\nRetorno ao sumário"

        self.assertFalse(_is_low_value_section(section))

    def test_summary_section_is_ignored(self) -> None:
        section = "SUMÁRIO\nObrigatoriedade 23\nRestituição 48"

        self.assertTrue(_is_low_value_section(section))


if __name__ == "__main__":
    unittest.main()
