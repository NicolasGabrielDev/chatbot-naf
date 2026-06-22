import unittest

from app.services.llm_service import LLMServiceError, _is_rate_limit_error, _parse_topic_ids
from app.services.topic_catalog import pages_for_topics, simplify_question
from scripts.extract_context import build_topic_index


class TopicCatalogTest(unittest.TestCase):
    def test_build_topic_index_maps_question_to_pages(self) -> None:
        pages = [
            {
                "page": 23,
                "text": (
                    "OBRIGATORIEDADE DE APRESENTAÇÃO\n"
                    "001 — Quem está obrigado a declarar?\n"
                    "Resposta iniciada."
                ),
            },
            {
                "page": 24,
                "text": (
                    "Continuação da resposta.\n"
                    "PESSOA FÍSICA DESOBRIGADA\n"
                    "002 — Pessoa desobrigada pode declarar?\n"
                    "Sim."
                ),
            },
        ]

        topics = build_topic_index(pages)

        self.assertEqual(topics[0]["id"], "001")
        self.assertEqual(topics[0]["section"], "OBRIGATORIEDADE DE APRESENTAÇÃO")
        self.assertEqual(topics[0]["pages"], [23, 24])
        self.assertEqual(topics[1]["pages"], [24])
        self.assertEqual(topics[-1]["id"], "conceito_restituicao")
        self.assertEqual(topics[-1]["pages"], [30, 48])

    def test_simplify_question_removes_excess_words(self) -> None:
        self.assertEqual(
            simplify_question("O que é a restituição do Imposto de Renda?"),
            "restituicao imposto renda",
        )

    def test_pages_for_topics_combines_selected_pages(self) -> None:
        topics = [
            {"id": "001", "pages": [23, 24]},
            {"id": "069", "pages": [48]},
        ]

        self.assertEqual(pages_for_topics(topics, ["069", "001"]), [23, 24, 48])

    def test_parse_topic_ids_rejects_invalid_response(self) -> None:
        with self.assertRaises(LLMServiceError):
            _parse_topic_ids("sem json")

    def test_identifies_provider_rate_limit(self) -> None:
        self.assertTrue(_is_rate_limit_error(Exception("429 quota exceeded")))


if __name__ == "__main__":
    unittest.main()
