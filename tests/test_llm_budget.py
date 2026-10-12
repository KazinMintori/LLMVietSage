import unittest
from unittest.mock import patch

from config.settings import settings
from app.services.generation_budget_service import (
    context_budget_for,
    generation_options,
    output_budget_for,
    requests_extended_output,
)
from app.services.ollama_service import OllamaService
from app.domain.models.generation import ProviderChunk
from app.services.llm_service import LLMService


class _OptionsStreamingOllama:
    def __init__(self):
        self.options = None

    async def chat_stream(self, _messages, **options):
        self.options = OllamaService._chat_options(**options)
        yield ProviderChunk("Câu trả lời đầy đủ", done=True)


class OutputBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_extended_requests_get_more_room_only_when_conservatively_safe(self):
        messages = [{"role": "user", "content": "Lập lịch trình Yên Bái 5 ngày"}]
        with (
            patch.object(settings, "ollama_num_ctx", 4096),
            patch.object(settings, "ollama_num_predict", 1024),
            patch.object(settings, "ollama_long_num_predict", 1536),
            patch.object(settings, "ollama_context_reserve_tokens", 128),
        ):
            self.assertEqual(output_budget_for(messages, long_request=True), 1536)
            self.assertIsNone(output_budget_for(messages, long_request=False))
            too_large = [{"role": "user", "content": "x" * 12000}]
            self.assertIsNone(output_budget_for(too_large, long_request=True))

    def test_unbounded_default_is_not_replaced_with_a_positive_cap(self):
        with patch.object(settings, "ollama_num_predict", -1):
            self.assertIsNone(output_budget_for([], long_request=True))

    def test_unbounded_output_preserves_full_context_for_short_turns(self):
        messages = [{"role": "user", "content": "Hello"}]
        for budget in (-1, -2):
            with (
                self.subTest(budget=budget),
                patch.object(settings, "ollama_num_predict", budget),
                patch.object(settings, "ollama_num_ctx", 4096),
                patch.object(settings, "ollama_short_num_ctx", 2048),
            ):
                self.assertIsNone(context_budget_for(messages, long_request=False))
                self.assertEqual(
                    generation_options(messages, long_request=False),
                    {"num_predict": None, "num_ctx": None},
                )

    def test_combined_plan_estimates_input_once_and_preserves_both_budgets(self):
        messages = [{"role": "user", "content": "A caller-supplied reference"}]
        expectations = (
            (False, {"num_predict": None, "num_ctx": 2048}),
            (True, {"num_predict": 1536, "num_ctx": None}),
        )
        for long_request, expected in expectations:
            with (
                self.subTest(long_request=long_request),
                patch.object(settings, "ollama_num_ctx", 4096),
                patch.object(settings, "ollama_short_num_ctx", 2048),
                patch.object(settings, "ollama_num_predict", 1024),
                patch.object(settings, "ollama_long_num_predict", 1536),
                patch.object(settings, "ollama_context_reserve_tokens", 128),
                patch("app.services.generation_budget_service.estimate_message_tokens", return_value=100) as estimate,
            ):
                self.assertEqual(generation_options(messages, long_request=long_request), expected)
                estimate.assert_called_once_with(messages)

    def test_vietnamese_context_estimate_allows_extended_budget(self):
        messages = [{"role": "system", "content": "Điểm đến có khí hậu mát, nhiều cung đường đi bộ và chợ địa phương."}]
        with (
            patch.object(settings, "ollama_num_ctx", 4096),
            patch.object(settings, "ollama_num_predict", 1024),
            patch.object(settings, "ollama_long_num_predict", 1536),
            patch.object(settings, "ollama_context_reserve_tokens", 128),
        ):
            self.assertEqual(output_budget_for(messages, long_request=True), 1536)

    def test_short_turn_uses_reduced_context_when_response_fits(self):
        messages = [{"role": "user", "content": "Giờ mở cửa là mấy giờ?"}]
        with (
            patch.object(settings, "ollama_num_ctx", 4096),
            patch.object(settings, "ollama_short_num_ctx", 2048),
            patch.object(settings, "ollama_num_predict", 1024),
            patch.object(settings, "ollama_context_reserve_tokens", 128),
        ):
            self.assertEqual(context_budget_for(messages, long_request=False), 2048)
            self.assertIsNone(context_budget_for(messages, long_request=True))

    def test_only_explicit_long_form_requests_raise_the_cap(self):
        self.assertTrue(requests_extended_output("Lập lịch trình Yên Bái 5 ngày 4 đêm"))
        self.assertTrue(requests_extended_output("Compare Tram Tau and Suoi Giang in detail"))
        self.assertTrue(requests_extended_output("请给我逐日行程"))
        self.assertFalse(requests_extended_output("Hồ Thác Bà có gì đáng xem?"))

    async def test_client_passes_an_explicit_extended_budget_to_provider(self):
        provider = _OptionsStreamingOllama()
        client = LLMService(provider)

        answer = "".join([
            part async for part in client.chat_stream(
                [{"role": "user", "content": "Lịch trình 5 ngày"}],
                num_predict=1536,
            )
        ])

        self.assertEqual(answer, "Câu trả lời đầy đủ")
        self.assertEqual(provider.options["num_predict"], 1536)


if __name__ == "__main__":
    unittest.main()
