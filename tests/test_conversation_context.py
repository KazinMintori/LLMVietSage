"""Caller-supplied history stays intact and request-scoped during inference."""
import asyncio
from copy import deepcopy
import unittest

from app.domain.models.generation import ProviderReply
from app.domain.schema.inference_schema import ChatRequest
from app.services.inference_service import InferenceService, prepare_chat
from app.services.llm_service import LLMService
from config.prompts import CHAT_SYSTEM_PROMPT


class ConversationContextTests(unittest.TestCase):
    def test_history_preserves_roles_order_and_exact_content(self):
        turns = [
            {"role": "user", "content": "  Tôi tên Linh.\n"},
            {"role": "assistant", "content": "Chào Linh!"},
            {"role": "user", "content": "Correction: call me Minh.  "},
            {"role": "user", "content": "What name did I just choose?"},
        ]
        request = ChatRequest(messages=turns)
        original = deepcopy(request.model_dump())
        messages, _ = prepare_chat(request)

        self.assertEqual(messages[0], {"role": "system", "content": CHAT_SYSTEM_PROMPT})
        self.assertEqual(messages[1:], turns)
        self.assertEqual(request.model_dump(), original)

    def test_prepared_mutation_does_not_change_caller_input_or_next_preparation(self):
        request = ChatRequest(messages=[{"role": "user", "content": "Keep this exact text."}])
        first, first_options = prepare_chat(request)
        first[-1]["content"] = "mutated"
        first.append({"role": "assistant", "content": "injected"})
        first_options["num_ctx"] = -1

        second, second_options = prepare_chat(request)
        self.assertEqual(second[-1], {"role": "user", "content": "Keep this exact text."})
        self.assertEqual(len(second), 2)
        self.assertGreater(second_options["num_ctx"], 0)
        self.assertEqual(request.messages[0].content, "Keep this exact text.")

    def test_single_message_never_reuses_another_requests_history(self):
        history, _ = prepare_chat(ChatRequest(messages=[
            {"role": "user", "content": "Private request A reference."},
            {"role": "assistant", "content": "A reply."},
            {"role": "user", "content": "Follow up A."},
        ]))
        fresh, _ = prepare_chat(ChatRequest(message="Unrelated request B."))
        self.assertEqual(len(history), 4)
        self.assertEqual(fresh, [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "Unrelated request B."},
        ])

    def test_long_reference_is_not_truncated_summarized_or_deduplicated(self):
        reference = "Thông tin tham chiếu: 🧭\n" + ("Dòng lặp có chủ ý.\n" * 400)
        request = ChatRequest(message=reference)
        messages, _ = prepare_chat(request)
        self.assertGreater(len(reference), 4000)
        self.assertEqual(messages[-1]["content"], reference)

    def test_all_history_is_forwarded_without_legacy_turn_limit(self):
        turns = [{"role": "user", "content": f"Fact {index}."} for index in range(100)]
        messages, _ = prepare_chat(ChatRequest(messages=turns))
        self.assertEqual(messages[1:], turns)
        self.assertEqual(messages[1]["content"], "Fact 0.")

    def test_travel_correction_remains_caller_text_without_retrieval_scope(self):
        turns = [
            {"role": "user", "content": "Đi Huế 3 ngày, ngân sách 5 triệu."},
            {"role": "assistant", "content": "Bạn đã chọn Huế."},
            {"role": "user", "content": "Đính chính: đổi thành Hội An."},
        ]
        messages, _ = prepare_chat(ChatRequest(messages=turns))
        self.assertEqual(messages[1:], turns)
        self.assertTrue(all(set(turn) == {"role", "content"} for turn in messages))


class ConversationIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_only_receive_their_own_history(self):
        class FakeProvider:
            provider_name = "fake"

            async def chat(self, messages, **_options):
                await asyncio.sleep(0)
                return ProviderReply(" | ".join(turn["content"] for turn in messages[1:]))

        service = InferenceService(LLMService(FakeProvider()))
        first = ChatRequest(messages=[
            {"role": "user", "content": "A private fact"},
            {"role": "assistant", "content": "A previous reply"},
            {"role": "user", "content": "A followup"},
        ])
        second = ChatRequest(message="B standalone question")
        results = await asyncio.gather(service.chat(first), service.chat(second))
        self.assertEqual(results[0].reply, "A private fact | A previous reply | A followup")
        self.assertEqual(results[1].reply, "B standalone question")


if __name__ == "__main__":
    unittest.main()
