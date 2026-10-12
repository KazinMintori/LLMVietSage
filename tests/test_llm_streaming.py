from io import BytesIO
import unittest
from unittest.mock import patch

from app.domain.models.generation import ProviderChunk
from app.services.llm_service import LLMService
from app.services.response_sanitization_service import parse_structured_reply, strip_thinking, strip_thinking_stream
from app.domain.errors import InvalidModelResponseError
from scripts.measure_model_latency import ask


async def _chunks(parts):
    for part in parts:
        yield part


class _StreamingOllama:
    async def chat_stream(self, messages, **options):
        yield ProviderChunk("Mảnh thứ nhất. ")
        yield ProviderChunk("Mảnh thứ hai.", done=True)


class StreamingFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_visible_content_is_yielded_incrementally(self):
        received = []
        async for part in strip_thinking_stream(_chunks(["Xin ", "chào", " bạn"])):
            received.append(part)
        self.assertGreater(len(received), 1)
        self.assertEqual("".join(received), "Xin chào bạn")

    async def test_split_thinking_tags_are_removed(self):
        received = []
        parts = ["<thi", "nk>bí mật", "</thi", "nk>Trả ", "lời"]
        async for part in strip_thinking_stream(_chunks(parts)):
            received.append(part)
        self.assertEqual("".join(received), "Trả lời")

    async def test_unclosed_thinking_is_not_exposed(self):
        received = []
        async for part in strip_thinking_stream(_chunks(["<think>", "bí mật"])):
            received.append(part)
        self.assertEqual(received, [])

    async def test_truncated_tag_prefix_is_not_emitted_at_end_of_stream(self):
        received = [part async for part in strip_thinking_stream(_chunks(["Public answer", "<thi"]))]
        self.assertEqual("".join(received), "Public answer")

    async def test_llm_facade_does_not_buffer_complete_answer(self):
        client = LLMService(_StreamingOllama())

        received = []
        async for part in client.chat_stream(
            [{"role": "user", "content": "Xin chào"}]
        ):
            received.append(part)

        self.assertEqual(received, ["Mảnh thứ nhất. ", "Mảnh thứ hai."])

    def test_complete_reply_drops_reasoning_before_stray_closing_tag(self):
        reply = "Internal reasoning that must not leak.</think>Final answer"
        self.assertEqual(strip_thinking(reply), "Final answer")

    def test_structured_reply_rejects_nonfinite_json_numbers(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(InvalidModelResponseError):
                parse_structured_reply('{"value":' + value + '}')


class LatencyStreamContractTests(unittest.TestCase):
    def test_success_requires_done_and_times_first_token_incrementally(self):
        response = BytesIO(
            b'event: token\ndata: {"text":"Hello "}\n\n'
            b'event: token\ndata: {"text":"world"}\n\n'
            b'event: done\ndata: {"model":"fake","usage":{}}\n\n'
        )
        with (
            patch("scripts.measure_model_latency.urllib.request.urlopen", return_value=response) as open_stream,
            patch("scripts.measure_model_latency.time.perf_counter", side_effect=[10.0, 10.25, 11.5]),
        ):
            self.assertEqual(ask("http://fake-inference", "Public question", 12), (0.25, 1.5, 2))
        self.assertEqual(open_stream.call_args.args[0].full_url, "http://fake-inference/api/chat/stream")
        self.assertEqual(open_stream.call_args.kwargs, {"timeout": 12})
        self.assertTrue(response.closed)

    def test_error_before_or_after_tokens_is_never_success_or_private_detail(self):
        for prefix in (b"", b'event: token\ndata: {"text":"Partial answer"}\n\n'):
            response = BytesIO(prefix + b'event: error\ndata: {"message":"private-provider-url and private context"}\n\n')
            with self.subTest(prefix=prefix), patch(
                "scripts.measure_model_latency.urllib.request.urlopen", return_value=response,
            ):
                with self.assertRaises(RuntimeError) as failure:
                    ask("http://fake-inference", "Public question", 12)
            self.assertEqual(str(failure.exception), "Inference stream failed.")
            self.assertNotIn("private", str(failure.exception))
            self.assertTrue(response.closed)

    def test_token_stream_without_done_is_incomplete(self):
        response = BytesIO(b'event: token\ndata: {"text":"Partial answer"}\n\n')
        with patch("scripts.measure_model_latency.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "before completion"):
                ask("http://fake-inference", "Public question", 12)
        self.assertTrue(response.closed)

    def test_done_without_tokens_is_not_an_answer(self):
        response = BytesIO(b'event: done\ndata: {"model":"fake","usage":{}}\n\n')
        with patch("scripts.measure_model_latency.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "without visible tokens"):
                ask("http://fake-inference", "Public question", 12)

    def test_invalid_token_payload_does_not_echo_the_payload(self):
        response = BytesIO(b'event: token\ndata: private-provider-url\n\n')
        with patch("scripts.measure_model_latency.urllib.request.urlopen", return_value=response):
            with self.assertRaises(RuntimeError) as failure:
                ask("http://fake-inference", "Public question", 12)
        self.assertEqual(str(failure.exception), "Inference returned an invalid stream.")
        self.assertIsNone(failure.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
