import unittest
from unittest.mock import patch

from app.services.llm_service import LLMService
from app.domain.errors import ModelUnavailableError
from app.domain.models.generation import ProviderReply


class _FakeOllama:
    def __init__(self):
        self.calls = []

    async def chat(self, messages):
        self.calls.append(("chat", messages))
        return ProviderReply("OK")


class LLMWarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_calls_only_generation_provider(self):
        provider = _FakeOllama()
        client = LLMService(provider)

        await client.warmup()
        self.assertEqual([call[0] for call in provider.calls], ["chat"])

    async def test_warmup_failure_is_left_to_application_lifecycle_policy(self):
        provider = _FakeOllama()
        client = LLMService(provider)
        with patch.object(provider, "chat", side_effect=ModelUnavailableError("private")):
            with self.assertRaises(ModelUnavailableError):
                await client.warmup()


if __name__ == "__main__":
    unittest.main()
