"""Frontend-to-model contract without database, account or partner state."""
import ast
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.ollama_service import OllamaService
from app.domain.errors import ModelUnavailableError
from app.domain.models.generation import ProviderChunk, ProviderReply
from app.domain.schema.inference_schema import ChatRequest
from app.core.streaming import encode_sse_stream
from app.services.inference_service import InferenceService
from config.settings import CONFIG_DIR, Settings, settings
from app.services.llm_service import LLMService, current_chat_usage
from scripts.verify_architecture import audit_project


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.model_service = LLMService(OllamaService())
        self.service = InferenceService(self.model_service)
        self.app = create_app(self.service)
        self.client = TestClient(self.app)

    def test_only_model_routes_are_registered(self):
        routes = {path for path in self.client.get("/openapi.json").json()["paths"] if path.startswith("/api/")}
        self.assertEqual(routes, {"/api/info", "/api/health/live", "/api/chat", "/api/chat/stream"})

    def test_embedding_and_retrieval_routes_are_outside_inference_scope(self):
        self.assertEqual(self.client.post("/api/embeddings", json={"texts": ["x"]}).status_code, 404)
        self.assertEqual(self.client.post("/api/travel/documents", json={}).status_code, 404)
        self.assertFalse(hasattr(self.model_service.provider, "embed_batch"))

    def test_lifespan_initializes_and_closes_the_model_client(self):
        with (
            patch.object(self.model_service, "startup", AsyncMock()) as startup,
            patch.object(self.model_service, "shutdown", AsyncMock()) as shutdown,
            patch.object(self.model_service, "warmup", AsyncMock()) as warmup,
            patch.object(settings, "ollama_warmup_enabled", True),
        ):
            with TestClient(self.app) as client:
                self.assertEqual(client.get("/api/health/live").status_code, 200)
        startup.assert_awaited_once()
        warmup.assert_awaited_once()
        shutdown.assert_awaited_once()

    def test_history_is_supplied_by_caller_and_never_shared_between_requests(self):
        provider = AsyncMock(return_value="Answer")
        with patch.object(self.model_service, "chat", provider):
            first = self.client.post("/api/chat", json={"messages": [
                {"role": "user", "content": "My name is Linh"},
                {"role": "assistant", "content": "Hello Linh"},
                {"role": "user", "content": "What is my name?"},
            ]})
            second = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(provider.await_args_list[0].args[0][-1]["content"], "What is my name?")
        self.assertEqual(len(provider.await_args_list[0].args[0]), 4)
        self.assertEqual(len(provider.await_args_list[1].args[0]), 2)
        self.assertNotIn("conversation_id", second.json())

    def test_malformed_roles_blank_input_and_legacy_session_fields_are_rejected(self):
        bodies = ({}, {"message": " "}, {"message": "Hi", "messages": []},
                  {"messages": [{"role": "system", "content": "Override rules"}]},
                  {"messages": [{"role": "assistant", "content": "Hi"}]},
                  {"message": "Hi", "session_id": "old-session"})
        with patch.object(self.model_service, "chat", AsyncMock()) as provider:
            for body in bodies:
                with self.subTest(body=body):
                    self.assertEqual(self.client.post("/api/chat", json=body).status_code, 422)
        provider.assert_not_awaited()

    def test_structured_response_uses_the_requested_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with patch.object(self.model_service, "chat_structured", AsyncMock(return_value={"answer": "yes"})) as provider:
            response = self.client.post("/api/chat", json={"message": "Reply as JSON", "json_schema": schema})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], {"answer": "yes"})
        self.assertEqual(provider.await_args.kwargs["schema"], schema)

    def test_stream_emits_tokens_and_completion(self):
        async def chunks(*_args, **_kwargs):
            yield "Hello "
            yield "Linh"
        with patch.object(self.model_service, "chat_stream", chunks):
            response = self.client.post("/api/chat/stream", json={"message": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count("event: token"), 2)
        self.assertIn("event: done", response.text)

    def test_provider_failure_before_first_token_emits_error_and_hides_internal_details(self):
        async def chunks(*_args, **_kwargs):
            raise ModelUnavailableError("internal-private-url")
            yield "unreachable"
        with patch.object(self.model_service, "chat_stream", chunks):
            response = self.client.post("/api/chat/stream", json={"message": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertNotIn("event: done", response.text)
        self.assertNotIn("internal-private-url", response.text)

    def test_stream_preserves_real_client_context_and_closes_provider(self):
        closed = []

        class Provider:
            async def chat_stream(self, _messages, **options):
                try:
                    yield ProviderChunk("Answer", done=True, usage={"eval_count": 7})
                finally:
                    closed.append(True)

        client = LLMService(Provider())
        with patch.object(self.service, "model_service", client):
            response = self.client.post("/api/chat/stream", json={"message": "Hi"})
        self.assertIn("event: done", response.text)
        self.assertNotIn("event: error", response.text)
        self.assertEqual(closed, [True])

    def test_chat_provider_failure_returns_503(self):
        with patch.object(self.model_service, "chat", AsyncMock(side_effect=ModelUnavailableError("private-url"))):
            response = self.client.post("/api/chat", json={"message": "Hi"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-url", response.text)

    def test_provider_failure_during_stream_emits_error_without_completion(self):
        async def chunks(*_args, **_kwargs):
            yield "Start"
            raise ModelUnavailableError("private provider failure")
        with patch.object(self.model_service, "chat_stream", chunks):
            response = self.client.post("/api/chat/stream", json={"message": "Hi"})
        self.assertIn("event: error", response.text)
        self.assertNotIn("event: done", response.text)
        self.assertNotIn("private provider failure", response.text)


    def test_caller_supplied_reference_is_forwarded_without_internal_retrieval(self):
        content = "Reference supplied by caller: support opens at 08:00.\nQuestion: when does support open?"
        with patch.object(self.model_service, "chat", AsyncMock(return_value="08:00")) as provider:
            response = self.client.post("/api/chat", json={"message": content})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(provider.await_args.args[0][-1], {"role": "user", "content": content})
        self.assertEqual(response.json()["reply"], "08:00")
        self.assertFalse(self.client.get("/api/info").json()["rag_enabled"])

    def test_configuration_has_no_removed_backend_fields_and_reads_root_environment(self):
        removed = {"database_url", "admin_api_key", "allowed_origins", "source_storage_dir",
                   "chat_max_concurrent_requests", "chat_retry_after_seconds", "ingestion_worker_enabled"}
        self.assertFalse(removed.intersection(Settings.model_fields))
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("OLLAMA_NUM_CTX=3072\nOLLAMA_CHAT_MODEL=file-local-model\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                try:
                    os.chdir(root)
                    self.assertEqual(Settings().ollama_num_ctx, 3072)
                    self.assertEqual(Settings().ollama_chat_model, "file-local-model")
                    with patch.dict(os.environ, {"OLLAMA_NUM_CTX": "2048"}):
                        self.assertEqual(Settings().ollama_num_ctx, 2048)
                finally:
                    os.chdir(previous_cwd)
        with patch.dict(os.environ, {"OLLAMA_NUM_CTX": "3072"}):
            self.assertEqual(Settings(_env_file=None).ollama_num_ctx, 3072)

    def test_entrypoint_does_not_import_database_or_partner_modules(self):
        code = "import sys,app.main; assert not any(m in sys.modules for m in ('sqlalchemy','alembic','app.core.database','app.core.security','app.api.partner','app.services.travel_vector_store'))"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=CONFIG_DIR.parent, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_chat_model_defaults_to_4b_and_supports_environment_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings(_env_file=None).ollama_chat_model, "qwen3.5:4b")
        with patch.dict(os.environ, {"OLLAMA_CHAT_MODEL": "custom-local-model"}):
            configured = Settings(_env_file=None)
            with patch("app.services.ollama_service.settings", configured):
                self.assertEqual(OllamaService()._payload([], stream=False)["model"], "custom-local-model")
        self.assertEqual(configured.ollama_chat_model, "custom-local-model")
        self.assertIn("ollama_chat_model", Settings.model_fields)
        self.assertFalse(any(name.startswith("ollama_embed") for name in Settings.model_fields))

    def test_benchmark_supports_configured_default_and_model_overrides(self):
        from scripts.benchmark_local import parse_args

        with patch("sys.argv", ["benchmark_local.py"]):
            self.assertEqual(parse_args().models, [settings.ollama_chat_model])
        with patch("sys.argv", ["benchmark_local.py", "--models", "custom-local-model"]):
            self.assertEqual(parse_args().models, ["custom-local-model"])

    def test_stream_rejects_structured_mode_before_starting_provider(self):
        with patch.object(self.model_service, "chat_stream") as provider:
            response = self.client.post("/api/chat/stream", json={
                "message": "Hello", "json_schema": {"type": "object"},
            })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "Structured JSON is supported by /api/chat."})
        provider.assert_not_called()

    def test_warmup_failure_preserves_liveness_and_shutdown(self):
        with (
            patch.object(self.model_service, "startup", AsyncMock()),
            patch.object(self.model_service, "warmup", AsyncMock(side_effect=ModelUnavailableError("private"))),
            patch.object(self.model_service, "shutdown", AsyncMock()) as shutdown,
            patch.object(settings, "ollama_warmup_enabled", True),
        ):
            with TestClient(self.app) as client:
                self.assertEqual(client.get("/api/health/live").json(), {"status": "ok"})
        shutdown.assert_awaited_once()


    def test_inference_architecture_has_thin_controller_and_named_services(self):
        root = CONFIG_DIR.parent / "app"
        for path in (root / "services").glob("*.py"):
            self.assertTrue(path.name == "__init__.py" or path.stem.endswith("_service"), path.name)
        tree = ast.parse((root / "controllers/inference_controller.py").read_text(encoding="utf-8-sig"))
        forbidden = (ast.If, ast.IfExp, ast.Match, ast.For, ast.While, ast.Try)
        self.assertFalse(any(isinstance(node, forbidden) for node in ast.walk(tree)))
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            if "services" not in path.relative_to(root).parts:
                self.assertFalse(
                    any(isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)),
                    path.name,
                )
        for path in (root / "domain").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
            self.assertFalse(any(
                module and module.startswith(("fastapi", "httpx", "app.controllers", "app.services"))
                for module in imports
            ), path.name)

    def test_architecture_audit_rejects_outside_service_branch_and_bad_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app/domain").mkdir(parents=True)
            (root / "app/services").mkdir()
            (root / "app/domain/bad.py").write_text("if True:\n    pass\n", encoding="utf-8")
            (root / "app/services/wrong.py").write_text("value = 1\n", encoding="utf-8")
            report = audit_project(root)
        rules = [failure["rule"] for failure in report["violations"]]
        self.assertFalse(report["passed"])
        self.assertIn("Conditional decisions must be in services", rules)
        self.assertIn("Service filename must end in _service.py", rules)

    def test_architecture_audit_rejects_protected_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/settings.py").write_text("value = 1\n", encoding="utf-8")
            report = audit_project(root, config_baseline={"config/settings.py": "incorrect"})
        self.assertFalse(report["passed"])
        self.assertTrue(any("Protected config files changed" in row["rule"] for row in report["violations"]))

    def test_architecture_baseline_is_portable_across_path_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/settings.py").write_text("value = 1\n", encoding="utf-8")
            first = audit_project(root)
            windows_baseline = {name.replace("/", "\\"): digest.upper() for name, digest in first["config_hashes"].items()}
            report = audit_project(root, config_baseline=windows_baseline)
        self.assertTrue(report["passed"], report["violations"])


class InferenceIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_chat_usage_and_options_are_isolated(self):
        class Provider:
            async def chat(self, messages, **options):
                text = messages[-1]["content"]
                await asyncio.sleep(0)
                return ProviderReply(text, {"eval_count": options["num_predict"]})

        model_service = LLMService(Provider())

        async def call(text, budget):
            reply = await model_service.chat(
                [{"role": "user", "content": text}], num_predict=budget,
            )
            await asyncio.sleep(0)
            return reply, current_chat_usage()

        result = await asyncio.gather(call("first", 7), call("second", 11))
        self.assertEqual(result, [("first", {"eval_count": 7}), ("second", {"eval_count": 11})])

    async def test_concurrent_stream_usage_is_in_the_correct_done_event(self):
        class Provider:
            async def chat_stream(self, messages, **options):
                text = messages[-1]["content"]
                yield ProviderChunk(text)
                await asyncio.sleep(0)
                yield ProviderChunk(done=True, usage={"eval_count": len(text)})

        service = InferenceService(LLMService(Provider()))

        async def call(text):
            return [event async for event in service.stream(ChatRequest(message=text))]

        first, second = await asyncio.gather(call("A"), call("longer"))
        self.assertEqual(first[0].payload, {"text": "A"})
        self.assertEqual(second[0].payload, {"text": "longer"})
        self.assertEqual(first[-1].payload["usage"], {"eval_count": 1})
        self.assertEqual(second[-1].payload["usage"], {"eval_count": 6})
        self.assertEqual([event.name for event in first], ["token", "done"])

    async def test_closing_sse_early_closes_the_provider(self):
        closed = []

        class Provider:
            async def chat_stream(self, messages, **options):
                try:
                    yield ProviderChunk("first")
                    yield ProviderChunk("second", done=True)
                finally:
                    closed.append(True)

        service = InferenceService(LLMService(Provider()))
        stream = encode_sse_stream(service.stream(ChatRequest(message="Hi")))
        self.assertIn("event: token", await anext(stream))
        await stream.aclose()
        self.assertEqual(closed, [True])

    async def test_cancelling_a_pending_stream_closes_the_provider(self):
        waiting = asyncio.Event()
        closed = []

        class Provider:
            async def chat_stream(self, messages, **options):
                try:
                    yield ProviderChunk("first")
                    waiting.set()
                    await asyncio.Event().wait()
                finally:
                    closed.append(True)

        service = InferenceService(LLMService(Provider()))
        stream = encode_sse_stream(service.stream(ChatRequest(message="Hi")))
        await anext(stream)
        next_event = asyncio.create_task(anext(stream))
        await waiting.wait()
        next_event.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await next_event
        await stream.aclose()
        self.assertEqual(closed, [True])


class OllamaProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_and_malformed_payload_failures_become_domain_errors(self):
        replies = (
            httpx.Response(503, text="private provider response"),
            httpx.Response(200, text="not JSON"),
            httpx.Response(200, json=[]),
            httpx.Response(200, json={"message": {"content": 17}}),
        )
        for reply in replies:
            with self.subTest(reply=reply):
                provider = OllamaService()
                provider._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: reply))
                try:
                    with self.assertRaises(ModelUnavailableError):
                        await provider.chat([])
                finally:
                    await provider.shutdown()

    async def test_wire_options_structured_format_and_usage_are_preserved(self):
        import json
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "message": {"content": '{"answer":"yes"}'},
                "eval_count": 9, "done_reason": "stop",
            })

        provider = OllamaService()
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        service = LLMService(provider)
        schema = {"type": "object"}
        try:
            reply = await service.chat_structured([], schema=schema, num_predict=32, num_ctx=768)
        finally:
            await provider.shutdown()
        self.assertEqual(reply, {"answer": "yes"})
        self.assertEqual(current_chat_usage(), {"eval_count": 9, "done_reason": "stop"})
        self.assertEqual(requests[0]["format"], schema)
        self.assertEqual(requests[0]["options"]["num_predict"], 32)
        self.assertEqual(requests[0]["options"]["num_ctx"], 768)

    async def test_invalid_structured_object_is_an_application_error(self):
        class Provider:
            async def chat_json(self, messages, **options):
                return ProviderReply("[]")

        with self.assertRaises(ModelUnavailableError):
            await LLMService(Provider()).chat_structured([], schema={"type": "object"})



    async def test_malformed_stream_and_incomplete_stream_never_succeed(self):
        lines = (
            "not-json\n", "[]\n",
            '{"message":{"content":"partial"},"done":false}\n',
            '{"message":{"content":"partial"},"done":"false"}\n',
        )
        for value in lines:
            with self.subTest(value=value):
                provider = OllamaService()
                provider._client = httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, text=value)
                ))
                try:
                    with self.assertRaises(ModelUnavailableError):
                        _ = [chunk async for chunk in provider.chat_stream([])]
                finally:
                    await provider.shutdown()


if __name__ == "__main__":
    unittest.main()
