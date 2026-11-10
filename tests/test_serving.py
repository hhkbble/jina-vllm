import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
import torch
from fastapi import FastAPI
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.scoring.api_router import router
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest, ScoreTextRequest
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers import cached_tokenizer_from_config

from jina_vllm.contract import DOC_TOKEN_ID
from jina_vllm.plugin import register
from jina_vllm.serving import Jina35ServingScores, install_serving


class FakeEngine:
    def __init__(self, config, renderer):
        self.vllm_config = config
        self.model_config = config.model_config
        self.renderer = renderer
        self.input_processor = None
        self.started = []
        self.closed = []
        self.aborted = []
        self.prompts = []
        self.fail = False
        self.hold = False
        self.fail_abort = False
        self.eager_fail = False
        self.missing = False
        self.misidentify = False
        self.active = asyncio.Event()
        self.abort_event = asyncio.Event()

    async def get_supported_tasks(self):
        return ("token_embed",)

    def encode(self, request_id, prompt, **kwargs):
        if self.eager_fail and request_id.endswith("-1"):
            raise RuntimeError("prepare failed")

        async def run():
            self.started.append(request_id)
            self.prompts.append(prompt)
            self.active.set()
            try:
                if self.hold:
                    await asyncio.Event().wait()
                await asyncio.sleep(0.01 if request_id.endswith("-1") else 0.03)
                if self.fail and request_id.endswith("-0"):
                    raise RuntimeError("block failed")
                if self.missing and request_id.endswith("-0"):
                    return
                ids = prompt["prompt_token_ids"]
                count = ids.count(DOC_TOKEN_ID)
                vectors = torch.ones(count + 1, 512, dtype=torch.bfloat16)
                yield PoolingRequestOutput(
                    request_id="wrong" if self.misidentify else request_id,
                    outputs=PoolingOutput(data=vectors),
                    prompt_token_ids=ids,
                    num_cached_tokens=0,
                    finished=True,
                )
            finally:
                self.closed.append(request_id)

        return run()

    async def abort(self, request_ids):
        self.aborted.extend(request_ids)
        self.abort_event.set()
        if self.fail_abort:
            raise RuntimeError("cleanup failed")


@pytest.fixture(scope="module")
def config_renderer(snapshot):
    import os

    os.environ["VLLM_PLUGINS"] = "jina_reranker_v35"
    register()
    mc = ModelConfig(
        model=str(snapshot),
        tokenizer=str(snapshot),
        runner="pooling",
        model_impl="vllm",
        dtype="bfloat16",
        max_model_len=131072,
        hf_overrides={"architectures": ["JinaRerankerV35"], "embedding_size": 512},
    )
    config = VllmConfig(model_config=mc)
    renderer = HfRenderer(
        config, cached_tokenizer_from_config(mc, local_files_only=True)
    )
    yield config, renderer
    renderer.shutdown()


@pytest_asyncio.fixture
async def serving(config_renderer):
    config, renderer = config_renderer
    engine = FakeEngine(config, renderer)
    models = OpenAIServingModels(
        engine, [BaseModelPath("jina35", config.model_config.model)]
    )
    handler = Jina35ServingScores(
        engine,
        models,
        supported_tasks=("token_embed",),
        enable_flash_late_interaction=False,
        request_logger=None,
        chat_template_config=ChatTemplateConfig(),
    )
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.serving_scores = handler
    app.include_router(router)
    init_exception_handler(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        yield engine, handler, client


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank", "/v2/rerank"])
@pytest.mark.parametrize("top_n", [None, 0, 1, 5])
async def test_native_rerank_formatter(serving, path, top_n):
    engine, handler, client = serving
    body = dict(model="jina35", query="query", documents=["original", "same", ""])
    if top_n is not None:
        body["top_n"] = top_n
    response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    count = 1 if top_n == 1 else 3
    assert [r["index"] for r in data["results"]] == list(range(count))
    assert [r["document"]["text"] for r in data["results"]] == body["documents"][:count]
    assert data["usage"]["prompt_tokens"] == sum(
        len(p["prompt_token_ids"]) for p in engine.prompts
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        ("text_1", "text_2"),
        ("data_1", "data_2"),
        ("queries", "documents"),
        ("queries", "items"),
    ],
)
@pytest.mark.parametrize(
    "queries,docs,blocks",
    [("q", ["a", "b"], 1), (["q", "q"], ["a", "b"], 2), ("q", "a", 1)],
)
async def test_score_groups_and_engine_ids(serving, fields, queries, docs, blocks):
    engine, handler, client = serving
    response = await client.post("/score", json={fields[0]: queries, fields[1]: docs})
    assert response.status_code == 200, response.text
    assert len(engine.started) == blocks
    assert [r["index"] for r in response.json()["data"]] == list(
        range(1 if isinstance(docs, str) else len(docs))
    )
    request = ScoreTextRequest(text_1=queries, text_2=docs)
    ctx = await handler._init_ctx(handler.io_processor, request)
    await handler._preprocessing(handler.io_processor, ctx)
    assert [list(b.prompt_token_ids) for b in ctx.plan.blocks] == [
        p["prompts"]["prompt_token_ids"] for p in ctx.engine_inputs
    ]
    for item in ctx.engine_inputs:
        assert item["params"].task == "token_embed"
        assert item["params"].late_interaction_params is None
        assert item["params"].dimensions is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        dict(max_tokens_per_query=512),
        dict(max_tokens_per_doc=100),
        dict(instruction="different"),
        dict(chat_template_kwargs={"enable_thinking": True}),
        dict(truncate_prompt_tokens=-1),
        dict(padding="max_length"),
        dict(use_activation=True),
        dict(top_n=-1),
        dict(top_n=None),
        dict(documents=[]),
        dict(query={"content": []}),
    ],
)
async def test_preflight_bad_options_submit_nothing(serving, extra):
    engine, _, client = serving
    body = dict(query="q", documents=["doc"])
    body.update(extra)
    response = await client.post("/rerank", json=body)
    assert 400 <= response.status_code < 500, response.text
    assert engine.started == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queries,docs",
    [
        ([], ["a"]),
        (["a", "b"], "doc"),
        (["a", "b"], ["doc"]),
        (["a", "b"], ["x", "y", "z"]),
    ],
)
async def test_invalid_score_shapes(serving, queries, docs):
    engine, _, client = serving
    response = await client.post("/score", json={"text_1": queries, "text_2": docs})
    assert 400 <= response.status_code < 500, response.text
    assert not engine.started


@pytest.mark.asyncio
async def test_concurrent_plans_and_reused_external_ids(serving):
    engine, _, client = serving
    responses = await asyncio.gather(
        *[
            client.post(
                "/rerank",
                headers={"X-Request-Id": "same"},
                json={"query": "q", "documents": ["doc"] * count},
            )
            for count in (2, 126)
        ]
    )
    assert [len(r.json()["results"]) for r in responses] == [2, 126]
    assert len(set(engine.started)) == 3
    assert all(r.startswith("score-same-") for r in engine.started)
    assert set(engine.closed) == set(engine.started)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fail", "missing", "eager_fail"])
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_failure_aborts_whole_request_and_preserves_error(
    serving, mode, cleanup_error
):
    engine, handler, _ = serving
    setattr(engine, mode, True)
    engine.fail_abort = cleanup_error
    with pytest.raises(
        (RuntimeError, ValueError), match="block failed|all prompts|prepare failed"
    ):
        await handler(RerankRequest(query="q", documents=["doc"] * 126))
    assert len(engine.aborted) == 2
    assert set(engine.closed) == set(engine.started)


@pytest.mark.asyncio
async def test_cancellation_closes_before_abort(serving):
    engine, handler, _ = serving
    engine.hold = True
    task = asyncio.create_task(
        handler(RerankRequest(query="q", documents=["doc"] * 126))
    )
    await asyncio.wait_for(engine.active.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(engine.aborted) == 2
    assert set(engine.closed) == set(engine.started)


@pytest.mark.asyncio
async def test_install_is_scoped_and_idempotent(serving):
    engine, handler, _ = serving
    old = ServingScores(
        engine,
        handler.models,
        supported_tasks=("token_embed",),
        enable_flash_late_interaction=False,
        request_logger=None,
        chat_template_config=ChatTemplateConfig(),
    )
    state = SimpleNamespace(serving_scores=old)
    await install_serving(engine, state)
    assert isinstance(state.serving_scores, Jina35ServingScores)
    replacement = state.serving_scores
    await install_serving(engine, state)
    assert state.serving_scores is replacement


@pytest.mark.asyncio
async def test_native_disconnect_cleans_engine_requests(serving):
    engine, _, client = serving
    engine.hold = True
    payload = json.dumps({"query": "q", "documents": ["doc"] * 126}).encode()
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await engine.active.wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    app = client._transport.app
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/rerank",
        "raw_path": b"/rerank",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), 3)
    await asyncio.wait_for(engine.abort_event.wait(), 3)
    assert len(engine.aborted) == 2
    assert set(engine.closed) == set(engine.started)


@pytest.mark.asyncio
async def test_fixed_profile_equivalent_options_and_empty_text(serving):
    engine, _, client = serving
    response = await client.post(
        "/v1/score",
        json={
            "text_1": "",
            "text_2": ["", "doc"],
            "max_tokens_per_query": 1024,
            "max_tokens_per_doc": 8192,
            "instruction": "",
            "use_activation": False,
            "padding": "do_not_pad",
            "truncation_side": "left",
        },
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "misidentify"])
async def test_internal_output_contract_errors_are_500(serving, failure):
    engine, _, client = serving
    setattr(engine, failure, True)
    response = await client.post(
        "/rerank", json={"query": "q", "documents": ["doc"] * 126}
    )
    assert response.status_code == 500, response.text
    assert "results" not in response.json()


def test_plugin_startup_logs_use_native_handlers():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.logger import configure_logging; "
            "from vllm.config import LoggingConfig; "
            "configure_logging(LoggingConfig()); "
            "from jina_vllm.serving import logger; logger.info('plugin-log-receipt')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "plugin-log-receipt" in result.stdout + result.stderr
