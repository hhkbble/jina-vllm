import os
import pickle
import subprocess
import sys
from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest

from jina_vllm.contract import ARCH, PLUGIN_NAME
from jina_vllm.plugin import Jina35EndpointPlugin


def test_entrypoints_and_lightweight_import():
    for group in ("vllm.general_plugins", "vllm.endpoint_plugins"):
        assert PLUGIN_NAME in entry_points(group=group).names
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import jina_vllm.plugin; import sys; "
            "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "allowed,registered", [(None, False), ("other", False), (PLUGIN_NAME, True)]
)
def test_alias_requires_explicit_allowlist(allowed, registered):
    code = (
        "from jina_vllm.plugin import register; register(); "
        "from vllm.model_executor.models import ModelRegistry; "
        "from vllm.model_executor.models.config import MODELS_CONFIG_MAP; "
        f"assert ({ARCH!r} in ModelRegistry.get_supported_archs()) == {registered!r}; "
        f"assert ({ARCH!r} in MODELS_CONFIG_MAP) == {registered!r}"
    )
    env = dict(os.environ)
    env.pop("VLLM_PLUGINS", None)
    if allowed is not None:
        env["VLLM_PLUGINS"] = allowed
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_fp8_backend_survives_spawn_deserialization():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from jina_vllm.config import FP8_PREFILL_BACKEND, register_prefill_modes

    register_prefill_modes()
    backend = AttentionBackendEnum[FP8_PREFILL_BACKEND]
    payload = pickle.dumps(backend)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys; "
            "from vllm.v1.attention.backends.registry import AttentionBackendEnum; "
            f"assert {FP8_PREFILL_BACKEND!r} not in AttentionBackendEnum.__members__; "
            "backend = pickle.loads(sys.stdin.buffer.read()); "
            f"assert backend is AttentionBackendEnum[{FP8_PREFILL_BACKEND!r}]; "
            f"assert backend.get_class().get_name() == {FP8_PREFILL_BACKEND!r}",
        ],
        input=payload,
        check=True,
    )


@pytest.mark.asyncio
async def test_endpoint_leaves_other_architecture_untouched():
    handler = object()
    state = SimpleNamespace(serving_scores=handler)
    engine = SimpleNamespace(model_config=SimpleNamespace(architecture="Other"))
    await Jina35EndpointPlugin().init_state(engine, state, None)
    assert state.serving_scores is handler
