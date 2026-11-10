import os
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
        f"assert ({ARCH!r} in ModelRegistry.get_supported_archs()) == {registered!r}"
    )
    env = dict(os.environ)
    env.pop("VLLM_PLUGINS", None)
    if allowed is not None:
        env["VLLM_PLUGINS"] = allowed
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@pytest.mark.asyncio
async def test_endpoint_leaves_other_architecture_untouched():
    handler = object()
    state = SimpleNamespace(serving_scores=handler)
    engine = SimpleNamespace(model_config=SimpleNamespace(architecture="Other"))
    await Jina35EndpointPlugin().init_state(engine, state, None)
    assert state.serving_scores is handler
