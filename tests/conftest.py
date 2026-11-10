import importlib.util
import os
import sys
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from jina_vllm.contract import MODEL_ID, REVISION


@pytest.fixture(scope="session")
def snapshot():
    path = os.environ.get("JINA_MODEL_DIR")
    if path:
        return Path(path)
    try:
        return Path(
            snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
        )
    except FileNotFoundError:
        pytest.skip("Fixed reference snapshot required; set JINA_MODEL_DIR")


@pytest.fixture(scope="session")
def tokenizer(snapshot):
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


@pytest.fixture(scope="session")
def reference_module(snapshot):
    spec = importlib.util.spec_from_file_location(
        "jina_reference", snapshot / "modeling.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
