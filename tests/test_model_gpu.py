"""Check model structure and kernels using trusted native GPU receipts."""

import gzip
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu


def validate_structure(run):
    meta = run["metadata"]
    linears = meta["linears"]
    attention = meta["attention"]
    assert len(linears) == 112
    assert len(attention) == 28
    assert (
        sum(a["window"] == 1024 and a["spec"] == "SlidingWindowSpec" for a in attention)
        == 16
    )
    assert (
        sum(a["window"] is None and a["spec"] == "FullAttentionSpec" for a in attention)
        == 12
    )
    assert all(
        a["attn_type"] == "decoder"
        and a["kv_dtype"] == "bfloat16"
        and a["backend"] == "FLASH_ATTN"
        and a["flash_version"] == 2
        for a in attention
    )
    assert meta["embedding_dtype"] == "torch.bfloat16"
    assert meta["projector_dtypes"] == ["torch.bfloat16", "torch.bfloat16"]
    assert meta["pooler_head"] == "None"
    if run["mode"] == "fp8":
        assert all(
            linear["method"] == "Fp8PtpcOnlineLinearMethod"
            and linear["kernel"] == "CutlassFP8ScaledMMLinearKernel"
            and linear["dtype"] == "torch.float8_e4m3fn"
            and linear["scale_dtype"] == "torch.float32"
            and linear["input_scale"] == "None"
            for linear in linears
        )
    else:
        assert all(
            linear["method"] == "UnquantizedLinearMethod"
            and linear["dtype"] == "torch.bfloat16"
            for linear in linears
        )


@pytest.fixture(scope="module")
def artifacts():
    path = os.environ.get("JINA_GPU_ARTIFACTS")
    if not path:
        pytest.skip("Set JINA_GPU_ARTIFACTS to real offline execution receipts")
    directory = Path(path)
    return directory, {
        mode: torch.load(directory / f"{mode}.pt", weights_only=False)
        for mode in ["bf16", "fp8"]
    }


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_actual_model_structure_and_raw_outputs(artifacts, mode):
    _, runs = artifacts
    run = runs[mode]
    validate_structure(run)
    assert len(run["results"]) >= 4
    by_id = {row["id"]: row for row in run["results"]}
    assert len(by_id["count126"]["raw"]) == 2
    assert len(by_id["capacity"]["raw"]) == 2
    assert max(map(len, by_id["capacity"]["prompt_token_ids"])) > 8192
    for row in run["results"]:
        assert all(
            t.dtype == torch.bfloat16 and t.device.type == "cpu" and t.shape[1] == 512
            for t in row["raw"]
        )
        assert all(torch.isfinite(t).all() for t in row["raw"])
        assert len(row["raw"]) == len(row["prompt_token_ids"])


def test_actual_sm120_fp8_kernel_trace(artifacts):
    directory, _ = artifacts
    with gzip.open(next((directory / "fp8-trace").glob("*.json.gz"))) as stream:
        trace = json.load(stream)
    events = trace["traceEvents"]
    assert any(e.get("name") == "_C::cutlass_scaled_mm" for e in events)
    assert any(
        e.get("cat") == "kernel"
        and "sm120" in e.get("name", "").lower()
        and "float_e4m3" in e.get("name", "")
        for e in events
    )
