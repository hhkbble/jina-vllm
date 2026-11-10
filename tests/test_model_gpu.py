"""Assertions over real execution receipts produced by the offline tools.

Set JINA_GPU_ARTIFACTS to the trusted local acceptance directory after running
compare_reference in all three modes. These tests never substitute synthetic
vectors for GPU receipts. The full numerical gate intentionally remains visible.
"""

import gzip
import json
import os
from pathlib import Path

import pytest
import torch

from tools.evaluate_acceptance import validate_structure

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def artifacts():
    path = os.environ.get("JINA_GPU_ARTIFACTS")
    if not path:
        pytest.skip("Set JINA_GPU_ARTIFACTS to real offline execution receipts")
    directory = Path(path)
    return directory, {
        mode: torch.load(directory / f"{mode}.pt", weights_only=False)
        for mode in ["hf", "bf16", "fp8"]
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
        assert (
            row["prompt_token_ids"]
            == next(r for r in runs["hf"]["results"] if r["id"] == row["id"])[
                "prompt_token_ids"
            ]
        )


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


@pytest.mark.parametrize("key", ["bf16_vs_hf", "fp8_vs_bf16", "cpu_merge_vs_hf"])
def test_frozen_numerical_gates(artifacts, key):
    directory, _ = artifacts
    gates = json.loads((directory / "numerical-acceptance.json").read_text())["gates"][
        key
    ]
    assert all(gates.values()) if isinstance(gates, dict) else gates, gates
