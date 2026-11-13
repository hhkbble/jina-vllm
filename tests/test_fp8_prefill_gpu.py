"""Independent FP32 and FP8-policy oracles for the CUDA prefill kernel."""

import math
import os

import pytest
import torch

from jina_vllm.fp8_prefill import attend, attention_tile, interleaved_qkv, pack

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("JINA_RUN_FP8_PREFILL_TESTS") != "1",
        reason="Set JINA_RUN_FP8_PREFILL_TESTS=1 to exercise the CUDA kernels",
    ),
]


def oracle(q, k, v, lengths, window):
    output = []
    start = 0
    for length in lengths:
        end = start + length
        qh, kh, vh = [x[start:end].float().transpose(0, 1) for x in (q, k, v)]
        kh, vh = [x.repeat_interleave(2, dim=0) for x in (kh, vh)]
        pos = torch.arange(length, device=q.device)
        delta = pos[:, None] - pos[None, :]
        mask = delta >= 0
        if window:
            mask &= delta < window
        logits = (qh @ kh.transpose(-1, -2)) / math.sqrt(128)
        output.append(
            (logits.masked_fill(~mask, -torch.inf).softmax(-1) @ vh).transpose(0, 1)
        )
        start = end
    return torch.cat(output)


def policy_oracle(packed, lengths, window, n, tile_p):
    # Evaluate all rows in FP32 without using the kernel's CTA mapping,
    # launch bounds or tile pruning.
    data, scales = packed
    tokens = data.shape[0] // 32
    q, k, v = (
        x.reshape(tokens, heads, 128)
        for x, heads in zip(
            data.split((16 * tokens, 8 * tokens, 8 * tokens)), (16, 8, 8), strict=True
        )
    )
    sq = scales[: 16 * tokens].view(tokens, 16)
    skv = scales[16 * tokens :].view(2, tokens, 8)
    out = []
    start = 0
    for length in lengths:
        end = start + length

        def heads(x, gqa=False, start=start, end=end):
            y = x[start:end].float().transpose(0, 1)
            return y.repeat_interleave(2, 0) if gqa else y

        qh, kh, vh = heads(q), heads(k, True), heads(v, True)
        qs, ks, vs = heads(sq), heads(skv[0], True), heads(skv[1], True)
        score = (
            (qh @ kh.transpose(-1, -2))
            * (qs[..., None] / math.sqrt(128))
            * ks[:, None, :]
        )
        pos = torch.arange(length, device=q.device)
        maximum = torch.full_like(qs, -torch.inf)
        denominator = torch.zeros_like(qs)
        acc = torch.zeros_like(qh)
        for tile in range(0, length, n):
            delta = pos[:, None] - pos[None, tile : tile + n]
            mask = delta >= 0
            if window:
                mask &= delta < window
            logits = score[..., tile : tile + n].masked_fill(~mask, -torch.inf)
            new = torch.maximum(maximum, logits.amax(-1))
            safe = new.masked_fill(new == -torch.inf, 0)
            alpha = (maximum - safe).exp()
            p = (logits - safe[..., None]).exp()
            denominator = denominator * alpha + p.sum(-1)
            vs_tile = vs[:, tile : tile + n]
            weighted = p * vs_tile[:, None, :]
            if tile_p:
                ps = (vs_tile.amax(-1)[:, None] / 256).clamp_min(1e-15)
            else:
                ps = (weighted.amax(-1) / 448).clamp_min(1e-15)
            p8 = (
                (weighted / ps[..., None]).clamp(0, 448).to(torch.float8_e4m3fn).float()
            )
            acc = acc * alpha[..., None] + (p8 @ vh[:, tile : tile + n]) * ps[..., None]
            maximum = new
        out.append((acc / denominator[..., None]).transpose(0, 1))
        start = end
    return torch.cat(out).to(torch.bfloat16)


def run_attention(inputs, lengths, window, tile_p):
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    q = inputs[0]
    tile = attention_tile(torch.cuda.get_device_capability(), window)
    out = torch.full(q.shape, torch.nan, device=q.device, dtype=q.dtype)
    packed = pack(*inputs)
    attend(packed, offsets, max(lengths), out, window, tile_p, tile, 128**-0.5)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    return out, packed, tile[1]


@pytest.mark.parametrize("window", [0, 1024])
@pytest.mark.parametrize("tile_p", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_fp8_prefill_matches_independent_oracles(window, tile_p, fused):
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(71)
    lengths = [19, 1153, 37]
    tokens = sum(lengths)
    qkv = torch.randn((tokens + 1, 32, 128), device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[1:].split((16, 8, 8), dim=1)
    v *= torch.rand((tokens, 8, 1), device="cuda") ** 2
    if not fused:
        q, k = q.contiguous(), k.contiguous()
    assert interleaved_qkv(q, k, v) == fused
    out, packed, n = run_attention((q, k, v), lengths, window, tile_p)
    expected = policy_oracle(packed, lengths, window, n, tile_p)
    original = oracle(q, k, v, lengths, window)
    assert (out.float() - expected.float()).norm() / expected.float().norm() < 0.006
    assert (out.float() - original).norm() / original.norm() < 0.05

    parts = [
        run_attention(views, [len(views[0])], window, tile_p)[0]
        for views in zip(
            q.split(lengths), k.split(lengths), v.split(lengths), strict=True
        )
    ]
    torch.testing.assert_close(out, torch.cat(parts), rtol=0, atol=0)


@pytest.mark.parametrize("window", [0, 1024])
@pytest.mark.parametrize("tile_p", [False, True])
@torch.inference_mode()
def test_fp8_prefill_window_boundary_and_zero(window, tile_p):
    lengths = [13, 1153, 2]
    q = torch.zeros((sum(lengths), 16, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q[:, :8]), torch.zeros_like(q[:, :8])
    zero, _, _ = run_attention((q, k, v), lengths, window, tile_p)
    assert torch.count_nonzero(zero).item() == 0

    start = lengths[0]
    v[start] = torch.arange(1, 9, device=q.device)[:, None] * 4
    v[start + 127] = 16
    v[start + 1024] = 2
    v[-2:] = 4
    v[-1] = 8
    out, _, _ = run_attention((q, k, v), lengths, window, tile_p)
    expected = oracle(q, k, v, lengths, window)
    rows = [start + i for i in (0, 1023, 1024, 1150, 1151, 1152)] + [-2, -1]
    torch.testing.assert_close(
        out[rows].float(), expected[rows], atol=0.00015, rtol=0.025
    )


@torch.inference_mode()
def test_fused_qkv_views_pack_like_independent_inputs():
    torch.manual_seed(29)
    qkv = torch.randn((21, 32, 128), device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[1:-1].split((16, 8, 8), dim=1)
    assert (q.storage_offset(), k.storage_offset(), v.storage_offset()) == (
        4096,
        6144,
        7168,
    )
    assert interleaved_qkv(q, k, v)
    assert not interleaved_qkv(q, k, qkv[:-2, 24:])
    assert not interleaved_qkv(q, k, v.clone())
    data, scales = pack(q, k, v)
    separate_data, separate_scales = pack(q.contiguous(), k.contiguous(), v)
    torch.testing.assert_close(
        data.view(torch.uint8), separate_data.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, separate_scales, atol=0, rtol=0)
    expected = torch.cat([x.reshape(-1, 128) for x in (q, k, v)]).float()
    expected_scales = (expected.abs().amax(-1) / 448).clamp_min(1e-12)
    expected_fp8 = (expected / expected_scales[:, None]).clamp(-448, 448)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=2e-7)
    torch.testing.assert_close(
        data.view(torch.uint8),
        expected_fp8.to(data.dtype).view(torch.uint8),
        atol=0,
        rtol=0,
    )
