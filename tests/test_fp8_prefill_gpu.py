"""Independent FP32 and FP8-policy oracles for the CUDA prefill kernel."""

import math
import os

import pytest
import torch
from vllm.triton_utils import triton

from jina_vllm import fp8_block_prefill as block
from jina_vllm.fp8_prefill import attend, attention_tile, pack

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


def quantize_blocks(x):
    length, heads, dim = x.shape
    padded = torch.nn.functional.pad(x.float(), (0, 0, 0, 0, 0, -length % 128))
    tiles = padded.reshape(-1, 128, heads, dim)
    scales = tiles.abs().amax((1, 3)) * (1.0 / 448.0)
    scales = torch.where(scales > 0, scales, 1.0 / 448.0)
    data = (tiles / scales[:, None, :, None]).to(torch.float8_e4m3fn)
    return data, scales


def block_oracle(q, k, v, window, n, scale):
    length = len(q)
    payload = [quantize_blocks(x) for x in (q, k, v)]
    data = [x[0].flatten(0, 1)[:length].float().transpose(0, 1) for x in payload]
    scales = [x[1].transpose(0, 1) for x in payload]
    qh, kh, vh = data
    sq, sk, sv = scales
    qk = qh @ kh.repeat_interleave(2, 0).transpose(-1, -2)
    sq = sq.repeat_interleave(128, 1)[:, :length] * scale
    sk, sv = [x.repeat_interleave(2, 0) for x in (sk, sv)]
    pos = torch.arange(length, device=q.device)
    maximum = qk.new_full((16, length), -torch.inf)
    den = torch.zeros_like(maximum)
    acc = torch.zeros_like(qh)
    for i in range(0, length, n):
        cols = pos[i : i + n]
        delta = pos[:, None] - cols
        mask = delta >= 0
        if window:
            mask &= delta < window
        logits = qk[..., i : i + n] * (sq * sk[:, i // 128, None])[..., None]
        logits = logits.masked_fill(~mask, -torch.inf)
        new = torch.maximum(maximum, logits.amax(-1))
        safe = torch.where(new.isneginf(), 0.0, new)
        alpha = (maximum - safe).exp()
        p = (logits - safe[..., None]).exp()
        p8 = (p * 256).to(torch.float8_e4m3fn).float()
        values = vh[:, i : i + n].repeat_interleave(2, 0)
        acc = (
            acc * alpha[..., None]
            + (p8 @ values) * (sv[:, i // 128] / 256)[:, None, None]
        )
        den = den * alpha + p.sum(-1)
        maximum = new
    return (acc / den[..., None]).transpose(0, 1), payload


@pytest.mark.parametrize("window,m,n", [(0, 128, 64), (1024, 32, 128), (1024, 64, 32)])
@torch.inference_mode()
def test_block_prefill_matches_independent_policy(window, m, n):
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    lengths = [19, 1153, 37]
    tokens = sum(lengths)
    qkv = torch.randn((tokens + 1, 32, 128), device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[1:].split((16, 8, 8), dim=1)
    v[19:147].zero_()
    v[147:] *= 1.5
    offsets = torch.tensor([0, 19, 1172, tokens], dtype=torch.int32, device="cuda")
    packed = block.pack(q, k, v, offsets)
    backing = torch.full(
        (tokens + 3, 33, 140), -42, device="cuda", dtype=torch.bfloat16
    )
    out = backing[2:-1, 1:33:2, 4:132]
    scale = 0.125
    config = triton.Config({"M": m, "N": n}, num_warps=4, num_stages=2)
    block.attend(packed, offsets, out, window, scale, {(window, 2048): config})

    parts = []
    start = 0
    for seq, length in enumerate(lengths):
        end = start + length
        inputs = [x[start:end] for x in (q, k, v)]
        expected, payload = block_oracle(*inputs, window, n, scale)
        error = (out[start:end].float() - expected).norm() / expected.norm()
        assert error < 0.003

        slot = start // 128 + seq
        for i, (quantized, scales) in enumerate(payload):
            count, _, heads, _ = quantized.shape
            data, factors = packed[2 * i : 2 * i + 2]
            expected_data = quantized.permute(2, 0, 1, 3)
            if i == 2:
                expected_data = expected_data.transpose(2, 3)
            torch.testing.assert_close(
                data.reshape(heads, -1, 128, 128)[:, slot : slot + count].view(
                    torch.uint8
                ),
                expected_data.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(factors[:, slot : slot + count], scales.T)
        single = torch.empty_like(out[start:end])
        cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
        block.attend(
            block.pack(*inputs, cu),
            cu,
            single,
            window,
            scale,
            {(window, block.token_bucket(length)): config},
        )
        parts.append(single)
        start = end
    torch.testing.assert_close(out, torch.cat(parts), atol=0, rtol=0)
    out.fill_(-42)
    assert bool((backing == -42).all())


@torch.inference_mode()
def test_block_prefill_window_excludes_key_at_distance_1024():
    length = 1153
    q = torch.zeros((length, 16, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q[:, :8]), torch.zeros_like(q[:, :8])
    v[0] = 1024
    v[127] = 2048
    cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    config = triton.Config({"M": 128, "N": 64}, num_warps=8, num_stages=3)
    block.attend(
        block.pack(q, k, v, cu), cu, out, 1024, 128**-0.5, {(1024, 2048): config}
    )
    expected = torch.tensor([3, 2, 2, 0, 0], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        out[[1023, 1024, 1150, 1151, 1152], 0, 0], expected, atol=0, rtol=0
    )
