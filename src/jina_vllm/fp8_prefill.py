# SPDX-License-Identifier: Apache-2.0
"""Ephemeral FP8 Q/K/V and causal packed prefill for Jina's D128 GQA."""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import find_seq_idx

# FP8 MMA targets of the pinned Triton CUDA backend.
FP8_DEVICE_CAPABILITIES = {(8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)}


def attention_tile(
    capability: tuple[int, int], window: int
) -> tuple[int, int, int, int]:
    # Matrix rows include both query heads of each KV head. Use smaller
    # K/V tiles on targets without the RTX 5090 tuning measurements.
    if capability == (12, 0):
        return (64, 64, 4, 3) if window else (64, 128, 4, 2)
    if capability in ((9, 0), (10, 0), (10, 3)):
        return (64, 64, 4, 2)
    return (32, 64, 4, 2)


@triton.jit
def _quantize_heads(x):
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1.0e-12)
    quantized = tl.clamp(x / scale[:, None], -448.0, 448.0).to(tl.float8e4nv)
    return quantized, scale


@triton.jit
def _pack(Q, K, V, Packed, Scales, T, QS, KS, VS, R: tl.constexpr):
    # Use exactly as many blocks as each projection needs.
    pid = tl.program_id(0)
    q_blocks = tl.cdiv(T * 16, R)
    kv_blocks = tl.cdiv(T * 8, R)
    rows = tl.arange(0, R)
    dim = tl.arange(0, 128)
    if pid < q_blocks:
        idx = pid * R + rows
        src = Q + (idx // 16) * QS + idx % 16 * 128
        dst = idx
        valid = idx < T * 16
    else:
        kv_block = pid - q_blocks
        if kv_block < kv_blocks:
            idx = kv_block * R + rows
            src = K + (idx // 8) * KS + idx % 8 * 128
            dst = T * 16 + idx
        else:
            idx = (kv_block - kv_blocks) * R + rows
            src = V + (idx // 8) * VS + idx % 8 * 128
            dst = T * 24 + idx
        valid = idx < T * 8
    x = tl.load(src[:, None] + dim[None, :], valid[:, None], 0).to(tl.float32)
    quantized, scale = _quantize_heads(x)
    tl.store(Packed + dst[:, None] * 128 + dim[None, :], quantized, valid[:, None])
    tl.store(Scales + dst, scale, valid)


@triton.jit
def _pack_qkv(Q, Packed, Scales, T, R: tl.constexpr):
    # The fused Q/K norm and RoPE output is a single interleaved QKV matrix.
    idx = tl.program_id(0) * R + tl.arange(0, R)
    token = idx // 32
    head = idx % 32
    dim = tl.arange(0, 128)
    valid = token < T
    x = tl.load(Q + idx[:, None] * 128 + dim[None, :], valid[:, None], 0)
    quantized, scale = _quantize_heads(x.to(tl.float32))

    # Store the three projections in separate planes of the same allocation.
    kv_row = token * 8 + head % 8
    dst = tl.where(head < 16, token * 16 + head, T * 16 + kv_row)
    dst += tl.where(head >= 24, T * 8, 0)
    tl.store(Packed + dst[:, None] * 128 + dim[None, :], quantized, valid[:, None])
    tl.store(Scales + dst, scale, valid)


def interleaved_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.stride() == k.stride() == v.stride() == (4096, 128, 1)
        and k.data_ptr() == q.data_ptr() + 2048 * q.element_size()
        and v.data_ptr() == q.data_ptr() + 3072 * q.element_size()
        and q.untyped_storage().data_ptr()
        == k.untyped_storage().data_ptr()
        == v.untyped_storage().data_ptr()
    )


def pack(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Qwen3 has contiguous head dimensions, with per-tensor token strides.
    tokens = q.shape[0]
    packed = torch.empty((tokens * 32, 128), device=q.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(tokens * 32, device=q.device, dtype=torch.float32)
    rows = 32
    if interleaved_qkv(q, k, v):
        _pack_qkv[(triton.cdiv(tokens * 32, rows),)](
            q, packed, scales, tokens, rows, num_warps=4
        )
    else:
        blocks = triton.cdiv(tokens * 16, rows) + 2 * triton.cdiv(tokens * 8, rows)
        _pack[(blocks,)](
            q,
            k,
            v,
            packed,
            scales,
            tokens,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            rows,
            num_warps=4,
        )
    return packed, scales


@triton.jit
def _attention(
    Packed,
    Scales,
    Out,
    Offsets,
    T,
    OS,
    OHS,
    SM,
    B: tl.constexpr,
    WINDOW: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_P: tl.constexpr,
):
    qblock: tl.constexpr = M // 2
    tile_id = tl.program_id(0)
    batch = find_seq_idx(Offsets, tile_id, B, qblock, True) if B > 1 else 0
    start = tl.load(Offsets + batch)
    block = (tile_id - (start // qblock + batch)) * qblock
    kv_head = tl.program_id(1)
    length = tl.load(Offsets + batch + 1) - start
    if block < length:
        row_idx = tl.arange(0, M)
        rows = block + row_idx // 2
        q_head = kv_head * 2 + row_idx % 2
        cols = tl.arange(0, N)
        d = tl.arange(0, 128)
        q_row = (start + rows) * 16 + q_head
        q = tl.load(
            Packed + q_row[:, None] * 128 + d[None, :], rows[:, None] < length, 0.0
        )
        sq = tl.load(Scales + q_row, rows < length, 0) * SM
        maximum = tl.full((M,), -float("inf"), tl.float32)
        denominator = tl.zeros((M,), tl.float32)
        acc = tl.zeros((M, 128), tl.float32)
        lo = tl.maximum(0, ((block - WINDOW + 1) // N) * N) if WINDOW else 0
        hi = tl.minimum(length, block + qblock)
        for tile in range(lo, hi, N):
            kpos = tile + cols
            k_row = T * 16 + (start + kpos) * 8 + kv_head
            v_row = k_row + T * 8
            k = tl.load(
                Packed + k_row[None, :] * 128 + d[:, None], kpos[None, :] < length, 0.0
            )
            logits = tl.dot(q, k)  # QK: E4M3 x E4M3.
            sk = tl.load(Scales + k_row, kpos < length, 0)
            logits = logits * sq[:, None] * sk[None, :]
            mask = (
                (rows[:, None] < length)
                & (rows[:, None] >= kpos[None, :])
                & (kpos[None, :] < length)
            )
            if WINDOW:
                mask &= rows[:, None] - kpos[None, :] < WINDOW
            logits = tl.where(mask, logits, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(logits, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            p = tl.exp2(logits - safe_max[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)

            v = tl.load(
                Packed + v_row[:, None] * 128 + d[None, :], kpos[:, None] < length, 0.0
            )
            sv = tl.load(Scales + v_row, kpos < length, 0)
            if TILE_P:
                scale_p = tl.maximum(tl.max(sv, 0) / 256.0, 1.0e-15)
                p8 = (p * (sv / scale_p)[None, :]).to(tl.float8e4nv)
                contrib = tl.dot(p8, v)  # PV with per-tile P scale: E4M3 x E4M3.
                acc = acc * alpha[:, None] + contrib * scale_p
            else:
                weighted = p * sv[None, :]
                scale_p = tl.maximum(tl.max(weighted, 1) / 448.0, 1.0e-15)
                p8 = tl.clamp(weighted / scale_p[:, None], 0.0, 448.0).to(tl.float8e4nv)
                contrib = tl.dot(p8, v)  # PV with per-row P scale: E4M3 x E4M3.
                acc = acc * alpha[:, None] + contrib * scale_p[:, None]
            maximum = new_max
        out = Out + (start + rows[:, None]) * OS + q_head[:, None] * OHS + d[None, :]
        normalized = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(out, normalized, rows[:, None] < length)


def attend(
    packed: tuple[torch.Tensor, torch.Tensor],
    offsets: torch.Tensor,
    max_seq_len: int,
    output: torch.Tensor,
    window: int,
    tile_p: bool,
    tile: tuple[int, int, int, int],
    scale: float,
):
    data, scales = packed
    m, n, warps, stages = tile
    batch = offsets.numel() - 1
    tokens = data.shape[0] // 32
    qblock = m // 2
    blocks = (
        triton.cdiv(max_seq_len, qblock) if batch == 1 else tokens // qblock + batch
    )
    return _attention[(blocks, 8)](
        data,
        scales,
        output,
        offsets,
        tokens,
        output.stride(0),
        output.stride(1),
        scale * math.log2(math.e),
        batch,
        window,
        m,
        n,
        tile_p,
        num_warps=warps,
        num_stages=stages,
    )
