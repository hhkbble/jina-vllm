# SPDX-License-Identifier: Apache-2.0
"""Sequence-local block FP8 prefill and worker startup tuning."""

import math
from functools import partial
from itertools import product
from time import monotonic

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.triton_attention_helpers import find_seq_idx

logger = init_logger("vllm.jina_vllm.fp8_block_prefill")

BLOCK = 128
_BLOCK = tl.constexpr(BLOCK)
WINDOWS = (0, 1024)
BUCKETS = (1024, 2048, 4096, 8192, 16384)


def token_bucket(tokens: int) -> int:
    return min(BUCKETS[-1], max(BUCKETS[0], next_power_of_2(tokens)))


@triton.jit
def _pack(X, Data, Scales, Offsets, Pitch, TS, HS, B, TRANSPOSE: tl.constexpr):
    slot = tl.program_id(0)
    head = tl.program_id(1)
    seq = find_seq_idx(Offsets, slot, B, _BLOCK, True) if B > 1 else 0
    start = tl.load(Offsets + seq)
    length = tl.load(Offsets + seq + 1) - start
    block = (slot - (start // _BLOCK + seq)) * _BLOCK
    if block < length:
        row = block + tl.arange(0, _BLOCK)
        dim = tl.arange(0, 128)
        src = (start + row).to(tl.int64) * TS + head.to(tl.int64) * HS
        x = tl.load(X + src[:, None] + dim[None, :], row[:, None] < length, 0)
        x = x.to(tl.float32)
        scale = tl.max(tl.max(tl.abs(x), 1), 0) * (1.0 / 448.0)
        scale = tl.where(scale > 0, scale, 1.0 / 448.0)
        fp8 = tl.div_rn(x, scale).to(tl.float8e4nv)
        dst = (head.to(tl.int64) * Pitch + slot * _BLOCK) * 128
        token = row - block
        if TRANSPOSE:
            dst = dst + dim[None, :] * _BLOCK + token[:, None]
        else:
            dst = dst + token[:, None] * 128 + dim[None, :]
        tl.store(Data + dst, fp8)
        tl.store(Scales + head * (Pitch // _BLOCK) + slot, scale)


def pack(q, k, v, offsets) -> tuple[torch.Tensor, ...]:
    tokens = q.shape[0]
    batch = offsets.numel() - 1
    # Each sequence starts at slot floor(start / BLOCK) + sequence index.
    # This reserves padding per sequence and avoids a host read of the offsets.
    slots = triton.cdiv(tokens, BLOCK) if batch == 1 else tokens // BLOCK + batch
    pitch = slots * BLOCK
    packed = []
    for index, x in enumerate((q, k, v)):
        heads = x.shape[1]
        data = torch.empty(
            (heads, pitch, 128), device=x.device, dtype=torch.float8_e4m3fn
        )
        scales = torch.empty((heads, slots), device=x.device, dtype=torch.float32)
        _pack[(slots, heads)](
            x,
            data,
            scales,
            offsets,
            pitch,
            x.stride(0),
            x.stride(1),
            batch,
            TRANSPOSE=index == 2,
            num_warps=8,
        )
        packed.extend((data, scales))
    return tuple(packed)


@triton.jit
def _step(
    q,
    K,
    V,
    KS,
    VS,
    qscale,
    length,
    rows,
    dim,
    cols,
    tile,
    base,
    kh,
    pitch,
    maximum,
    denominator,
    acc,
    WINDOW: tl.constexpr,
    MASK: tl.constexpr,
):
    pos = tile + cols
    slot = kh * (pitch // _BLOCK) + (base + tile) // _BLOCK
    sk = tl.load(KS + slot)
    sv = tl.load(VS + slot)
    plane = kh.to(tl.int64) * pitch + base
    k = tl.load(K + (plane + pos[None, :]) * 128 + dim[:, None])
    v = tl.load(
        V
        + (plane + tile // _BLOCK * _BLOCK) * 128
        + (pos % _BLOCK)[:, None]
        + dim[None, :] * _BLOCK
    )
    logits = tl.dot(q, k, max_num_imprecise_acc=0) * (qscale * sk)
    if MASK:
        delta = rows[:, None] - pos[None, :]
        valid = (delta >= 0) & (pos[None, :] < length)
        if WINDOW:
            valid &= delta < WINDOW
        logits = tl.where(valid, logits, -float("inf"))

    new_max = tl.maximum(maximum, tl.max(logits, 1))
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp2(maximum - safe_max)
    p = tl.exp2(logits - safe_max[:, None])
    denominator = denominator * alpha + tl.sum(p, 1)
    p8 = (p * 256.0).to(tl.float8e4nv)
    contrib = tl.dot(p8, v, max_num_imprecise_acc=0)
    acc = acc * alpha[:, None] + contrib * (sv * (1.0 / 256.0))
    return new_max, denominator, acc


@triton.jit(do_not_specialize=["B", "BUCKET"])
def _attention(
    Q,
    K,
    V,
    QS,
    KS,
    VS,
    Out,
    Offsets,
    Pitch,
    OS,
    OHS,
    SM,
    B: tl.int32,
    WINDOW: tl.constexpr,
    BUCKET: tl.int32,
    M: tl.constexpr,
    N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    qh = tl.program_id(1)
    kh = qh // 2
    seq = find_seq_idx(Offsets, tile_id, B, M, True) if B > 1 else 0
    start = tl.load(Offsets + seq)
    length = tl.load(Offsets + seq + 1) - start
    block = (tile_id - (start // M + seq)) * M
    if block < length:
        base = (start // _BLOCK + seq) * _BLOCK
        rows = block + tl.arange(0, M)
        cols = tl.arange(0, N)
        dim = tl.arange(0, 128)
        qscale = tl.load(QS + qh * (Pitch // _BLOCK) + (base + block) // _BLOCK) * SM
        plane = qh.to(tl.int64) * Pitch + base
        q = tl.load(Q + (plane + rows[:, None]) * 128 + dim[None, :])
        maximum = tl.full((M,), -float("inf"), tl.float32)
        denominator = tl.zeros((M,), tl.float32)
        acc = tl.zeros((M, 128), tl.float32)

        end = tl.minimum(length, block + M)
        right = block // N * N
        left = 0
        if WINDOW:
            left = tl.maximum(0, ((block - WINDOW + 1) // N) * N)
            # The last real query row determines the fully visible left edge.
            left_end = tl.minimum(right, tl.cdiv(tl.maximum(0, end - WINDOW), N) * N)
            for tile in tl.range(left, left_end, N):
                maximum, denominator, acc = _step(
                    q,
                    K,
                    V,
                    KS,
                    VS,
                    qscale,
                    length,
                    rows,
                    dim,
                    cols,
                    tile,
                    base,
                    kh,
                    Pitch,
                    maximum,
                    denominator,
                    acc,
                    WINDOW,
                    True,
                )
            left = left_end
        for tile in tl.range(left, right, N):
            maximum, denominator, acc = _step(
                q,
                K,
                V,
                KS,
                VS,
                qscale,
                length,
                rows,
                dim,
                cols,
                tile,
                base,
                kh,
                Pitch,
                maximum,
                denominator,
                acc,
                WINDOW,
                False,
            )
        for tile in tl.range(right, end, N):
            maximum, denominator, acc = _step(
                q,
                K,
                V,
                KS,
                VS,
                qscale,
                length,
                rows,
                dim,
                cols,
                tile,
                base,
                kh,
                Pitch,
                maximum,
                denominator,
                acc,
                WINDOW,
                True,
            )
        dst = (start + rows).to(tl.int64) * OS + qh.to(tl.int64) * OHS
        result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(Out + dst[:, None] + dim[None, :], result, rows[:, None] < length)


def _launch(kernel, packed, offsets, output, window, scale, config=None):
    q, qs, k, ks, v, vs = packed
    tokens = output.shape[0]
    batch = offsets.numel() - 1

    def grid(meta):
        m = meta["M"]
        blocks = triton.cdiv(tokens, m) if batch == 1 else tokens // m + batch
        return blocks, 16

    return kernel[grid](
        q,
        k,
        v,
        qs,
        ks,
        vs,
        output,
        offsets,
        q.shape[1],
        output.stride(0),
        output.stride(1),
        scale * math.log2(math.e),
        batch,
        WINDOW=window,
        BUCKET=token_bucket(tokens),
        **(config.all_kwargs() if config is not None else {}),
    )


def attend(packed, offsets, output, window, scale, configs):
    key = (window, token_bucket(output.shape[0]))
    return _launch(_attention, packed, offsets, output, window, scale, configs[key])


@torch.inference_mode()
def warmup(device: torch.device) -> dict[tuple[int, int], triton.Config]:
    from triton.testing import do_bench_cudagraph

    candidates = [
        triton.Config({"M": m, "N": n}, num_warps=w, num_stages=s)
        for m, n, w, s in product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3, 4))
    ]
    # A fresh tuner per worker keeps the in-memory cache tied to its device.
    # Requests use the raw JIT function with a selected config.
    tuned = triton.autotune(
        configs=candidates,
        key=["WINDOW", "BUCKET"],
        cache_results=True,
        do_bench=partial(do_bench_cudagraph, rep=10),
    )(_attention)
    configs = {}
    start = monotonic()
    with torch.cuda.device(device):
        rng = torch.Generator(device=device).manual_seed(73)
        for bucket in BUCKETS:
            qkv = torch.randn(
                (bucket, 32, 128), device=device, dtype=torch.bfloat16, generator=rng
            )
            offsets = torch.tensor([0, bucket], device=device, dtype=torch.int32)
            packed = pack(*qkv.split((16, 8, 8), dim=1), offsets)
            output = torch.empty((bucket, 16, 128), device=device, dtype=torch.bfloat16)
            # Triton's graph benchmark reads the packed buffers on its own stream.
            torch.cuda.synchronize(device)
            for window in WINDOWS:
                _launch(tuned, packed, offsets, output, window, 128**-0.5)
                torch.cuda.synchronize(device)
                configs[window, bucket] = tuned.best_config
                logger.info(
                    "FP8 block startup on %s: window=%d bucket=%d config=%s",
                    device,
                    window,
                    bucket,
                    tuned.best_config,
                )
            del qkv, packed, offsets, output
    logger.info(
        "FP8 block startup on %s selected %d keys in %.2f seconds",
        device,
        len(configs),
        monotonic() - start,
    )
    return configs
