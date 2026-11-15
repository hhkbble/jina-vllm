from functools import cache, wraps
from time import monotonic
from typing import TYPE_CHECKING

import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .embeddings_v5_text_small import ARCH as TEXT_SMALL_ARCH
from .reranker_v35.contract import ARCH as RERANKER_ARCH

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.core import EngineCoreProc

logger = init_logger("vllm.jina_vllm.idle_trim")
OPTION = "additional_config.jina_vllm.idle_trim"


def idle_trim_ratio(additional_config) -> float:
    options = additional_config if isinstance(additional_config, dict) else {}
    jina = options.get("jina_vllm", {})
    if not isinstance(jina, dict):
        raise ValueError("additional_config.jina_vllm must be an object")
    ratio = jina.get("idle_trim", 0)
    if type(ratio) not in (float, int) or not 0 <= ratio < 1:
        raise ValueError(f"{OPTION} must be a number in [0, 1)")
    return float(ratio)


def validate_idle_trim_config(config: "VllmConfig") -> None:
    if not idle_trim_ratio(config.additional_config):
        return

    # Dense DP children reset their DP size to one before the core starts.
    parallel = config.parallel_config
    if (
        not current_platform.is_cuda()
        or not envs.VLLM_ENABLE_V1_MULTIPROCESSING
        or config.model_config.runner_type != "pooling"
        or (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
        != (1, 1, 1, 1, 1)
    ):
        raise ValueError(
            f"{OPTION} requires single-GPU CUDA pooling with EngineCoreProc"
        )

    offload = config.offload_config
    if (
        config.model_config.enable_sleep_mode
        or config.model_config.enable_cumem_allocator
        or offload.uva.cpu_offload_gb
        or offload.prefetch.offload_group_size
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
        or config.cache_config.kv_offloading_size is not None
    ):
        raise ValueError(
            f"{OPTION} requires pooling without sleep, CuMem, offload or transfers"
        )


def _allocator_bytes(device: torch.device) -> tuple[int, int]:
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    return (
        stats["reserved_bytes"]["all"]["current"],
        stats["allocated_bytes"]["all"]["current"],
    )


class _IdleTrim:
    def __init__(
        self, core: "EngineCoreProc", device: torch.device, threshold_bytes: int
    ) -> None:
        self.core = core
        self.device = device
        self.threshold_bytes = threshold_bytes
        self.pending = False

    def maybe_trim(self) -> None:
        core = self.core
        queue = core.input_queue
        if (
            not self.pending
            or not core.is_running()
            or core.has_work()
            or core.is_sleeping()
            or not queue.empty()
        ):
            return

        reserved, allocated = _allocator_bytes(self.device)
        if reserved - allocated <= self.threshold_bytes:
            self.pending = False
            return
        if not core.is_running() or not queue.empty():
            return

        start = monotonic()
        torch.cuda.synchronize(self.device)
        synchronized = monotonic()

        # Order the commitment against ADD and other messages using the same
        # mutex as Queue.put. GPU calls stay outside the critical section.
        with queue.mutex:
            if queue._qsize() or not core.is_running():
                return
            self.pending = False

        trim_start = monotonic()
        torch.cuda.empty_cache()
        trimmed = monotonic()
        reserved_after, allocated_after = _allocator_bytes(self.device)
        logger.info(
            "Idle trim on %s: threshold=%d bytes, reserved=%d -> %d bytes, "
            "allocated=%d -> %d bytes, released=%d bytes, sync=%.3f ms, trim=%.3f ms",
            self.device,
            self.threshold_bytes,
            reserved,
            reserved_after,
            allocated,
            allocated_after,
            reserved - reserved_after,
            (synchronized - start) * 1000,
            (trimmed - trim_start) * 1000,
        )


def _enable_idle_trim(core: "EngineCoreProc", ratio: float) -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    config = core.vllm_config
    parallel = config.parallel_config
    if (
        type(core.model_executor) is not UniProcExecutor
        or type(core.scheduler) is not Scheduler
        or core.batch_queue is not None
        or not config.use_v2_model_runner
        or parallel.worker_cls != "vllm.v1.worker.gpu_worker.Worker"
        or parallel.worker_extension_cls
    ):
        raise ValueError(f"{OPTION} requires the native synchronous CUDA V2 worker")

    kv_cache = core.scheduler.kv_cache_config
    if kv_cache.kv_cache_groups or kv_cache.kv_cache_tensors:
        raise ValueError(
            f"{OPTION} requires pool-free prefill. Set idle_trim to 0 for paged KV."
        )
    if torch.cuda.device_count() != 1:
        raise ValueError(f"{OPTION} requires exactly one visible CUDA device")
    if (
        torch.cuda.get_allocator_backend() != "native"
        or not torch._C._cuda_cudaCachingAllocator_is_enabled()
    ):
        raise ValueError(f"{OPTION} requires the PyTorch native CUDA caching allocator")

    # UniProc's worker runs in the core process on this thread.
    device = core.model_executor.driver_worker.device
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    trim = _IdleTrim(core, device, int(total_bytes * ratio))
    process_step = core._process_engine_step
    notify_idle = core._notify_idle_state_callbacks

    def process_step_with_trim() -> bool:
        executed = process_step()
        if executed:
            trim.pending = True
        return executed

    def notify_idle_with_trim() -> None:
        notify_idle()
        trim.maybe_trim()

    core._process_engine_step = process_step_with_trim
    core._notify_idle_state_callbacks = notify_idle_with_trim
    logger.info(
        "Idle trim enabled on %s: ratio=%g, total=%d bytes, threshold=%d bytes",
        device,
        ratio,
        total_bytes,
        trim.threshold_bytes,
    )


@cache
def install_idle_trim() -> None:
    from vllm.v1.engine.core import EngineCoreProc

    # vLLM 0.33.0 notifies idle callbacks before reading the ingress queue.
    # Keep the native loop and attach hooks only to opted-in cores.
    run_busy_loop = EngineCoreProc.run_busy_loop

    @wraps(run_busy_loop)
    def run_with_idle_trim(core: EngineCoreProc):
        config = core.vllm_config
        if config.model_config.architecture in (TEXT_SMALL_ARCH, RERANKER_ARCH):
            ratio = idle_trim_ratio(config.additional_config)
            if ratio:
                _enable_idle_trim(core, ratio)
        return run_busy_loop(core)

    EngineCoreProc.run_busy_loop = run_with_idle_trim
