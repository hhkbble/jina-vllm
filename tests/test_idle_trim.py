from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jina_vllm import idle_trim


def test_idle_trim_setting_is_opt_in_and_coexists_with_reranker_options():
    assert idle_trim.idle_trim_ratio({}) == 0
    assert idle_trim.idle_trim_ratio({"jina_vllm": {"idle_trim": 0}}) == 0
    options = {"jina_vllm": {"idle_trim": 0.1, "reranker_v35": {"block_size": 300}}}
    assert idle_trim.idle_trim_ratio(options) == 0.1


@pytest.mark.parametrize(
    "value", [True, None, "0.1", {}, -0.1, 1, 2, float("nan"), float("inf")]
)
def test_idle_trim_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=r"idle_trim must be a number in \[0, 1\)"):
        idle_trim.idle_trim_ratio({"jina_vllm": {"idle_trim": value}})


def test_disabled_skips_topology_checks_but_enabled_rejects_dense_dp(monkeypatch):
    monkeypatch.setattr(idle_trim.current_platform, "is_cuda", lambda: True)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    config = SimpleNamespace(
        additional_config={"jina_vllm": {"idle_trim": 0}},
        model_config=SimpleNamespace(runner_type="pooling"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    idle_trim.validate_idle_trim_config(config)
    config.additional_config["jina_vllm"]["idle_trim"] = 0.1
    with pytest.raises(ValueError, match="requires single-GPU CUDA pooling"):
        idle_trim.validate_idle_trim_config(config)


def test_idle_trim_rejects_cumem_without_sleep(monkeypatch):
    monkeypatch.setattr(idle_trim.current_platform, "is_cuda", lambda: True)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    config = SimpleNamespace(
        additional_config={"jina_vllm": {"idle_trim": 0.1}},
        model_config=SimpleNamespace(
            runner_type="pooling", enable_sleep_mode=False, enable_cumem_allocator=True
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        offload_config=None,
    )
    with pytest.raises(ValueError, match="CuMem"):
        idle_trim.validate_idle_trim_config(config)


@pytest.fixture
def candidate(monkeypatch):
    core = SimpleNamespace(
        input_queue=Queue(),
        is_running=Mock(return_value=True),
        has_work=Mock(return_value=False),
        is_sleeping=Mock(return_value=False),
    )
    stats = Mock(return_value=(200, 50))
    synchronize = Mock()
    empty_cache = Mock()
    monkeypatch.setattr(idle_trim, "_allocator_bytes", stats)
    monkeypatch.setattr(idle_trim.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(idle_trim.torch.cuda, "empty_cache", empty_cache)
    return idle_trim._IdleTrim(core, "cuda:0", 100), stats, synchronize, empty_cache


def test_trim_requires_pending_work_and_a_strictly_exceeded_threshold(candidate):
    trim, stats, synchronize, empty_cache = candidate
    trim.maybe_trim()
    stats.assert_not_called()

    trim.pending = True
    stats.return_value = 150, 50
    trim.maybe_trim()
    assert not trim.pending
    synchronize.assert_not_called()
    empty_cache.assert_not_called()

    trim.pending = True
    stats.return_value = 151, 50
    trim.maybe_trim()
    trim.maybe_trim()
    synchronize.assert_called_once_with("cuda:0")
    empty_cache.assert_called_once_with()


@pytest.mark.parametrize("blocking", ["ingress", "scheduler", "pause", "shutdown"])
def test_active_core_skips_cuda_work(candidate, blocking):
    trim, stats, synchronize, empty_cache = candidate
    trim.pending = True
    if blocking == "ingress":
        trim.core.input_queue.put_nowait("ADD")
    elif blocking == "scheduler":
        trim.core.has_work.return_value = True
    elif blocking == "pause":
        trim.core.is_sleeping.return_value = True
    else:
        trim.core.is_running.return_value = False
    trim.maybe_trim()
    assert trim.pending
    stats.assert_not_called()
    synchronize.assert_not_called()
    empty_cache.assert_not_called()


@pytest.mark.parametrize("before_commit", [True, False])
def test_add_orders_against_trim_commit(candidate, before_commit):
    trim, _, synchronize, empty_cache = candidate
    trim.pending = True
    entered, release = Event(), Event()

    def barrier(*_):
        entered.set()
        assert release.wait(timeout=5)

    operation = synchronize if before_commit else empty_cache
    operation.side_effect = barrier
    with ThreadPoolExecutor(max_workers=2) as executor:
        trimming = executor.submit(trim.maybe_trim)
        try:
            assert entered.wait(timeout=5)
            queue = trim.core.input_queue
            executor.submit(queue.put_nowait, "ADD").result(timeout=2)
        finally:
            release.set()
        trimming.result(timeout=5)

    assert queue.get_nowait() == "ADD"
    assert trim.pending is before_commit
    assert empty_cache.call_count == int(not before_commit)
