# SPDX-License-Identifier: Apache-2.0
"""Regression test for LMCache#3318.

The vLLM v1 adapter previously asserted
``len(slot_mapping) == len(token_ids)`` inside ``wait_for_save``. When the
state desynced (e.g. upstream allocation failure or preemption-induced
mismatch) the assertion fired as an unhandled ``AssertionError`` and
killed the entire EngineCore process for every connected user.

The fix replaces the assert with a logged ``continue`` so the engine
stays alive and only the affected request's save is dropped. This test
locks in that behavior by feeding ``wait_for_save`` a request whose
``slot_mapping`` and ``token_ids`` lengths disagree and asserting:

1. ``wait_for_save`` does not raise.
2. A warning is emitted naming the request id and both lengths.
3. ``lmcache_engine.store`` is not called for the desynced request
   (the save is dropped, not silently corrupted).
4. ``lookup_unpin`` is still called so the pin count stays balanced.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import logging

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request

# First Party
from lmcache.inprocess.integration.vllm import vllm_v1_adapter as adapter
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
)
from lmcache.inprocess.v1.config import LMCacheEngineConfig


@pytest.mark.parametrize("resolved_block_size", [16, 32, 64])
def test_scheduler_slots_use_resolved_kv_group_block_size(
    monkeypatch: pytest.MonkeyPatch, resolved_block_size: int
) -> None:
    """Allocator block IDs must map to physical KV slots, not CLI block units."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    monkeypatch.setattr(adapter, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(adapter, "VllmServiceFactory", MagicMock())
    manager = MagicMock(lmcache_engine=None, lmcache_engine_metadata=None)
    monkeypatch.setattr(adapter, "LMCacheManager", MagicMock(return_value=manager))
    monkeypatch.setattr(adapter.LMCStatsMonitor, "GetOrCreate", MagicMock())
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(get_num_layers=lambda _: 1),
        num_hash_lookahead_tokens=1,
    )
    spec = FullAttentionSpec(
        block_size=resolved_block_size,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
    )
    connector = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        MagicMock(),
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]),
    )
    request = Request(
        "resolved-slots",
        list(range(resolved_block_size + 1)),
        SamplingParams(max_tokens=1),
        None,
    )
    connector.update_state_after_alloc(request, 0)
    metadata = connector.build_connector_meta(
        SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[NewRequestData.from_request(request, ([2],))],
            num_scheduled_tokens={request.request_id: resolved_block_size},
            scheduled_cached_reqs=[],
        )
    )
    assert len(metadata.requests) == 1
    assert metadata.requests[0].slot_mapping.tolist() == list(
        range(2 * resolved_block_size, 3 * resolved_block_size)
    )
    assert metadata.requests[0].lookahead_token_ids == [resolved_block_size]


@pytest.mark.parametrize("lookahead", [1, 3, 17])
def test_hash_context_does_not_allocate_or_save_extra_kv(lookahead: int) -> None:
    tracker = RequestTracker(
        req_id="lookahead",
        prompt_len=96,
        token_ids=list(range(32)),
        allocated_block_ids=[2, 3],
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        16,
        16,
        num_hash_lookahead_tokens=lookahead,
        hash_context=list(range(32 + lookahead)),
    )
    assert meta is not None
    assert meta.token_ids == list(range(32))
    assert meta.lookahead_token_ids == list(range(32, 32 + lookahead))
    assert meta.slot_mapping.tolist() == list(range(32, 64))
    assert tracker.num_saved_tokens == 32


def test_final_prompt_chunk_waits_for_hash_context_without_decode_saves() -> None:
    tracker = RequestTracker(
        req_id="lookahead",
        prompt_len=32,
        token_ids=list(range(32)),
        allocated_block_ids=[2, 3],
    )
    first = ReqMeta.from_request_tracker(
        tracker,
        16,
        16,
        num_hash_lookahead_tokens=1,
        hash_context=list(range(32)),
    )
    assert first is not None
    assert len(first.token_ids) == tracker.num_saved_tokens == 16
    tracker.is_decode_phase = True
    second = ReqMeta.from_request_tracker(
        tracker,
        16,
        16,
        num_hash_lookahead_tokens=1,
        hash_context=list(range(33)),
    )
    assert second is not None
    assert len(second.token_ids) == tracker.num_saved_tokens == 32
    assert second.save_spec == SaveSpec(skip_leading_tokens=16, can_save=True)
    assert second.lookahead_token_ids == [32]


def test_decode_load_keeps_hash_context_when_decode_save_is_disabled() -> None:
    tracker = RequestTracker(
        req_id="load",
        prompt_len=16,
        token_ids=list(range(48)),
        allocated_block_ids=[2, 3, 4],
        num_saved_tokens=48,
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        16,
        16,
        load_spec=LoadSpec(0, 48, True),
        num_hash_lookahead_tokens=1,
        hash_context=list(range(49)),
    )
    assert meta is not None
    assert meta.save_spec is not None and not meta.save_spec.can_save
    assert len(meta.token_ids) == len(meta.slot_mapping) == 48
    assert meta.lookahead_token_ids == [48]
    assert tracker.num_saved_tokens == 48


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


class _FakeEngine:
    """Records calls to ``lookup_unpin`` and ``store`` so the test can
    assert which paths fired."""

    def __init__(self) -> None:
        self.unpinned: list[str] = []
        self.store_calls: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store(self, *args, **kwargs) -> None:
        self.store_calls.append(kwargs.get("req_id", "<unknown>"))


def _make_desync_request(
    req_id: str, token_ids_len: int, slot_mapping_len: int
) -> SimpleNamespace:
    """Build a request whose ``token_ids`` and ``slot_mapping`` lengths
    disagree, simulating a state desync."""
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(token_ids_len)),
        slot_mapping=torch.arange(slot_mapping_len, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        disagg_spec=None,
        is_last_prefill=True,
        request_configs=None,
    )


def _make_connector(
    requests: list[SimpleNamespace],
) -> tuple[LMCacheConnectorV1Impl, _FakeEngine]:
    metadata = LMCacheConnectorMetadata(requests=requests)  # type: ignore[arg-type]
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    # ``lmcache_engine`` is a read-only property backed by ``self._manager``;
    # inject the fake engine through the manager so the property resolves to it.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine
    )
    connector.kv_role = "kv_producer"
    connector.use_layerwise = False
    connector.enable_blending = False
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.config = SimpleNamespace(pd_bidirectional=False)
    return connector, engine


def test_wait_for_save_skips_desynced_request_and_keeps_engine_alive() -> None:
    """Length mismatch must drop only the affected request's save, log a
    warning, and let ``wait_for_save`` return normally.

    Regression for https://github.com/LMCache/LMCache/issues/3318.
    """
    # lmcache's ``init_logger`` sets ``propagate = False`` on the adapter
    # logger so its records do not reach pytest's ``caplog`` (which
    # attaches to the root logger). Toggling ``propagate`` is fragile --
    # any lazy import that re-runs ``init_logger`` resets it. Attach a
    # local handler directly to the named logger instead so we capture
    # the warning regardless of how lmcache configures propagation.
    captured_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    adapter_logger = logging.getLogger(
        "lmcache.inprocess.integration.vllm.vllm_v1_adapter"
    )
    # ``init_logger`` sets the logger level from ``LMCACHE_LOG_LEVEL`` (default
    # INFO). If a prior import set it above WARNING, ``logger.warning`` would be
    # filtered before reaching our handler. Force WARNING for the duration of
    # the test and restore the original level in ``finally``.
    original_level = adapter_logger.level
    adapter_logger.setLevel(logging.WARNING)
    adapter_logger.addHandler(handler)
    try:
        desync_req = _make_desync_request(
            "req-desync", token_ids_len=4, slot_mapping_len=3
        )
        connector, engine = _make_connector([desync_req])

        connector.wait_for_save()

        # 1. lookup_unpin still ran (pin balance preserved)
        assert engine.unpinned == ["req-desync"]

        # 2. store was NOT called for the desynced request (save dropped)
        assert engine.store_calls == []

        # 3. A warning was emitted naming the request and both lengths
        warnings = [r for r in captured_records if r.levelno == logging.WARNING]
        assert any(
            "req-desync" in r.getMessage()
            and "slot_mapping=3" in r.getMessage()
            and "token_ids=4" in r.getMessage()
            for r in warnings
        ), (
            "Expected desync warning naming req-desync; "
            f"got {[r.getMessage() for r in warnings]}"
        )
    finally:
        adapter_logger.removeHandler(handler)
        adapter_logger.setLevel(original_level)
