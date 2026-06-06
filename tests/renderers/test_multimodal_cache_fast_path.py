# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.renderers.base import BaseRenderer
from vllm.utils.async_utils import make_async
from vllm.utils.counter import AtomicCounter


class _FakeInfo:
    @staticmethod
    def parse_mm_data(mm_data: dict[str, Any]):
        return mm_data


class _FakeProcessor:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.info = _FakeInfo()
        self.started = started
        self.release = release

    def try_apply_cached(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        if inputs.mm_data_items["kind"] != "hit":
            return None

        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


    def get_cache_missing_hashes(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        return None

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        self.started.set()
        assert self.release.wait(timeout=5)
        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


class _FakeTimingRegistry:
    @staticmethod
    def get(request_id: str):
        return TimingContext(enabled=False)


class _TestRenderer(BaseRenderer):
    def render_messages(self, messages, params):
        raise NotImplementedError


def _make_renderer(processor: _FakeProcessor):
    renderer: Any = object.__new__(_TestRenderer)
    renderer.api_process_rank = 0
    renderer._mm_req_counter = AtomicCounter()
    renderer._readonly_mm_processor = None
    renderer.mm_processor = processor
    renderer._mm_timing_registry = _FakeTimingRegistry()
    renderer._process_mm_uuids = lambda *args, **kwargs: None
    renderer.update_mm_cache_stats = lambda: None
    renderer.config = type(
        "Config",
        (),
        {
            "cache_config": type(
                "CacheConfig",
                (),
                {"enable_prefix_caching": True},
            )()
        },
    )()
    renderer.model_config = type(
        "ModelConfig",
        (),
        {"multimodal_config": None},
    )()
    renderer.get_mm_processor = lambda: processor
    renderer._mm_cache_inflight_futures = {}
    executor = ThreadPoolExecutor(max_workers=1)
    renderer._process_multimodal_in_executor = make_async(
        renderer._process_multimodal,
        executor=executor,
    )
    renderer._test_executor = executor
    return renderer


@pytest.mark.asyncio
async def test_cached_multimodal_request_bypasses_busy_executor():
    started = threading.Event()
    release = threading.Event()
    renderer = _make_renderer(_FakeProcessor(started, release))

    miss_task = asyncio.create_task(
        renderer._process_multimodal_async(
            [1],
            {"kind": "miss"},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
        )
    )

    try:
        assert await asyncio.to_thread(started.wait, 5)

        hit_result = await asyncio.wait_for(
            renderer._process_multimodal_async(
                [2],
                {"kind": "hit"},
                mm_uuids=None,
                mm_processor_kwargs=None,
                tokenization_kwargs=None,
            ),
            timeout=0.5,
        )
        assert hit_result["prompt_token_ids"] == [2]
    finally:
        release.set()
        await miss_task
        renderer._test_executor.shutdown(wait=True)
