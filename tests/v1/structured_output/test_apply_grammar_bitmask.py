# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from vllm.v1.structured_output.utils import apply_grammar_bitmask


def _make_bitmask(rows: int, vocab_size: int) -> np.ndarray:
    cols = (vocab_size + 31) // 32
    bitmask = np.full((rows, cols), -1, dtype=np.int32)
    if rows > 0:
        bitmask[0, :] = 0
    if rows > 1:
        bitmask[1, :] = 0
        bitmask[1, 0] = 1 << 3
    if rows > 2:
        bitmask[2, :] = 0
        bitmask[2, 1] = 1 << 5
    return bitmask


def _reference_apply(
    logits: torch.Tensor,
    raw_bitmask: np.ndarray,
    out_indices: list[int],
) -> torch.Tensor:
    bitmask = np.full(
        (logits.shape[0], raw_bitmask.shape[1]), -1, dtype=raw_bitmask.dtype
    )
    for raw_row, logit_row in enumerate(out_indices):
        bitmask[logit_row] = raw_bitmask[raw_row]

    expected = logits.clone()
    bitmask_tensor = torch.from_numpy(bitmask).to(expected.device)
    import xgrammar as xgr

    xgr.apply_token_bitmask_inplace(expected, bitmask_tensor, indices=out_indices)
    return expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_apply_grammar_bitmask_maps_compact_rows(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    vocab_size = 64
    logits = torch.arange(5 * vocab_size, dtype=torch.float32, device=device).reshape(
        5, vocab_size
    )
    original = logits.clone()
    raw_bitmask = _make_bitmask(3, vocab_size)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"structured": [11, 12]}
    )
    grammar_output = SimpleNamespace(
        grammar_bitmask=raw_bitmask,
        structured_output_request_ids=["structured"],
    )
    input_batch = SimpleNamespace(req_ids=["plain_before", "structured", "plain_after"])

    apply_grammar_bitmask(
        cast(Any, scheduler_output),
        cast(Any, grammar_output),
        cast(Any, input_batch),
        logits,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    expected = _reference_apply(original, raw_bitmask, [1, 2, 3])
    assert torch.equal(logits, expected)
    assert torch.equal(logits[0], original[0])
    assert torch.equal(logits[4], original[4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_apply_grammar_bitmask_reuses_gpu_buffers() -> None:
    vocab_size = 64
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={})
    grammar_output = SimpleNamespace(
        grammar_bitmask=_make_bitmask(1, vocab_size),
        structured_output_request_ids=["structured"],
    )
    input_batch = SimpleNamespace(req_ids=["plain", "structured"])

    logits = torch.zeros((2, vocab_size), dtype=torch.float32, device="cuda")
    apply_grammar_bitmask(
        cast(Any, scheduler_output),
        cast(Any, grammar_output),
        cast(Any, input_batch),
        logits,
    )
    first_full, first_compact = cast(
        tuple[torch.Tensor, torch.Tensor],
        input_batch._grammar_bitmask_gpu_buffers,
    )

    logits = torch.zeros((2, vocab_size), dtype=torch.float32, device="cuda")
    apply_grammar_bitmask(
        cast(Any, scheduler_output),
        cast(Any, grammar_output),
        cast(Any, input_batch),
        logits,
    )
    second_full, second_compact = cast(
        tuple[torch.Tensor, torch.Tensor],
        input_batch._grammar_bitmask_gpu_buffers,
    )

    assert first_full.data_ptr() == second_full.data_ptr()
    assert first_compact.data_ptr() == second_compact.data_ptr()
