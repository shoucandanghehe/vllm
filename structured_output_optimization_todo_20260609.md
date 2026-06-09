# 2026-06-09 Optimization TODO

## Confirmed optimization: structured output grammar bitmask

- Hotspot: `vllm/v1/structured_output/utils.py::apply_grammar_bitmask`.
- Trace evidence: structured-output decode steps spend p50 about 34 ms in grammar mask application; most of that is H2D copy of a full `(batch_rows, bitmask_cols)` int32 mask even when only 1-3 requests are structured.
- Implemented direction: keep a reusable full-batch GPU mask buffer, fill it on GPU, copy only compact structured rows from CPU, and scatter them to the original logits rows before calling xgrammar.
- Math invariant: the bitmask visible to `xgrammar.apply_token_bitmask_inplace` is equivalent to the old full-batch CPU-expanded bitmask.
- Current verification: targeted structured-output tests pass locally, including compact-row mapping and spec-decode row offsets.

## Next exploration path 1: decode-only scheduler host path

- Trace target: latest decode-only no-grammar steady-state lines show `schedule_ms` about 6.9-7.2 ms while `sampler_call_ms` for full 72-row steps is about 2.8-3.0 ms.
- Code target: `vllm/v1/core/sched/scheduler.py::schedule`.
- Current evidence:
  - RUNNING decode still iterates every running request and calls `kv_cache_manager.allocate_slots(...)` per request.
  - The scheduler always computes `num_common_prefix_blocks` for cascade attention when `self.running` is non-empty.
  - `_update_after_schedule()` loops scheduled requests again and mutates structured-output/request state.
- Investigation tasks:
  - Split scheduler timing into RUNNING loop, WAITING loop, common-prefix computation, cached request output construction, connector metadata, and `_update_after_schedule`.
  - Check whether steady pure decode can bypass WAITING work, LoRA set construction, common-prefix work, or repeated KV allocation checks when no block boundary is crossed.
  - Preserve scheduler semantics: preemption, KV block allocation, async placeholders, structured-output detection, and mixed prefill/decode behavior must still fall back to the general path.

## Next exploration path 2: decode input-preparation / attention-metadata host path

- Trace target: latest 72-request decode-only no-grammar lines show `preprocess` about 10.9-11.3 ms and `forward` about 0.5 ms once CUDA graph is warm.
- Code target: `vllm/v1/worker/gpu_model_runner.py::execute_model`, especially `_update_states()`, `_prepare_inputs()`, `_get_slot_mappings()`, and `_build_attention_metadata()`.
- Current evidence:
  - Pure local mocks for `np.repeat`/cumsum/token gather, full block-table commit, and slot-mapping kernel are sub-ms on RTX 4060, so the remote 11 ms is unlikely to be one isolated primitive.
  - The full legacy path still rebuilds request arrays, query metadata, block table state, slot mappings, attention metadata, and FlashInfer planning objects every decode step.
  - `sample_hidden_states = hidden_states[logits_indices]` is an advanced-index gather even when pure decode logits rows are already contiguous.
- Investigation tasks:
  - Split preprocess timing into `_update_states`, `_prepare_inputs`, block-table commit, slot mapping, CUDA graph dispatch, attention metadata build, FlashInfer plan, and `_preprocess`.
  - Look for a narrowly gated pure-decode fast path that reuses stable metadata and falls back on any batch reorder, new/finished request, encoder input, spec decode, M-RoPE/XD-RoPE, DCP/DBO, or block-boundary allocation.
  - Verify whether an identity-gather skip for pure decode logits is safe and measurable.

## Deprioritized path: no-grammar sampler

- Earlier aggregate suggested `sampler_call_ms` p50 around 16 ms, but direct latest trace lines show steady 72-row no-grammar steps at about 2.8-3.0 ms.
- There are still sampler outliers, e.g. 10-85 ms at changing batch shapes, but `has_logprobs=False` and repair is small; current instrumentation times Python wall time without CUDA events, so queued GPU work or allocator/JIT effects can be charged to sampler.
- Keep as a secondary path unless CUDA-event sampler timing or repeated outliers prove it is steady-state decode cost.

## Non-goals for now

- Do not treat long-prefill mixed decode as a bug by default; it is a throughput-oriented scheduler design.
- Do not restart or hotpatch the production/business service without explicit approval.
- Do not open or update PR/issue text before user review.
