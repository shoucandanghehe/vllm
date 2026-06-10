# Structured-output grammar bitmask PR TODO

## PR scope

- Target optimization: `vllm/v1/structured_output/utils.py::apply_grammar_bitmask` aligned full-batch CUDA fast path.
- Include only grammar bitmask changes and their regression tests.
- Exclude separate reasoning-gate optimizations:
  - `perf: pass decode deltas to reasoning gate`
  - `perf: avoid qwen3 reasoning delta rescans`

## Evidence from 8470Q trace

Before:

- Trace: `/root/vllm/business_trace/8470q-yappi-20260611-001236`
- Stable decode-only filter: `Running=127/128`, `Waiting=0`, `decode_only=True`, `all_one_token=True`, `struct_req_ids=127/128`.
- `grammar.total_ms p50 = 24.993 ms`
- `grammar.h2d_ms p50 = 24.080 ms`
- Generation throughput p50: `2107 tok/s`

After:

- Trace: `/root/vllm/business_trace/8470q-yappi-grammaropt-20260611-004744`
- Same filter.
- `grammar.total_ms p50 = 1.026 ms`
- `grammar.h2d_ms p50 = 0.267 ms`
- Generation throughput p50: `3002 tok/s`

Observed improvement:

- Grammar bitmask p50: `24.993 ms -> 1.026 ms`.
- H2D/fill/copy path p50: `24.080 ms -> 0.267 ms`.
- Generation throughput p50: `2107 tok/s -> 3002 tok/s`.

## Implementation summary

- Reuse a pinned CPU staging tensor for compact grammar masks when pin memory is available.
- Detect aligned full-batch structured-output decode:
  - `out_indices == [0..N)`
  - `bitmask_row_indices == [0..N)`
- In that case pass the compact GPU bitmask directly to xgrammar with `indices=None`.
- Skip full-batch GPU mask fill and `index_copy_` on the aligned path.
- Preserve existing full-buffer scatter path for partial or reordered batches.

## Verification already run locally

```bash
uv run --with ruff ruff check \
  vllm/v1/structured_output/utils.py \
  tests/v1/structured_output/test_apply_grammar_bitmask.py
```

```bash
.venv/bin/python -m pytest tests/v1/structured_output/test_apply_grammar_bitmask.py -q
# 5 passed
```

```bash
.venv/bin/python -m pytest \
  tests/v1/structured_output/test_backend_guidance.py::test_grammar_bitmask_with_specdec \
  tests/v1/structured_output/test_reasoning_structured_output.py -q
# 11 passed
```

## PR branch plan

- Create an isolated branch for this optimization only.
- Base it on the current fork autoresearch baseline if keeping prior compact-mask infrastructure as dependency.
- If targeting upstream `main`, include the earlier compact grammar-mask infrastructure commit as a prerequisite or fold it into this PR.
- PR body should mention AI assistance, duplicate-work checks, and the exact trace/test evidence above.

## Follow-up optimization candidates after grammar fix

Trace source:

- `/root/vllm/business_trace/8470q-yappi-grammaropt-20260611-004744`
- Stable decode-only filter: `Running=127/128`, `Waiting=0`, `decode_only=True`, `all_one_token=True`, `encoder_reqs=0`.

Current stable p50 after grammar optimization:

- `scheduler.total_ms = 11.342 ms`
- `scheduler.running_loop_ms = 9.996 ms`
- `runner.total_ms = 5.232 ms`
- `runner.prepare_inputs_ms = 2.321 ms`
- `runner.attn_metadata_ms = 1.213 ms`
- `grammar.total_ms = 1.026 ms`

Yappi 30s windows show these remaining source hotspots:

- `AsyncScheduler.schedule`: about `10.9-11.8 ms/step` in 127/128 decode windows.
- `KVCacheManager.allocate_slots`: about `5.8-7.0 s/window`, around `87k-108k calls/window`, about one call per running request per step.
- `GPUModelRunner.execute_model`: about `8.8-15.5 ms/step`, including CPU prep and GPU wait.
- `AsyncOutputFuture.result`: about `8-10 ms/step` in the same windows, mostly pipeline/future wait rather than local CPU work.
- `GPUModelRunner._prepare_inputs`: about `2.2-2.8 ms/step`.
- `GPUModelRunner._build_attention_metadata`: about `1.2-1.4 ms/step`.
- `AsyncScheduler.update_from_output`: about `3.7-4.1 ms/step`.

Priority 1: scheduler KV/Mamba decode fast path

- Source chain:
  - `vllm/v1/core/sched/scheduler.py::Scheduler.schedule`
  - `vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots`
  - `vllm/v1/core/kv_cache_coordinator.py`
  - `vllm/v1/core/single_type_kv_cache_manager.py::MambaManager`
- Current behavior: even pure one-token decode with no waiting and no encoder input calls `allocate_slots()` once per running request, then every call walks manager-level `remove_skipped_blocks`, `get_num_blocks_to_allocate`, `allocate_new_blocks`, and `cache_blocks`.
- Safe starting sub-candidates:
  - Skip WAITING queue setup when `self.waiting` and `self.skipped_waiting` are both empty.
  - Avoid `_mamba_block_aligned_split()` for normal decode where `request.num_computed_tokens >= max(request.num_prompt_tokens, request.num_tokens - 1)` and there are no external/prefix-computed tokens.
  - Return `None` for `new_block_ids` directly in `_make_cached_request_data()` when `req_to_new_blocks[req_id]` is the known empty cache-block object.
- Higher-risk candidate:
  - A no-new-block `allocate_slots()` fast return for one-token decode when full-attention and Mamba block counts are already satisfied.
  - Must preserve full-attention allocation at block boundaries, Mamba align current-state allocation/copy/free behavior, prefix-cache block events, and spec decode lookahead.

Priority 2: runner pure-decode input-prep fast path

- Source chain:
  - `vllm/v1/worker/gpu_model_runner.py::_prepare_inputs`
  - `vllm/v1/worker/block_table.py::commit_block_table`
  - `vllm/v1/worker/block_table.py::compute_slot_mapping`
- Current behavior: one-token decode still performs `np.repeat`, cumsum/arange construction, CPU token index gather, full active block-table copy, full query-start-loc copy, request discard-mask list construction, and slot-mapping padding.
- Safer sub-candidates:
  - Contiguous logits path: when no spec decode and all scheduled rows are one-token decode, use `hidden_states[:num_reqs]` instead of `hidden_states[logits_indices]`.
  - Skip `update_req_spec_token_ids` in `_update_states()` when `self.num_spec_tokens == 0` and no request has scheduled spec tokens.
  - Avoid reorder helper when all active rows are already done-prefill one-token decode.
- Higher-risk candidate:
  - Dirty-row block-table tracking to avoid full block-table H2D copy on non-boundary decode steps.
  - Must mark dirty on row add/remove/move/swap/append and every KV block-boundary allocation.

Priority 3: output/update and IPC cleanup

- Source chain:
  - `vllm/v1/core/sched/scheduler.py::AsyncScheduler.update_from_output`
  - `vllm/v1/request.py::Request.append_output_token_ids`
  - `vllm/v1/engine/core.py::process_output_sockets`
- Lower-risk sub-candidates:
  - Skip `structured_output_manager.should_advance()` call for requests where `request.use_structured_output` is false. This does not help the current all-structured workload but helps mixed/plain workloads.
  - Avoid block-hash closure calls until appending tokens can complete a full hash block. Preserve parent-hash/MM/LoRA/cache-salt semantics.
- Bigger protocol candidate:
  - Columnar/internal layout for `EngineCoreOutputs` to reduce per-request msgpack/object overhead. This is not a local fast path and should be deferred.
