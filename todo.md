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
