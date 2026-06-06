#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(dirname "$(readlink -f "$0")")
cd "$ROOT_DIR"

if [[ -n "${VLLM_BENCH_PYTHON:-}" ]]; then
  PYTHON="$VLLM_BENCH_PYTHON"
else
  UV_TOOL_DIR=$(uv tool dir)
  TOOL_PYTHON="$UV_TOOL_DIR/vllm/bin/python"
  if [[ -x "$TOOL_PYTHON" ]]; then
    PYTHON="$TOOL_PYTHON"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    echo "No benchmark Python found. Set VLLM_BENCH_PYTHON or install vLLM via uv tool." >&2
    exit 1
  fi
fi

PYTHONHASHSEED=0 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" benchmarks/overheads/benchmark_multimodal_preprocessing_scheduler.py \
  --concurrency "${VLLM_MM_SCHED_CONCURRENCY:-72}" \
  --renderer-workers "${VLLM_MM_SCHED_RENDERER_WORKERS:-4}" \
  --miss-cost "${VLLM_MM_SCHED_MISS_COST:-0.02}" \
  --hot-images "${VLLM_MM_SCHED_HOT_IMAGES:-256}" \
  --unique-new-images "${VLLM_MM_SCHED_UNIQUE_NEW_IMAGES:-12}" \
  --repeats-per-new-image "${VLLM_MM_SCHED_REPEATS_PER_NEW_IMAGE:-6}" \
  --cached-hol-requests "${VLLM_MM_SCHED_CACHED_HOL_REQUESTS:-71}" \
  --mixed-requests "${VLLM_MM_SCHED_MIXED_REQUESTS:-71}" \
  --repetitions "${VLLM_MM_SCHED_REPETITIONS:-5}"
