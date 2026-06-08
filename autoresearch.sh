#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(dirname "$(readlink -f "$0")")
cd "$ROOT_DIR"

LOCK_PATH="${VLLM_TEST_LOCK:-/home/scdhh/code/vllm/test-lock}"
exec 9>"$LOCK_PATH"
flock 9


if [[ -n "${VLLM_BENCH_PYTHON:-}" ]]; then
  PYTHON="$VLLM_BENCH_PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x "$ROOT_DIR/../.."/.venv/bin/python ]]; then
  PYTHON="$ROOT_DIR/../.."/.venv/bin/python
else
  echo "No benchmark Python found. Set VLLM_BENCH_PYTHON or create a vLLM .venv." >&2
  exit 1
fi

PYTHONHASHSEED=0 \
TOKENIZERS_PARALLELISM=false \
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" benchmarks/overheads/benchmark_v1_sampler_greedy.py \
  --batch-size "${VLLM_SAMPLER_BATCH_SIZE:-72}" \
  --vocab-size "${VLLM_SAMPLER_VOCAB_SIZE:-152064}" \
  --warmups "${VLLM_SAMPLER_WARMUPS:-20}" \
  --iterations "${VLLM_SAMPLER_ITERATIONS:-100}" \
  --repetitions "${VLLM_SAMPLER_REPETITIONS:-5}" \
  --dtype "${VLLM_SAMPLER_DTYPE:-float16}" \
  --mode "${VLLM_SAMPLER_MODE:-qwen-default}"
