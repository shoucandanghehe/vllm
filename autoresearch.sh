#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(dirname "$(readlink -f "$0")")
WORKTREE_DIR="${VLLM_HFPROC_WORKTREE:-$ROOT_DIR/.worktrees/hf-processor-opt}"
cd "$WORKTREE_DIR"

if [[ -n "${VLLM_BENCH_PYTHON:-}" ]]; then
  PYTHON="$VLLM_BENCH_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  UV_TOOL_DIR=$(uv tool dir)
  TOOL_PYTHON="$UV_TOOL_DIR/vllm/bin/python"
  if [[ -x "$TOOL_PYTHON" ]]; then
    PYTHON="$TOOL_PYTHON"
  else
    echo "No benchmark Python found. Set VLLM_BENCH_PYTHON or install vLLM via uv tool." >&2
    exit 1
  fi
fi

PYTHONHASHSEED=0 \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS="${VLLM_HFPROC_OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${VLLM_HFPROC_MKL_NUM_THREADS:-1}" \
OPENBLAS_NUM_THREADS="${VLLM_HFPROC_OPENBLAS_NUM_THREADS:-1}" \
PYTHONPATH="$WORKTREE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" benchmarks/overheads/benchmark_qwen3_hf_processor.py \
  --requests "${VLLM_HFPROC_REQUESTS:-12}" \
  --images-per-request "${VLLM_HFPROC_IMAGES_PER_REQUEST:-4}" \
  --width "${VLLM_HFPROC_IMAGE_WIDTH:-1786}" \
  --height "${VLLM_HFPROC_IMAGE_HEIGHT:-2526}" \
  --text-tokens "${VLLM_HFPROC_TEXT_TOKENS:-1024}" \
  --workers "${VLLM_HFPROC_WORKERS:-4}" \
  --warmups "${VLLM_HFPROC_WARMUPS:-1}" \
  --repetitions "${VLLM_HFPROC_REPETITIONS:-2}"
