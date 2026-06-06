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
OMP_NUM_THREADS="${VLLM_HFPROC_OMP_THREADS:-8}" \
MKL_NUM_THREADS="${VLLM_HFPROC_MKL_THREADS:-8}" \
OPENBLAS_NUM_THREADS="${VLLM_HFPROC_OPENBLAS_THREADS:-8}" \
RAYON_NUM_THREADS="${VLLM_HFPROC_RAYON_THREADS:-8}" \
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" benchmarks/overheads/benchmark_qwen3_hf_processor.py \
  --requests "${VLLM_HFPROC_REQUESTS:-12}" \
  --images-per-request "${VLLM_HFPROC_IMAGES_PER_REQUEST:-4}" \
  --width "${VLLM_HFPROC_IMAGE_WIDTH:-1786}" \
  --height "${VLLM_HFPROC_IMAGE_HEIGHT:-2526}" \
  --text-tokens "${VLLM_HFPROC_TEXT_TOKENS:-1024}" \
  --workers "${VLLM_HFPROC_WORKERS:-4}" \
  --warmups "${VLLM_HFPROC_WARMUPS:-1}" \
  --repetitions "${VLLM_HFPROC_REPETITIONS:-2}"
