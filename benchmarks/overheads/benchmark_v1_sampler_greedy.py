# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the V1 greedy sampler path for decode-sized batches."""

import argparse
import statistics
import time

import torch

from vllm.config import VllmConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


def make_metadata(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> SamplingMetadata:
    logitsprocs = build_logitsprocs(
        vllm_config=VllmConfig(),
        device=device,
        is_pin_memory=is_pin_memory_available(),
        is_pooling_model=False,
    )
    return SamplingMetadata(
        temperature=torch.zeros((batch_size,), dtype=torch.float32, device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(
            (batch_size,), dtype=torch.float32, device=device
        ),
        presence_penalties=torch.zeros(
            (batch_size,), dtype=torch.float32, device=device
        ),
        repetition_penalties=torch.ones(
            (batch_size,), dtype=torch.float32, device=device
        ),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs,
    )


def make_logits(
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    logits = torch.randn((batch_size, vocab_size), dtype=dtype, device=device)
    row_ids = torch.arange(batch_size, device=device)
    winners = (row_ids * 9973 + 17) % vocab_size
    logits[row_ids, winners] = 100.0
    return logits


def time_sampler_cuda(
    sampler: Sampler,
    logits: torch.Tensor,
    metadata: SamplingMetadata,
    iterations: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        output = sampler(logits, metadata)
    end.record()
    torch.cuda.synchronize()
    if output.sampled_token_ids.shape[0] != logits.shape[0]:
        raise AssertionError("Sampler returned the wrong batch size")
    return start.elapsed_time(end) / iterations


def time_sampler_cpu(
    sampler: Sampler,
    logits: torch.Tensor,
    metadata: SamplingMetadata,
    iterations: int,
) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        output = sampler(logits, metadata)
    elapsed_s = time.perf_counter() - start
    if output.sampled_token_ids.shape[0] != logits.shape[0]:
        raise AssertionError("Sampler returned the wrong batch size")
    return elapsed_s * 1000.0 / iterations


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--vocab-size", type=int, default=152064)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    sampler = Sampler().to(device=device)
    metadata = make_metadata(args.batch_size, args.vocab_size, device)
    logits = make_logits(args.batch_size, args.vocab_size, dtype, device)

    timer = time_sampler_cuda if device.type == "cuda" else time_sampler_cpu
    for _ in range(args.warmups):
        sampler(logits, metadata)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = [
        timer(sampler, logits, metadata, args.iterations)
        for _ in range(args.repetitions)
    ]
    mean_ms = statistics.fmean(times_ms)
    print(f"METRIC v1_sampler_greedy_mean_ms={mean_ms:.6f}")
    print(f"METRIC v1_sampler_greedy_p50_ms={percentile(times_ms, 0.50):.6f}")
    print(f"METRIC v1_sampler_greedy_p95_ms={percentile(times_ms, 0.95):.6f}")
    print(f"METRIC v1_sampler_batch_size={float(args.batch_size):.6f}")
    print(f"METRIC v1_sampler_vocab_size={float(args.vocab_size):.6f}")
    device_is_cuda = 1.0 if device.type == "cuda" else 0.0
    print(f"METRIC v1_sampler_device_is_cuda={device_is_cuda:.6f}")


if __name__ == "__main__":
    main()
