# RTX PRO 6000 vLLM GPU 脉冲日志分析报告

- 生成时间：2026-06-08T04:42:40.488730
- 测试机：autodl-rtx6000-2
- 模型：Qwen/Qwen3.6-35B-A3B-FP8
- 并发：72
- 数据源：`main.load.summary.json`、`main.load.jsonl`、`metrics.jsonl`、`dmon.log`、`server.log` backend runner trace。
- 限制：`dmon.log` 无绝对时间戳，和 runner trace 只能按场景窗口/采集顺序粗对齐；runner trace 是 5s 聚合，不是逐 scheduler step。

## 结论

1. 当前日志足够支撑第一轮判断：剩余 GPU 脉冲不是 MoE backend 选择或多模态预处理 cache contention 的直接问题。
2. 生产默认仍应使用 `auto/TRITON FP8 MoE`。Marlin 在四个场景中只比 TRITON 高 0.2%~2.2%，属于噪声/低收益区间。
3. `flashinfer_trtllm`、`flashinfer_cutlass`、`deep_gemm` 在当前 RTX PRO 6000 + Qwen3.6 FP8 配置下启动即失败，不是可用生产路径。
4. MTP 明确排除：纯文本吞吐从 Marlin 的 2632 tok/s 掉到 970 tok/s，多模态 warmup OOM；acceptance 平均 88%，问题不是接受率低。
5. GPU 脉冲分两类：纯文本/cache-hit 更像固定并发 synthetic load 的同步完成与补位；cache-miss/business 更像长 prompt、多图 encoder/prefill/admission 把 decode 切碎。

## 场景总览

| Backend | 场景 | 完成 tok/s | ok/err | lat p95(s) | SM avg/p50/p95 | SM<70% | low段max(s) | running avg/max | waiting avg/max | decode% | prefill/mixed% | max_query_len | tokens_max | submit_gap max(ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRITON | text_decode | 2599.00 | 4608/0 | 14.58 | 79.73/86.00/97.00 | 22.2% | 5 | 64.93/72.00 | 1.83/44.00 | 55.8% | 41.4% | 620 | 16384 | 1301.42 |
| TRITON | mm_cache_hit | 1136.36 | 925/0 | 31.35 | 56.58/65.00/87.00 | 52.7% | 7 | 58.37/72.00 | 0.00/0.00 | 1.7% | 98.3% | 1061 | 16384 | 4811.48 |
| TRITON | mm_cache_miss | 193.52 | 159/0 | 250.53 | 81.02/100.00/100.00 | 25.0% | 3 | 56.19/72.00 | 14.56/61.00 | 1.8% | 98.2% | 12576 | 12644 | 4855.25 |
| TRITON | business_sim | 229.90 | 144/0 | 261.94 | 85.83/100.00/100.00 | 20.6% | 4 | 56.96/72.00 | 13.83/50.00 | 3.6% | 96.4% | 12576 | 12646 | 20398.32 |
| MARLIN | text_decode | 2632.49 | 1584/0 | 14.50 | 83.18/89.00/100.00 | 17.2% | 4 | 65.48/72.00 | 1.82/44.00 | 56.5% | 40.3% | 620 | 16384 | 1326.77 |
| MARLIN | mm_cache_hit | 1161.82 | 943/0 | 30.46 | 63.33/72.00/99.00 | 47.6% | 9 | 58.23/72.00 | 0.00/0.00 | 3.3% | 96.7% | 1061 | 16384 | 3107.66 |
| MARLIN | mm_cache_miss | 193.86 | 159/0 | 245.85 | 85.95/100.00/100.00 | 20.3% | 3 | 55.20/72.00 | 15.09/60.00 | 1.8% | 98.2% | 12576 | 12644 | 5597.09 |
| MARLIN | business_sim | 230.80 | 144/0 | 264.32 | 90.69/100.00/100.00 | 14.9% | 3 | 57.80/72.00 | 13.10/48.00 | 3.7% | 96.3% | 12576 | 12640 | 23478.14 |

## TRITON vs Marlin

| 场景 | TRITON tok/s | Marlin tok/s | Marlin 相对变化 | 结论 |
|---|---:|---:|---:|---|
| text_decode | 2599.00 | 2632.49 | 1.29% | 噪声级/不建议切换 |
| mm_cache_hit | 1136.36 | 1161.82 | 2.24% | 噪声级/不建议切换 |
| mm_cache_miss | 193.52 | 193.86 | 0.18% | 噪声级/不建议切换 |
| business_sim | 229.90 | 230.80 | 0.39% | 噪声级/不建议切换 |

## 具体发现

### 1. 纯文本 text_decode

- TRITON 纯文本吞吐 2599.00 tok/s，SM avg/p50/p95=79.73/86.00/97.00。
- runner trace: decode-only=55.8%，prefill/mixed=41.4%。虽然没有 encoder，仍出现 tokens_max=16384、max_query_len_max=620、submit_gap max=1301.42ms。
- 解释：72 并发、固定 max_tokens、ignore_eos 的 synthetic load 会造成同批请求同步完成/补位；补位时新 prompt prefill 插入 decode 流，形成周期性 gap。

### 2. 多模态 cache-hit

- TRITON cache-hit 吞吐 1136.36 tok/s，SM avg/p50/p95=56.58/65.00/87.00，SM<70%=52.7%。
- metrics: waiting avg/max=0.00/0.00，mm cache hit rate=1.0000。
- 解释：cache-hit 场景等待队列不堆积，说明预处理 cache contention 不是当前低 GPU 的主因；更像 cache/prefix 命中后 prompt 工作量小、请求完成与补位节奏不连续。

### 3. 多模态 cache-miss 与 business_sim

- mm_cache_miss: completion=193.52 tok/s，lat p95=250.53s，SM avg/p50/p95=81.02/100.00/100.00，waiting avg/max=14.56/61.00。
  runner trace: prefill/mixed=98.2%，encoder_reqs p95=0.80，max_query_len_max=12576，tokens_max=12644。
- business_sim: completion=229.90 tok/s，lat p95=261.94s，SM avg/p50/p95=85.83/100.00/100.00，waiting avg/max=13.83/50.00。
  runner trace: prefill/mixed=96.4%，encoder_reqs p95=0.80，max_query_len_max=12576，tokens_max=12646。
- 解释：这两类场景 p50/p95 SM 已经 100%，同时 waiting 上升；瓶颈更接近长 prompt、多图 encoder、prefill/admission 把 decode 切碎，而不是 MoE backend 或 preprocessing cache。

### 4. 后端与 MTP 排除

- `flashinfer_trtllm`: (EngineCore pid=5997) ValueError: FP8 MoE backend FLASHINFER_TRTLLM does not support the deployment configuration since kernel does not support current device cuda.
- `flashinfer_cutlass`: (EngineCore pid=6327) ValueError: FP8 MoE backend FLASHINFER_CUTLASS does not support the deployment configuration since kernel does not support quantization scheme QuantKey(f8e4m3fn,scale(f32,static,GroupShape(row=128, col=128)),symmetric)xQuantKey(f8e4m3fn,scale(f32,dynamic,GroupShape(row=1, col=128)),symmetric).
- `deep_gemm`: (EngineCore pid=8550) ValueError: FP8 MoE backend DEEPGEMM does not support the deployment configuration since kernel does not support current device cuda.
- `marlin_mtp1` text_decode: 970.16 tok/s；非 MTP Marlin text_decode: 2632.49 tok/s。
- MTP acceptance 平均 88.04%（min/max=80.00/96.50），但多模态 warmup OOM：(EngineCore pid=9004) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 116.06 MiB is free. Including non-PyTorch memory, this process has 94.85 GiB memory in use. Of the allocated memory 92.97 GiB is allocated by PyTorch, with 100.00 MiB allocated in private pools (e.g., CUDA Graphs), and 612.22 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

## 现有日志够与不够

### 够

- 判断 GPU 低谷是否伴随 prefill/decode 切换。
- 判断 waiting/running 与低 GPU 的粗粒度关系。
- 区分 cache-hit、cache-miss、business 三类场景的瓶颈形态。
- 排除 Marlin/MTP/FlashInfer/DeepGEMM 作为当前生产优化主线。

### 不够

- `dmon.log` 无绝对时间戳，不能和单个 runner interval 精确对齐。
- runner trace 是 5s 聚合，不是每个 scheduler step 明细。
- 缺少 per-request admission/render/preprocess/client refill 时间，不能精确定责补位空窗在 API server、scheduler、renderer 还是 client。
- 这是 synthetic load，不等价于真实生产流量分布。

## 后续建议

1. 当前优化 PR 可以继续收敛：后端/MTP 方向没有显示出比 preprocessing PR 更直接的收益。
2. 生产侧继续保持 `auto/TRITON FP8 MoE`。
3. 如果继续追 GPU 脉冲，下一轮不要再扫 backend；应加窄 trace：每 step 的 decode/prefill token、request arrival/admitted/first scheduled/finished/replacement submitted、renderer/MM preprocess 分段时间，并让 dmon 带绝对时间戳。
4. 在真实业务上采同样指标，确认 synthetic 同步完成波形是否存在于线上。

## 文件

- 结构化汇总：`/root/vllm/pulse_bench/results/gpu_pulse_analysis_summary_20260608.json`
- 本报告：`/root/vllm/pulse_bench/results/gpu_pulse_analysis_report_20260608.md`
