# Business trace bottleneck analysis

- trace dir: `/root/vllm/business_trace/real-biz-20260608-122908`
- active duration: 4738.2s
- generated at: 1780898627.741

## Key numbers

- GPU util avg/p50/p95/min/max: 59.155/48.0/100.0/0.0/100.0
- running avg/p50/p95/min/max: 69.331/70.0/72.0/1.0/72.0
- waiting avg/p95/max: 0.327/2.0/35.0
- ready_decode avg/p50/p95/min: 69.082/70.0/72.0/1.0
- scheduled generation tokens avg/p50/p95/min: 69.082/70.0/72.0/1.0
- scheduled context tokens avg/p95/max: 467.045/3227.0/16339.0
- runner submit_gap avg/p95/max: 137.194/588.109/1685.781 ms
- runner mode counts: {'NONE': 3789, 'PIECEWISE': 157, 'FULL': 30589}
- decode FULL forward avg/p95/max: 0.529/0.575/8.719 ms
- decode FULL sample total avg/p95/max: 43.767/66.802/1650.685 ms
- prefill NONE forward avg/p95/max: 480.483/708.432/1050.263 ms

## Low GPU vs high GPU runner buckets

### low
- n=18027, modes={'NONE': 1773, 'FULL': 16168, 'PIECEWISE': 86}, encoder_rate=0.0467, none_rate=0.0984
- submit_gap avg/p95/max=143.706/549.324/1685.781 ms
- forward avg/p95/max=46.467/437.703/801.321 ms
- sample_total avg/p95/max=55.038/67.516/1650.685 ms

### mid
- n=7700, modes={'FULL': 7293, 'NONE': 389, 'PIECEWISE': 18}, encoder_rate=0.0314, none_rate=0.0505
- submit_gap avg/p95/max=99.183/480.521/1281.172 ms
- forward avg/p95/max=24.171/421.269/761.27 ms
- sample_total avg/p95/max=37.801/62.08/374.249 ms

### high
- n=8808, modes={'NONE': 1627, 'PIECEWISE': 53, 'FULL': 7128}, encoder_rate=0.1515, none_rate=0.1847
- submit_gap avg/p95/max=157.098/656.654/1485.141 ms
- forward avg/p95/max=95.061/512.365/1050.263 ms
- sample_total avg/p95/max=29.715/73.559/488.37 ms

## Requests

{
  "admit_count": 3054,
  "admit_wait_s": {
    "avg": 2.599,
    "max": 21.746,
    "min": 0.015,
    "n": 3054,
    "p50": 2.225,
    "p90": 4.791,
    "p95": 6.035,
    "p99": 13.072
  },
  "engine_add_count": 3028,
  "engine_wait_s": {
    "avg": 2.154,
    "max": 10.781,
    "min": 0.014,
    "n": 3028,
    "p50": 1.936,
    "p90": 3.904,
    "p95": 4.858,
    "p99": 7.453
  },
  "finish_age_s": {
    "avg": 106.637,
    "max": 1800.063,
    "min": 2.078,
    "n": 3029,
    "p50": 62.528,
    "p90": 259.319,
    "p95": 386.403,
    "p99": 730.701
  },
  "finish_count": 3029,
  "finish_output_tokens": {
    "avg": 753.133,
    "max": 13226.0,
    "min": 18.0,
    "n": 3029,
    "p50": 433.0,
    "p90": 1902.0,
    "p95": 2760.0,
    "p99": 5237.0
  },
  "mm_features_added": {
    "avg": 3.764,
    "max": 9.0,
    "min": 0.0,
    "n": 3028,
    "p50": 4.0,
    "p90": 7.0,
    "p95": 8.0,
    "p99": 9.0
  },
  "preemptions_total": 0,
  "prompt_tokens_added": {
    "avg": 22984.385,
    "max": 49298.0,
    "min": 1307.0,
    "n": 3028,
    "p50": 21872.0,
    "p90": 41226.0,
    "p95": 43176.0,
    "p99": 46755.0
  }
}