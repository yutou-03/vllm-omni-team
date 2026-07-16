# Bench Scripts

Small wrappers for running `vllm bench serve` against vLLM-Omni servers.

## Qwen3-Omni random-mm

```bash
cd /home/zhongyu/project/vllm-omni-team

benchmarks/workload_generation/bench_scripts/run_qwen3_omni_random_mm.sh
```

Common overrides:

```bash
PORT=8001 \
REQUEST_RATE=10 \
BURSTINESS=1.0 \
BENCH_DURATION=20 \
MAX_CONCURRENCY=100 \
RANDOM_INPUT_LEN=32 \
RANDOM_OUTPUT_LEN=64 \
benchmarks/workload_generation/bench_scripts/run_scripts.sh
```

With the defaults above, `NUM_PROMPTS` is computed as
`REQUEST_RATE * BENCH_DURATION`, so the run sends 200 requests over roughly
20 seconds using Poisson arrivals.

The script assumes a Qwen3-Omni OpenAI-compatible server is already running on
the selected port.
