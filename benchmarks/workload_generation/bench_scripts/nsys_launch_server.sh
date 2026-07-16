#!/usr/bin/env bash
set -euo pipefail

# Nsight Systems recommends spawn for vLLM multiprocessing.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
export TMPDIR=/home/zhongyu/project/vllm-omni-team/output/nsys_tmp
mkdir -p "${TMPDIR}"

nsys launch --session=zhongyu \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  bash /home/zhongyu/project/vllm-omni-team/benchmarks/workload_generation/bench_scripts/serve.bash


#gzip -k /home/zhongyu/project/vllm-omni-team/output/nsys-qwen-20_requests.nsys-rep
#scp -C AMS02:/home/zhongyu/project/vllm-omni-team/output/nsys-qwen-20_requests.nsys-rep D:\