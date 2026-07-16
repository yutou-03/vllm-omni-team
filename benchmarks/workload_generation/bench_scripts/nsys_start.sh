#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/home/zhongyu/project/vllm-omni-team/output/nsys_tmp
mkdir -p "${TMPDIR}"

nsys start --session=zhongyu \
  --gpu-metrics-devices all \
  --force-overwrite=true \
  -o /home/zhongyu/project/vllm-omni-team/output/nsys-qwen-20_requests
