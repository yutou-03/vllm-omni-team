#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/home/zhongyu/project/vllm-omni-team/output/nsys_tmp
mkdir -p "${TMPDIR}"

nsys stop --session=zhongyu
