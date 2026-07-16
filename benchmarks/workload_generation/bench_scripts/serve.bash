vllm serve /data/models/Qwen3-Omni-30B-A3B-Instruct \
     --omni \
     --port 8000 \
     --stage-configs-path /home/zhongyu/project/vllm-omni-team/vllm_omni/deploy/qwen3_omni_moe.yaml

#启动带profiler 的 server
#export VLLM_RPC_TIMEOUT=1800000
#vllm serve /data/models/Qwen3-Omni-30B-A3B-Instruct \
#  --omni \
#  --port 8000 \
#  --stage-configs-path /home/zhongyu/project/vllm-omni-team/vllm_omni/deploy/qwen3_omni_moe.yaml
