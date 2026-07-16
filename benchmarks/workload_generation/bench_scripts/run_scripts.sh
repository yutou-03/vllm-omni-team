#!/usr/bin/env bash
#stage_profile
#curl -X POST http://localhost:8000/start_profile
#vllm bench serve request-rate 1 + num-prompts 1
vllm bench serve \
  --omni \
  --dataset-name random-mm \
  --port 8000 \
  --model /data/models/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --request-rate 10 \
  --burstiness 2.0 \
  --num-prompts 20 \
  --random-input-len 32 \
  --random-range-ratio 0.0 \
  --random-mm-base-items-per-request 0 \
  --random-mm-num-mm-items-range-ratio 0 \
  --random-mm-limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10}' \
  --ignore-eos \
  --random-output-len 256 \
  --extra_body '{"modalities": ["text", "audio"]}'


#curl -X POST http://localhost:8000/stop_profile
#output生成的trace打开https://ui.perfetto.dev/

#不使用手动curl的方式就直接在vllm bench serve加 --profile \，但要注意限制时间否则trace很大。
