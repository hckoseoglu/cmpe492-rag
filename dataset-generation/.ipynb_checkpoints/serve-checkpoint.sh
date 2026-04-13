#!/bin/bash
HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m vllm.entrypoints.openai.api_server --model google/gemma-2-9b-it --quantization bitsandbytes --load-format bitsandbytes --max-model-len 8192 --port 8000 --gpu-memory-utilization 0.50
