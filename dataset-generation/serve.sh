#!/bin/bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="google/gemma-2-9b-it"
export LLM_API_KEY="dummy"
export LLM_IS_GEMMA="true"
export LD_LIBRARY_PATH="/opt/conda/envs/tensorflow/lib:${LD_LIBRARY_PATH}"

HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m vllm.entrypoints.openai.api_server --model google/gemma-2-9b-it --quantization bitsandbytes --load-format bitsandbytes --max-model-len 8192 --port 8000 --gpu-memory-utilization 0.50
