vllm serve Qwen/Qwen3-8B \
    --enable-auto-tool-choice \
    --tool-call-parser "hermes" \
    --reasoning-parser "deepseek_r1" \
    --max-model-len 16384