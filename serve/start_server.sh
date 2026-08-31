#!/bin/bash
# =============================================================================
# vLLM GLM-5.2 FP8 서버 시작
# B200 × 8GPU, FlashInfer Sparse MLA, MTP Speculative Decoding
# =============================================================================

MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"          # Tensor Parallel (단일 B200으로 시작, 필요시 늘리기)

echo "======================================"
echo "  vLLM GLM-5.2 FP8 서버 시작"
echo "  모델: $MODEL_DIR"
echo "  TP:   $TP GPUs"
echo "  포트: $PORT"
echo "======================================"

# 모델 존재 확인
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 모델 없음: $MODEL_DIR"
    echo "   bash setup/download_model.sh 먼저 실행하세요."
    exit 1
fi

# B200에서 FlashInfer 기본 backend (sm_100 자동 감지)
# Sparse MLA, FA3 모두 자동 활성화

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "glm-5.2" \
    --tensor-parallel-size "$TP" \
    --dtype float8_e4m3fn \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    2>&1 | tee /workspace/logs/vllm_server.log
