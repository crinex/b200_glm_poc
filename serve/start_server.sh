#!/bin/bash
# =============================================================================
# vLLM GLM-5.2 FP8 서버 — B200 × 8 실측 검증 구성
#
# 이 스크립트의 인자 조합은 2026-08-31 세션에서 실제로 기동·측정에 성공한 것이다.
# results/ 의 모든 측정값이 이 구성에서 나왔다.
#
# 인자별 근거는 FINDINGS.md 참조. 요약:
#   --tensor-parallel-size 8   필수. 모델 ~700GB, B200 1장(178GB)에 안 들어감.
#   --max-model-len 32768      필수. 모델 native 1,048,576 그대로 두면 KV cache가
#                              최대 길이 요청 하나를 못 담아 기동 거부.
#   --kv-cache-dtype fp8       선택적 최적화. KV cache 796,160 → 1,546,112 토큰.
#                              빼도 기동/추론 정상 (bf16). 실측 확인됨.
#   --trust-remote-code        필수. GLM 커스텀 모델링 코드.
#
# 주의: --dtype float8_e4m3fn 은 vLLM 0.28 에서 무효값이다. auto 를 쓸 것.
#       (이전 버전 스크립트가 이걸로 기동 실패했다)
# =============================================================================
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
MAX_LEN="${MAX_LEN:-32768}"
KV_DTYPE="${KV_DTYPE:-fp8}"          # fp8 | auto(=bf16)
MAX_BATCHED="${MAX_BATCHED:-8192}"
GPU_UTIL="${GPU_UTIL:-0.92}"         # 0.92 는 vLLM 기본값과 동일
LOG="${LOG:-/workspace/logs/vllm_server.log}"

# venv 인터프리터를 명시한다. 맨 python3 는 /usr/bin/python3 로 잡혀
# ModuleNotFoundError: No module named 'vllm' 이 난다.
PY="${PY:-/venv/main/bin/python3}"

if [ ! -x "$PY" ]; then
    echo "인터프리터 없음: $PY" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "모델 없음: $MODEL_DIR — bash setup/download_model.sh 먼저 실행" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo "  vLLM GLM-5.2 서버 시작"
echo "  모델        : $MODEL_DIR"
echo "  TP          : $TP"
echo "  max-model-len: $MAX_LEN"
echo "  kv-cache    : $KV_DTYPE"
echo "  max-batched : $MAX_BATCHED"
echo "  gpu-util    : $GPU_UTIL"
echo "  포트        : $PORT"
echo "  로그        : $LOG"
echo "======================================"

ARGS=(
    --model "$MODEL_DIR"
    --served-model-name glm-5.2
    --tensor-parallel-size "$TP"
    --dtype auto
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-batched-tokens "$MAX_BATCHED"
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --host "$HOST"
    --port "$PORT"
    --trust-remote-code
)
if [ "$KV_DTYPE" != "auto" ]; then
    ARGS+=(--kv-cache-dtype "$KV_DTYPE")
fi

# MTP speculative decoding — 아직 검증 안 됨. 켜려면 아래 주석 해제.
# ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')

exec "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" 2>&1 | tee "$LOG"
