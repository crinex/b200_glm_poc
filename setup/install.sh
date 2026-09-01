#!/bin/bash
# =============================================================================
# B200 × GLM-5.2 환경 설치
#
# 주의: 맨 python3 / pip3 를 쓰면 안 된다. /usr/bin/python3 로 잡히고
#       torch 는 venv 에만 있어서 vLLM 이 엉뚱한 곳에 설치된다.
#       반드시 venv 인터프리터를 명시한다.
# =============================================================================
set -e

# ── venv 인터프리터 결정 ───────────────────────────────────────
PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in /venv/main/bin/python3 /venv/bin/python3; do
        if [ -x "$c" ]; then PY="$c"; break; fi
    done
fi
if [ -z "$PY" ]; then
    echo "venv 인터프리터를 못 찾았습니다. PY=/경로/python3 로 지정하세요." >&2
    exit 1
fi

VLLM_VERSION="${VLLM_VERSION:-0.28.0}"   # FINDINGS.md 기준 검증 버전
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "======================================"
echo "  B200 × GLM-5.2 환경 설치"
echo "  인터프리터 : $PY"
echo "  vLLM       : $VLLM_VERSION"
echo "======================================"

echo "[1/5] 환경 확인"
"$PY" -V
nvcc --version 2>/dev/null | grep release || echo "  nvcc 없음 (문제 아님)"
nvidia-smi -L | head -3
"$PY" -c 'import torch; print("  torch", torch.__version__)' 2>/dev/null \
    || echo "  torch 미설치 (vLLM 설치 시 함께 들어옴)"

echo "[2/5] vLLM"
if "$PY" -c "import vllm" 2>/dev/null; then
    echo "  이미 설치됨: $("$PY" -c 'import vllm; print(vllm.__version__)')"
else
    echo "  설치 중... (torch 가 함께 갱신될 수 있음)"
    "$PY" -m pip install "vllm==$VLLM_VERSION" \
        --extra-index-url "$TORCH_INDEX" \
        --root-user-action=ignore
    echo "  완료: $("$PY" -c 'import vllm; print(vllm.__version__)')"
fi
"$PY" -c 'import torch; print("  torch:", torch.__version__, "| CUDA:", torch.version.cuda)'

echo "[3/5] flash-attn (선택)"
if "$PY" -c "import flash_attn" 2>/dev/null; then
    echo "  이미 설치됨"
else
    "$PY" -m pip install flash-attn --no-build-isolation -q \
        --root-user-action=ignore 2>/dev/null \
        && echo "  완료" \
        || echo "  실패 — 무시 가능. sm_100 에서는 vLLM 내장 FlashInfer 가 쓰인다"
fi

echo "[4/5] huggingface_hub"
# HF_XET_HIGH_PERFORMANCE 를 쓰려면 >= 1.28 필요 (download_model.sh 참조)
"$PY" -m pip install -U "huggingface_hub>=1.28" -q --root-user-action=ignore
"$PY" -c 'import huggingface_hub as h; print("  huggingface_hub", h.__version__)'

echo "[5/5] 모델 확인"
MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    n=$(find "$MODEL_DIR" -name '*.safetensors' | wc -l | tr -d ' ')
    echo "  존재: $MODEL_DIR ($n safetensors, $(du -sh "$MODEL_DIR" | cut -f1))"
    [ "$n" -eq 141 ] || echo "  경고: safetensors 141 개가 정상. 다운로드 미완일 수 있음"
else
    echo "  없음. 다음을 실행: HF_TOKEN=hf_xxx bash setup/download_model.sh"
fi

cat <<EOF

======================================
  설치 완료

  다음 순서:
    1) HF_TOKEN=hf_xxx bash setup/download_model.sh
    2) bash bench/workload/build_gen8k.sh /workspace/gen8k 1024
    3) bash serve/start_server.sh
======================================
EOF
