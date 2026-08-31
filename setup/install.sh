#!/bin/bash
# =============================================================================
# B200 × GLM-5.2 환경 초기화 스크립트
# 서버 리셋 후 실행: bash setup/install.sh
# =============================================================================
set -e

echo "======================================"
echo "  B200 × GLM-5.2 환경 설정 시작"
echo "======================================"

# ── 1. 기본 환경 확인 ──────────────────────────────────────────
echo "[1/5] 환경 확인..."
python3 --version
nvcc --version | grep release
nvidia-smi -L | head -3

# ── 2. vLLM 설치 ───────────────────────────────────────────────
echo "[2/5] vLLM 설치 확인..."
if python3 -c "import vllm" 2>/dev/null; then
    echo "  ✅ vLLM $(python3 -c 'import vllm; print(vllm.__version__)') 이미 설치됨"
else
    echo "  📦 vLLM 설치 중..."
    pip3 install vllm \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --root-user-action=ignore -q
    echo "  ✅ vLLM 설치 완료: $(python3 -c 'import vllm; print(vllm.__version__)')"
fi

# ── 3. flash-attn 확인 ─────────────────────────────────────────
echo "[3/5] flash-attn 확인..."
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "  ✅ flash-attn 이미 설치됨"
else
    echo "  📦 flash-attn 설치 중... (시간 걸릴 수 있음)"
    pip3 install flash-attn --no-build-isolation -q \
        --root-user-action=ignore 2>/dev/null || echo "  ⚠️  flash-attn 설치 실패 (FA3는 vLLM 내장으로 동작)"
fi

# ── 4. HuggingFace CLI 설치 ────────────────────────────────────
echo "[4/5] HuggingFace CLI..."
pip3 install huggingface_hub[hf_transfer] -q --root-user-action=ignore
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

# ── 5. 모델 존재 확인 ──────────────────────────────────────────
echo "[5/5] 모델 확인..."
MODEL_DIR="/workspace/models/GLM-5.2-FP8"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "  ✅ 모델 존재: $MODEL_DIR"
    du -sh "$MODEL_DIR"
else
    echo "  ⚠️  모델 없음. 다운로드 필요:"
    echo "       bash setup/download_model.sh"
fi

echo ""
echo "======================================"
echo "  설정 완료! 서버 시작:"
echo "  bash serve/start_server.sh"
echo "======================================"
