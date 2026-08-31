#!/bin/bash
# =============================================================================
# GLM-5.2 FP8 모델 다운로드
# 사용: HF_TOKEN=xxx bash setup/download_model.sh
# =============================================================================
set -e

MODEL_ID="zai-org/GLM-5.2-FP8"
MODEL_DIR="/workspace/models/GLM-5.2-FP8"

echo "======================================"
echo "  GLM-5.2 FP8 모델 다운로드"
echo "  소스: $MODEL_ID"
echo "  저장: $MODEL_DIR"
echo "======================================"

# HuggingFace 로그인
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
else
    echo "⚠️  HF_TOKEN 환경변수가 없습니다."
    echo "   export HF_TOKEN=hf_xxxx 후 재실행하세요."
    echo "   (public 모델이라면 토큰 없이도 가능)"
fi

mkdir -p "$MODEL_DIR"

# hf_transfer로 빠른 다운로드
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "📥 다운로드 시작..."
huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

echo ""
echo "✅ 다운로드 완료"
du -sh "$MODEL_DIR"
ls "$MODEL_DIR"
