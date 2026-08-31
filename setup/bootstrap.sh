#!/bin/bash
# =============================================================================
# 서버 리셋 후 최초 1회 실행 — 레포 클론 + 환경 복원
# 사용: curl -fsSL https://raw.githubusercontent.com/crinex/b200_glm_poc/main/setup/bootstrap.sh | bash
# =============================================================================
set -e

REPO="https://github.com/crinex/b200_glm_poc.git"
WORKDIR="/workspace/b200_glm_poc"

echo "======================================"
echo "  B200 GLM-5.2 Bootstrap"
echo "======================================"

# 레포 클론 or pull
if [ -d "$WORKDIR/.git" ]; then
    echo "📥 레포 업데이트..."
    git -C "$WORKDIR" pull --ff-only
else
    echo "📥 레포 클론..."
    git clone "$REPO" "$WORKDIR"
fi

cd "$WORKDIR"

# 환경 설치
bash setup/install.sh

# 모델이 없으면 다운로드 안내
MODEL_DIR="/workspace/models/GLM-5.2-FP8"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo ""
    echo "⚠️  모델 다운로드가 필요합니다:"
    echo "   export HF_TOKEN=hf_xxxx"
    echo "   bash $WORKDIR/setup/download_model.sh"
fi

echo ""
echo "✅ Bootstrap 완료!"
echo "   작업 디렉토리: $WORKDIR"
echo "   서버 시작: bash $WORKDIR/serve/start_server.sh"
