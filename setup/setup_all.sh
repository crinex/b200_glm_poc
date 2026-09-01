#!/bin/bash
# =============================================================================
# 통합 세팅 — 빈 인스턴스에서 이 스크립트 하나만 실행하면 측정 가능 상태까지 간다
#
#   git clone https://<PAT>@github.com/crinex/b200_glm_poc.git /workspace/b200_glm_poc \
#     || { sudo mkdir -p /ephemeral/workspace && sudo ln -s /ephemeral/workspace /workspace \
#          && git clone ... ; }   # /workspace 가 없는 프로바이더면 먼저 만들 것
#   cd /workspace/b200_glm_poc && bash setup/setup_all.sh
#
# 하는 일:
#   [감지] 프로바이더 환경 판별 → 필요할 때만 setup_deploygpu.sh
#          (venv 없음 / 드라이버 < 580 / Fabric Manager 미기동 중 하나라도 해당 시)
#   [세팅] full_setup.sh — vLLM 0.28.0, torchaudio 제거, 모델 704GB,
#          gen8k 1,024장, 하드웨어 지문
#
# 이후 실험: bench/experiments.conf 를 편집하고
#   bash bench/run_experiments.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=========== 환경 감지 ==========="
NEED_PROVIDER_INIT=0
REASONS=()

if [ ! -x /venv/main/bin/python3 ]; then
    NEED_PROVIDER_INIT=1; REASONS+=("/venv/main 없음")
fi
if [ ! -e /workspace ]; then
    NEED_PROVIDER_INIT=1; REASONS+=("/workspace 없음")
fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 0)
BRANCH="${DRV%%.*}"
if [ "${BRANCH:-0}" -lt 580 ] 2>/dev/null; then
    NEED_PROVIDER_INIT=1; REASONS+=("드라이버 $DRV — CUDA 13 미지원 브랜치")
fi
# NVL5 에서 FM 이 없으면 CUDA 가 Error 802 로 실패한다
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
    if ! systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
        # FM 유닛이 존재하는데 죽어 있으면 초기화 필요 (Vast 이미지는 유닛 자체가 없거나 active)
        if [ "${BRANCH:-0}" -lt 580 ] 2>/dev/null; then
            REASONS+=("Fabric Manager 비활성")
        fi
    fi
fi

echo "  드라이버: $DRV"
if [ "$NEED_PROVIDER_INIT" = "1" ]; then
    echo "  → 프로바이더 초기화 필요: ${REASONS[*]}"
    echo ""
    echo "=========== 프로바이더 초기화 (setup_deploygpu.sh) ==========="
    bash setup/setup_deploygpu.sh
else
    echo "  → 프로바이더 초기화 불필요 (Vast 계열 이미지로 판단)"
fi

echo ""
echo "=========== 본 세팅 (full_setup.sh) ==========="
bash setup/full_setup.sh

cat <<'EOF'

=========== 전체 세팅 완료 ===========
  다음:
    1) 실험 목록 편집     : bench/experiments.conf
    2) 실험 순차 실행     : setsid nohup bash bench/run_experiments.sh \
                             > /workspace/logs/experiments.out 2>&1 &
    3) 진행 확인          : tail -f /workspace/logs/experiments.out
    4) 결과               : /workspace/results/<실험이름>_sweep/sweep_summary.md
======================================
EOF
