#!/bin/bash
# =============================================================================
# deploygpu 계열 인스턴스 초기화 — full_setup.sh 실행 전에 1회
#
# 이 프로바이더의 빈 인스턴스는 Vast.ai 이미지와 달리 아무것도 없다:
#   - /venv/main 없음, /workspace 없음 (대신 /ephemeral 39TB)
#   - 드라이버 570.195.03 (CUDA 12.8 브랜치) — torch cu130 이 거부됨
#   - Fabric Manager 미기동 — CUDA 초기화가 Error 802 로 실패
#
# 2026-09-01 에 실측으로 확립한 복구 순서다. 각 단계의 에러 증상:
#   [A] 없음        → ensurepip 오류로 venv 생성 실패
#   [B] 없음        → torch.cuda.is_available() = False (드라이버 버전 부족)
#   [C] 없음        → Error 802: system not yet initialized
#       C 내부 순서: ibstat 없음 → ib_umad 미로드 → nvlsm 없음 순으로 FM 이 실패
#
# 사용: bash setup/setup_deploygpu.sh
# 이후: bash setup/full_setup.sh
# =============================================================================
set -euo pipefail

echo "=========== [A] 기본 구조 ==========="
sudo apt-get update -q
sudo apt-get install -y -q python3.12-venv infiniband-diags

# /workspace -> 대용량 ephemeral 볼륨
sudo mkdir -p /ephemeral/workspace
sudo chown "$USER:$USER" /ephemeral/workspace
[ -e /workspace ] || sudo ln -s /ephemeral/workspace /workspace

# /venv/main — 모든 스크립트가 이 경로를 전제한다
sudo mkdir -p /venv
sudo chown "$USER:$USER" /venv
if [ ! -x /venv/main/bin/python3 ]; then
    python3 -m venv /venv/main
fi
/venv/main/bin/python3 -m pip install -q --upgrade pip

echo "=========== [B] CUDA 13 forward compatibility ==========="
# 드라이버 570 은 CUDA 12.8 까지. torch 2.13.0+cu130(vLLM 0.28.0 의존)을
# 돌리려면 forward-compat libcuda 가 필요하다. 재부팅 불필요.
if ! ls /etc/apt/sources.list.d/ 2>/dev/null | grep -qi cuda; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null
    sudo apt-get update -q
fi
sudo apt-get install -y -q cuda-compat-13-0

# compat libcuda 를 ld 검색 최우선으로 (모든 프로세스에 적용, env 불필요)
echo /usr/local/cuda-13.0/compat | sudo tee /etc/ld.so.conf.d/00-cuda-compat.conf > /dev/null
sudo ldconfig
ldconfig -p | grep -m1 "libcuda.so.1" | grep -q compat \
    && echo "  compat libcuda 우선순위 OK" \
    || { echo "  경고: compat 가 최우선이 아님" >&2; exit 1; }

echo "=========== [C] Fabric Manager (NVL5 필수) ==========="
# 8×B200 NVLink5 는 FM 등록 전까지 CUDA 초기화가 Error 802 로 실패한다.
# FM 버전은 커널 드라이버와 정확히 일치해야 한다.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
BRANCH="${DRV%%.*}"
echo "  커널 드라이버: $DRV (브랜치 $BRANCH)"
sudo apt-get install -y -q "nvidia-fabricmanager-${BRANCH}=${DRV}-1" \
    || sudo apt-get install -y -q "nvidia-fabricmanager-${BRANCH}"

# NVL5 는 NVLink Subnet Manager 와 ib_umad 커널 모듈이 필요
sudo apt-get install -y -q nvlsm
sudo modprobe ib_umad
echo ib_umad | sudo tee /etc/modules-load.d/ib_umad.conf > /dev/null

sudo systemctl enable nvidia-fabricmanager
sudo systemctl restart nvidia-fabricmanager
sleep 10
systemctl is-active --quiet nvidia-fabricmanager \
    && echo "  Fabric Manager: active" \
    || { echo "  FM 기동 실패:" >&2
         sudo journalctl -u nvidia-fabricmanager --no-pager -n 12 >&2
         exit 1; }

echo "=========== 검증 ==========="
/venv/main/bin/python3 - <<'PYEOF' 2>/dev/null || echo "  (torch 미설치 — full_setup.sh 후 자동 검증됨)"
import torch
ok = torch.cuda.is_available()
print("  torch.cuda.is_available:", ok)
assert ok, "CUDA 초기화 실패"
print("  device:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())
PYEOF

cat <<EOF

=========== deploygpu 초기화 완료 ===========
  다음: bash setup/full_setup.sh
=============================================
EOF
