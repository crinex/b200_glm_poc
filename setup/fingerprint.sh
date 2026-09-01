#!/bin/bash
# =============================================================================
# 하드웨어/환경 지문 수집
#
# 목적: 측정 수치가 "어느 하드웨어에서 나온 값인지"를 남긴다.
# 서버 A/B/C 를 거치며 같은 구성이 인스턴스 간 -8~-16% 차이났는데,
# 원인 판정에 필요한 값(클럭·전력 상한, NVLink 토폴로지, CPU, 가상화,
# FM 파티션)을 수집하지 않아 인스턴스 소멸 후 소급 확인이 불가능했다.
# (FINDINGS.md §6-2 — 서버 간 편차가 구성 효과를 압도)
#
# full_setup.sh 가 자동 호출한다. 단독 실행도 가능:
#   bash setup/fingerprint.sh [출력파일]
#
# 출력 기본값: /workspace/results/fingerprint_<호스트>_<날짜>.txt
# 측정 결과를 레포로 회수할 때 이 파일도 함께 가져갈 것.
# =============================================================================
set -uo pipefail

HOST_ID="$(hostname)"
OUT="${1:-/workspace/results/fingerprint_${HOST_ID}_$(date +%Y%m%d).txt}"
mkdir -p "$(dirname "$OUT")"

# 실패해도 다음 섹션으로 넘어간다 — 지문은 부분적이어도 가치가 있다.
sec() { echo ""; echo "===== $* ====="; }

{
echo "# 하드웨어/환경 지문"
echo "host      : $HOST_ID"
echo "collected : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "uptime -s : $(uptime -s 2>/dev/null || true)"

sec "GPU 개요"
nvidia-smi --query-gpu=index,name,uuid,memory.total \
    --format=csv 2>/dev/null || echo "(nvidia-smi 실패)"

sec "드라이버 / 펌웨어"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
cat /proc/driver/nvidia/version 2>/dev/null | head -2
# forward-compat 사용 여부 (커널 드라이버와 libcuda 버전이 다르면 compat)
ldconfig -p 2>/dev/null | grep -m1 "libcuda.so.1"
ls /usr/local/cuda-*/compat/libcuda.so.* 2>/dev/null | head -2

sec "전력 / 클럭 상한  ← 서버 간 성능차 판정의 핵심"
nvidia-smi --query-gpu=index,power.limit,power.max_limit,power.default_limit,clocks.max.sm,clocks.max.mem,persistence_mode \
    --format=csv 2>/dev/null
nvidia-smi -q -d CLOCK -i 0 2>/dev/null | grep -A4 "Max Clocks" | head -6

sec "NVLink 토폴로지"
nvidia-smi topo -m 2>/dev/null
echo ""
nvidia-smi nvlink --status -i 0 2>/dev/null | head -20

sec "Fabric Manager / NVSwitch"
systemctl is-active nvidia-fabricmanager 2>/dev/null || true
# 파티션 id — 멀티테넌트 NVSwitch 분할 여부의 단서
journalctl -u nvidia-fabricmanager --no-pager 2>/dev/null \
    | grep -aE "partition|MASTER state" | tail -3
ls /dev/nvidia-nvswitch* 2>/dev/null || echo "(nvswitch 디바이스 노드 없음)"

sec "호스트 CPU / NUMA"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Socket|NUMA node|MHz" | head -8

sec "가상화 / 컨테이너"
systemd-detect-virt 2>/dev/null || true
grep -m1 -o hypervisor /proc/cpuinfo 2>/dev/null || echo "(hypervisor 플래그 없음)"
[ -f /.dockerenv ] && echo "docker 컨테이너" || echo "(dockerenv 없음)"
lspci 2>/dev/null | grep -ci nvidia | sed 's/^/lspci NVIDIA 장치 수: /'

sec "OS / 커널"
grep PRETTY /etc/os-release 2>/dev/null
uname -r

sec "메모리 / 디스크"
free -g 2>/dev/null | head -2
df -h / /workspace /ephemeral 2>/dev/null | awk 'NR==1 || !seen[$1]++'

sec "소프트웨어 스택"
PY="${PY:-/venv/main/bin/python3}"
if [ -x "$PY" ]; then
    "$PY" -V
    "$PY" - <<'PYEOF' 2>/dev/null || echo "(torch/vllm 미설치)"
import torch
print("torch :", torch.__version__, "| CUDA", torch.version.cuda)
try:
    import vllm
    print("vllm  :", vllm.__version__)
except Exception:
    print("vllm  : 미설치")
PYEOF
else
    echo "($PY 없음)"
fi
nvcc --version 2>/dev/null | grep release || echo "nvcc 없음"

sec "NCCL 관련"
"$PY" -c 'import torch; print("nccl:", torch.cuda.nccl.version())' 2>/dev/null || true
lsmod 2>/dev/null | grep -E "^ib_umad|^nvidia " | head -3
} > "$OUT" 2>&1

echo "지문 저장: $OUT"
echo "  ($(wc -l < "$OUT") 줄)"
