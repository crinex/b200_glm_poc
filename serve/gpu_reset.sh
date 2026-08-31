#!/bin/bash
# =============================================================================
# GPU 메모리 강제 해제
#
# pkill -f 'vllm' 은 워커를 못 잡는다. multiproc executor 가 띄운 워커
# 프로세스는 cmdline 에 'vllm' 문자열이 없어서 패턴에 걸리지 않고,
# 173GB/GPU 를 그대로 점유한 채 남는다. 그 상태로 재기동하면
#   ValueError: Free memory on device cuda:N (6.1/178.34 GiB) on startup is
#   less than desired GPU memory utilization (0.92, 164.08 GiB)
# 로 죽는다.
#
# nvidia-smi 가 보고하는 점유 PID 를 직접 죽이는 것이 확실하다.
# =============================================================================
set -uo pipefail

used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; }
pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u; }

echo "시작 사용량: $(used)"

for round in 1 2 3 4 5; do
    p=$(pids)
    [ -z "$p" ] && break
    echo "점유 PID: $(echo "$p" | tr '\n' ' ')-> SIGKILL"
    for x in $p; do kill -9 "$x" 2>/dev/null || true; done
    sleep 8
done

for _ in $(seq 1 18); do
    max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          | sort -rn | head -1)
    [ "${max:-999999}" -lt 2000 ] && break
    echo "  해제 대기... 최대 ${max} MiB"
    sleep 10
done

echo "최종 사용량: $(used)"
max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${max:-999999}" -ge 2000 ]; then
    echo "해제 실패 (최대 ${max} MiB). 재기동하면 OOM 으로 죽는다." >&2
    exit 1
fi
echo "정리 완료."
