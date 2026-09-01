#!/bin/bash
# =============================================================================
# GPU 메모리 + 분산 초기화 잔여물 정리
#
# 두 가지를 처리한다.
#
# 1) GPU 메모리
#    pkill -f 'vllm' 은 워커를 못 잡는다. multiproc executor 가 띄운 워커는
#    cmdline 이 'VLLM::Worker' 라서 'vllm' 패턴에 걸리지 않고 173GB/GPU 를
#    그대로 점유한 채 남는다. 그 상태로 재기동하면
#      ValueError: Free memory on device cuda:N (6.1/178.34 GiB) on startup is
#      less than desired GPU memory utilization (0.92, 164.08 GiB)
#    로 죽는다. nvidia-smi 가 보고하는 점유 PID 를 직접 죽여야 한다.
#
# 2) 분산 초기화 잔여물
#    /tmp/vllm_dist_* (torch FileStore rendezvous), /dev/shm 세그먼트.
#    남아 있으면 다음 기동에 영향을 줄 수 있다.
#
# 주의: 이 정리만으로는 SIGKILL 반복 후의 NCCL P2P 교착이 풀리지 않는다.
#       FINDINGS.md "NCCL P2P transport 교착" 항목 참조 — 그 경우 인스턴스
#       재시작이 필요하다. 그래서 가능하면 SIGKILL 대신 정상 종료를 쓸 것.
#
# 사용: bash serve/gpu_reset.sh [대기초]
# =============================================================================
set -uo pipefail

WAIT_AFTER="${1:-${WAIT_AFTER:-20}}"     # 정리 후 대기 (CUDA IPC 비동기 해제)

used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; }
maxused() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1; }
pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u; }

echo "시작 사용량: $(used)"

# ── 1. 정상 종료 먼저 시도 (SIGTERM) ──────────────────────────────
pkill -TERM -f 'vllm serve' 2>/dev/null || true
pkill -TERM -f 'openai\.api_server' 2>/dev/null || true
sleep 5

# ── 2. 남은 GPU 점유 프로세스 강제 종료 ───────────────────────────
for _ in 1 2 3 4 5; do
    p=$(pids)
    [ -z "$p" ] && break
    echo "점유 PID: $(echo "$p" | tr '\n' ' ')-> SIGKILL"
    for x in $p; do kill -9 "$x" 2>/dev/null || true; done
    sleep 8
done
pkill -9 -f 'VLLM::' 2>/dev/null || true

# ── 3. 메모리 해제 대기 ───────────────────────────────────────────
for _ in $(seq 1 18); do
    m=$(maxused)
    [ "${m:-999999}" -lt 2000 ] && break
    echo "  해제 대기... 최대 ${m} MiB"
    sleep 10
done

# ── 4. 분산 초기화 잔여물 ─────────────────────────────────────────
n_dist=$(ls -1 /tmp/vllm_dist_* 2>/dev/null | wc -l | tr -d ' ')
n_shm=$(ls -1 /dev/shm 2>/dev/null | wc -l | tr -d ' ')
rm -f /tmp/vllm_dist_* 2>/dev/null || true
rm -f /dev/shm/psm_* /dev/shm/sem.mp-* /dev/shm/nccl-* 2>/dev/null || true
echo "잔여물 정리: /tmp/vllm_dist_* ${n_dist}개, /dev/shm ${n_shm}개"

# ── 5. 최종 확인 ──────────────────────────────────────────────────
echo "최종 사용량: $(used)"
m=$(maxused)
if [ "${m:-999999}" -ge 2000 ]; then
    echo "해제 실패 (최대 ${m} MiB). 재기동하면 OOM 으로 죽는다." >&2
    exit 1
fi

if [ "$WAIT_AFTER" -gt 0 ] 2>/dev/null; then
    echo "${WAIT_AFTER}초 대기 (CUDA IPC 비동기 해제)"
    sleep "$WAIT_AFTER"
fi
echo "정리 완료."
