#!/bin/bash
# 실행 중인 측정의 로그를 태그로 골라 본다.
# 사용: bash serve/watch.sh <tag> [server|live|bench|out|result]
#   tag 예: mtp1_ep / mtp1_noep / nomtp_noep / mtp1_ep8_run2
#   생략 시 /workspace/logs 의 최신 vllm_*.log 태그를 자동 선택
set -u
LOGDIR="${LOGDIR:-/workspace/logs}"
TAG="${1:-$(ls -t "$LOGDIR"/vllm_*.log 2>/dev/null | head -1 | sed 's|.*/vllm_||; s|\.log$||')}"
MODE="${2:-server}"
[ -n "$TAG" ] || { echo "태그를 찾을 수 없음. 사용: watch.sh <tag> [mode]"; exit 1; }
echo "# tag=$TAG mode=$MODE"
case "$MODE" in
  server) tail -f "$LOGDIR/vllm_${TAG}.log" ;;
  live)   tail -f "$LOGDIR/vllm_${TAG}.log" | grep --line-buffered -E "Running:|acceptance" ;;
  bench)  tail -f "$LOGDIR/bench_${TAG}.log" ;;
  out)    tail -f "$LOGDIR/run_${TAG}.out" 2>/dev/null || tail -f "$LOGDIR"/run_*.out ;;
  result) cat "/workspace/results/${TAG}_sweep/sweep_summary.md" ;;
  *)      echo "mode: server|live|bench|out|result" ;;
esac
