#!/bin/bash
# =============================================================================
# 3구성 순차 측정 체인
#   1) nomtp_noep : Weight FP8, KV fp8, TP8, DP1, EP1, MTP 없음 (기준선)
#   2) mtp1_noep  : + MTP spec-tokens 1
#   3) mtp1_ep    : + MTP spec-tokens 1 + EP8
# 각 실험 결과 폴더에 server_args.txt / config.txt / run_output.txt /
# fingerprint.txt 가 동봉된다. 전체 진행: /workspace/logs/chain3.log
# 사용: setsid nohup bash bench/run_3config_chain.sh > /dev/null 2>&1 &
# =============================================================================
# 공통: gen8k 1,024장, ISL~8,200 / OSL 1,024 고정(ignore_eos), sweep 4~256
exec > /workspace/logs/chain3.log 2>&1
set -u
cd /workspace/b200_glm_poc

run_one() {
  local name="$1"; shift
  echo ""
  echo "########## [$name] 시작 $(date +%H:%M:%S) ##########"
  # 측정 (run 스크립트가 TAG 로 OUT/LOG/BENCH_LOG 자동 결정)
  env "$@" bash bench/run_mtp1_ep_sweep.sh 2>&1 | tee "/workspace/logs/run_${name}.out"
  local rc=$?
  # 재현 로그를 결과 폴더에 동봉
  local out="/workspace/results/${name}_sweep"
  mkdir -p "$out"
  cp -f "/workspace/logs/run_${name}.out" "$out/run_output.txt" 2>/dev/null
  grep -a "non-default args" "/workspace/logs/vllm_${name}.log" 2>/dev/null | tail -1 > "$out/server_args.txt"
  {
    echo "invoked : env $* bash bench/run_mtp1_ep_sweep.sh"
    echo "date    : $(date -u +%FT%TZ)"
    echo "client  : bench_sweep_b200.py --sweep 4,8,16,32,64,128,256 --requests 1024 --max-tokens 1024 (ignore_eos=1 기본, temperature/top_p 미전송, thinking off)"
  } > "$out/config.txt"
  echo "########## [$name] 종료 rc=$rc $(date +%H:%M:%S) ##########"
  return $rc
}

run_one nomtp_noep NO_MTP=1 NO_EP=1
run_one mtp1_noep  NO_EP=1 SPEC_TOKENS=1
run_one mtp1_ep    SPEC_TOKENS=1
echo "CHAIN3_DONE"
