#!/bin/bash
# =============================================================================
# 지정 서빙 구성 + MTP, concurrency 256 단일 측정
#
# 서빙 옵션은 전달받은 H200 구성 그대로이며 두 가지만 조정했다:
#   - 모델 경로를 현재 서버에 맞게 변경
#   - --ignore-eos 1 제외: vllm serve 인자가 아니다 (argparse 386개 옵션 중
#     eos 포함 0개). 벤치가 요청별로 ignore_eos=1 을 전송하므로 동작은 동일하고
#     OSL 이 정확히 --max-tokens 로 고정된다.
#   - --trust-remote-code 추가: GLM 커스텀 모델링 코드 로드에 필요
#
# MTP 는 --spec-method / --spec-tokens 로 준다. --speculative-config JSON 은
# bash 인용이 깨지기 쉽다.
#
# 사용:
#   bash bench/run_mtp_conc256.sh                 # 기동 + 측정
#   SKIP_BOOT=1 bash bench/run_mtp_conc256.sh     # 이미 떠 있으면 측정만
#   SPEC_TOKENS=1 bash bench/run_mtp_conc256.sh   # MTP 토큰 수 변경
# =============================================================================
set -uo pipefail

PY="${PY:-/venv/main/bin/python3}"
VLLM="${VLLM:-/venv/main/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"
SERVED="${SERVED:-glm-5.2-fp8}"
PORT="${PORT:-30269}"
GEN_DIR="${GEN_DIR:-/workspace/gen8k}"
CONC="${CONC:-256}"
REQUESTS="${REQUESTS:-1024}"      # conc 단일 실행이면 --requests 전량 전송
MAX_TOKENS="${MAX_TOKENS:-1024}"  # OSL. "8k1k 벤치" 문서 조건
SPEC_TOKENS="${SPEC_TOKENS:-2}"
OUT="${OUT:-/workspace/results/mtp_conc${CONC}}"
LOG="${LOG:-/workspace/logs/vllm_mtp.log}"
REPO="${REPO:-/workspace/b200_glm_poc}"
SKIP_BOOT="${SKIP_BOOT:-0}"

mkdir -p "$(dirname "$LOG")" "$OUT"

# ── 1. 기동 ───────────────────────────────────────────────────────
if [ "$SKIP_BOOT" != "1" ]; then
    echo "=== GPU 정리 ==="
    # pkill -f vllm 은 워커를 못 잡는다. nvidia-smi 점유 PID 를 직접 죽인다.
    bash "$REPO/serve/gpu_reset.sh" || exit 1

    echo "=== 서버 기동 (MTP spec_tokens=$SPEC_TOKENS) ==="
    : > "$LOG"
    setsid nohup "$VLLM" serve "$MODEL_DIR" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size 8 \
        --async-scheduling \
        --kv-cache-dtype fp8 \
        --served-model-name "$SERVED" \
        --max-model-len 16384 \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.93 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --trust-remote-code \
        --spec-method mtp \
        --spec-tokens "$SPEC_TOKENS" \
        > "$LOG" 2>&1 &

    echo "기동 대기 (최대 35분)..."
    for i in $(seq 1 105); do
        sleep 20
        if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
            echo "기동 완료 ($((i * 20))초)"
            break
        fi
        if grep -qE "AssertionError|Engine core initialization failed|CUDA error|out of memory|unrecognized arguments|error: argument" "$LOG" 2>/dev/null; then
            echo "기동 실패:" >&2
            grep -E -m1 -A6 "AssertionError|Engine core initialization failed|CUDA error|out of memory|unrecognized arguments|error: argument" "$LOG" >&2
            exit 1
        fi
        if ! pgrep -f 'openai.api_server' > /dev/null; then
            echo "프로세스 사망. 로그 마지막:" >&2
            tail -20 "$LOG" >&2
            exit 1
        fi
    done
    grep -q "Application startup complete" "$LOG" || { echo "기동 타임아웃" >&2; exit 1; }
fi

# ── 2. 기동 정보 ──────────────────────────────────────────────────
echo ""
echo "=== 서버 정보 ==="
grep -a "GPU KV cache size" "$LOG" 2>/dev/null | tail -1
grep -a "max_num_batched_tokens" "$LOG" 2>/dev/null | tail -1
grep -a "spec_method" "$LOG" 2>/dev/null | tail -1 | tr ',' '\n' | grep -E "spec_" || true
curl -s "http://localhost:$PORT/v1/models" | head -c 200; echo

# ── 3. 벤치 (conc 단일) ───────────────────────────────────────────
# --sweep 없이 --conc 를 주면 --requests 전량을 보낸다 (wave 축소 없음).
echo ""
echo "=== 벤치 conc=$CONC, requests=$REQUESTS, max_tokens=$MAX_TOKENS ==="
"$PY" -u "$REPO/bench/bench_sweep_b200.py" \
    --host localhost --port "$PORT" --model "$SERVED" \
    --device B200 --tp 8 --gpus 8 --precision FP8 \
    --gen-dir "$GEN_DIR" \
    --conc "$CONC" \
    --requests "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --out "$OUT" 2>&1 | grep -vE "req/s.*%\|"

echo ""
echo "=== 결과 ==="
"$PY" - "$OUT" <<'PYEOF'
import json, os, sys
p = os.path.join(sys.argv[1], 'result.json')
if not os.path.exists(p):
    print("result.json 없음:", p); raise SystemExit(1)
m = json.load(open(p))['metrics']
print("  ISL            : %.0f" % m['mean_isl'])
print("  OSL            : %.0f" % m['mean_osl'])
print("  concurrency    : %d" % m['concurrency'])
print("  완료/실패      : %d / %d" % (m['completed'], m['failed']))
print("  TPOT (ms)      : %.2f" % m['median_tpot_ms'])
print("  Interactivity  : %.2f tok/s/user" % m['interactivity_median'])
print("  Input TPS      : %.0f /server" % m['input_throughput'])
print("  Output TPS     : %.0f /server" % m['output_throughput'])
print("  Total TPS      : %.0f /server" % m['total_throughput'])
PYEOF
echo ""
echo "저장 위치: $OUT"
