#!/bin/bash
# =============================================================================
# 지정 서빙 구성 + MTP(spec-tokens 1) + Expert Parallel, 전 구간 sweep
#
# 기준 서빙 옵션은 전달받은 H200 구성이며 다음을 조정/추가했다:
#   - 모델 경로: 현재 서버 경로로 변경
#   - --ignore-eos 1 제외: vllm serve 인자가 아니다 (argparse 386개 중 eos 0개).
#     벤치가 요청별로 ignore_eos=1 을 보내므로 동작은 동일하고 OSL 이 고정된다.
#   - --trust-remote-code 추가: GLM 커스텀 모델링 코드 로드에 필요
#   - --spec-method mtp --spec-tokens 1 추가
#     spec-tokens 2 는 MTP 레이어가 1개뿐인 이 모델에서 같은 레이어를 두 번
#     forward 하므로 vLLM 이 수락률 저하를 경고한다. 실측 수락률은
#     position1 93.8% / position2 88.1% (accept length 2.76) 였다.
#     1 은 accept length 1.94 지만 draft 오버헤드와 KV cache 소모가 작다.
#   - --enable-expert-parallel 추가
#     기본값은 비활성. 켜지면 전문가 256개가 GPU 8장에 32개씩 통째로 배분된다.
#     판정 로그: "[EP Rank 0/8] Expert parallelism is enabled.
#                 Local/global number of experts: 32/256"
#     주의: MoEPrepareAndFinalizeNoDPEPMonolithic 은 EP 지표가 아니다.
#     EP 를 켜도 끄도 동일하게 출력된다 (NoDP + Monolithic 조건).
#
# sweep 은 4,8,16,32,64,128,256 — 원본 워크플로 문서 명령이자
# H200_GLM5.2_Measure.pdf 표의 concurrency 와 일치한다.
# 각 레벨의 요청 수는 conc*4 (--requests 1024 상한): 16/32/64/128/256/512/1024.
#
# 사용:
#   bash bench/run_mtp1_ep_sweep.sh
#   SKIP_BOOT=1 bash bench/run_mtp1_ep_sweep.sh      # 이미 떠 있으면 측정만
#   SPEC_TOKENS=2 bash bench/run_mtp1_ep_sweep.sh    # MTP 토큰 수 변경
#   NO_EP=1 bash bench/run_mtp1_ep_sweep.sh          # EP 끄고 비교
#   SWEEP=64,256 bash bench/run_mtp1_ep_sweep.sh     # 축소 측정
#   EXTRA_ARGS="--disable-custom-all-reduce" bash bench/run_mtp1_ep_sweep.sh
#                                                    # 임의 vllm serve 인자 추가
# =============================================================================
set -uo pipefail

PY="${PY:-/venv/main/bin/python3}"
VLLM="${VLLM:-/venv/main/bin/vllm}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"
SERVED="${SERVED:-glm-5.2-fp8}"
PORT="${PORT:-30269}"
GEN_DIR="${GEN_DIR:-/workspace/gen8k}"
SWEEP="${SWEEP:-4,8,16,32,64,128,256}"
REQUESTS="${REQUESTS:-1024}"
MAX_TOKENS="${MAX_TOKENS:-1024}"       # OSL. "8k1k 벤치" 문서 조건
SPEC_TOKENS="${SPEC_TOKENS:-1}"
NO_EP="${NO_EP:-0}"
REPO="${REPO:-/workspace/b200_glm_poc}"
SKIP_BOOT="${SKIP_BOOT:-0}"
# 추가 vllm serve 인자 (공백 구분). 예: EXTRA_ARGS="--disable-custom-all-reduce"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# 결과 디렉터리 접미사. 미지정 시 EXTRA_ARGS 에서 자동 생성
TAG_SUFFIX="${TAG_SUFFIX:-}"

TAG="mtp${SPEC_TOKENS}$([ "$NO_EP" = "1" ] && echo "_noep" || echo "_ep")"
if [ -n "$EXTRA_ARGS" ]; then
    if [ -n "$TAG_SUFFIX" ]; then
        TAG="${TAG}_${TAG_SUFFIX}"
    else
        # "--disable-custom-all-reduce" -> "disable_custom_all_reduce"
        TAG="${TAG}_$(echo "$EXTRA_ARGS" | tr -d '\n' | sed 's/--//g; s/[^a-zA-Z0-9]\+/_/g; s/^_//; s/_$//')"
    fi
fi
OUT="${OUT:-/workspace/results/${TAG}_sweep}"
LOG="${LOG:-/workspace/logs/vllm_${TAG}.log}"
BENCH_LOG="${BENCH_LOG:-/workspace/logs/bench_${TAG}.log}"

mkdir -p "$(dirname "$LOG")" "$OUT"

echo "======================================"
echo "  구성 : MTP spec-tokens=$SPEC_TOKENS, EP=$([ "$NO_EP" = "1" ] && echo off || echo on)"
echo "  추가 : ${EXTRA_ARGS:-(없음)}"
echo "  sweep: $SWEEP"
echo "  OSL  : $MAX_TOKENS"
echo "  결과 : $OUT"
echo "======================================"

# ── 1. 기동 ───────────────────────────────────────────────────────
if [ "$SKIP_BOOT" != "1" ]; then
    echo ""
    echo "=== GPU 정리 ==="
    # pkill -f vllm 은 워커를 못 잡는다. nvidia-smi 점유 PID 를 직접 죽인다.
    bash "$REPO/serve/gpu_reset.sh" || exit 1

    ARGS=(
        serve "$MODEL_DIR"
        --host 0.0.0.0
        --port "$PORT"
        --tensor-parallel-size 8
        --async-scheduling
        --kv-cache-dtype fp8
        --served-model-name "$SERVED"
        --max-model-len 16384
        --max-num-seqs 256
        --gpu-memory-utilization 0.93
        --enable-chunked-prefill
        --enable-prefix-caching
        --trust-remote-code
        --spec-method mtp
        --spec-tokens "$SPEC_TOKENS"
    )
    [ "$NO_EP" = "1" ] || ARGS+=(--enable-expert-parallel)
    # shellcheck disable=SC2206  # 공백 구분 인자를 그대로 펼친다
    [ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

    echo ""
    echo "=== 서버 기동 ==="
    printf '  %s\n' "${ARGS[*]}"
    : > "$LOG"
    setsid nohup "$VLLM" "${ARGS[@]}" > "$LOG" 2>&1 &
    SERVER_PID=$!
    echo "  PID $SERVER_PID"

    # 생존 판정은 실제 PID 로 한다.
    # pgrep -f 'openai.api_server' 는 기동 초기에 매칭되지 않는다.
    # 그 시점 cmdline 은 'vllm serve ...' 뿐이므로 살아있는 서버를 죽었다고
    # 오판해 스크립트가 종료된다 (2026-09-01 에 실제로 겪음).
    alive() {
        kill -0 "$SERVER_PID" 2>/dev/null && return 0
        pgrep -f 'vllm serve|openai\.api_server|VLLM::' > /dev/null && return 0
        return 1
    }

    echo "기동 대기 (최대 35분)..."
    for i in $(seq 1 105); do
        sleep 20
        if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
            echo "기동 완료 (약 $((i * 20))초)"
            break
        fi
        if grep -qE "AssertionError|Engine core initialization failed|CUDA error|out of memory|unrecognized arguments|error: argument" "$LOG" 2>/dev/null; then
            echo "기동 실패:" >&2
            grep -E -m1 -A8 "AssertionError|Engine core initialization failed|CUDA error|out of memory|unrecognized arguments|error: argument" "$LOG" >&2
            exit 1
        fi
        if ! alive; then
            echo "프로세스 사망. 로그 마지막 20줄:" >&2
            tail -20 "$LOG" >&2
            exit 1
        fi
        # 진행 표시 (조용히 오래 걸리는 구간 구분)
        stage=$(grep -aoE 'Loading safetensors checkpoint shards: +[0-9]+%|DeepGEMM warmup|Capturing CUDA graphs|GPU KV cache size|cubin_loader' "$LOG" 2>/dev/null | tail -1)
        [ -n "$stage" ] && echo "  ... $((i * 20))s  $stage"
    done
    grep -q "Application startup complete" "$LOG" || { echo "기동 타임아웃" >&2; exit 1; }
fi

# ── 2. 기동 정보 ──────────────────────────────────────────────────
echo ""
echo "=== 서버 정보 ==="
grep -a "GPU KV cache size" "$LOG" 2>/dev/null | tail -1
grep -a "max_num_batched_tokens" "$LOG" 2>/dev/null | tail -1
grep -a "non-default args" "$LOG" 2>/dev/null | tail -1 \
    | tr ',' '\n' | grep -E "spec_|expert|kv_cache|max_num_seqs|gpu_memory|max_model_len|async" || true
grep -a "MoEPrepareAndFinalize" "$LOG" 2>/dev/null | tail -1
grep -a "MoE backend" "$LOG" 2>/dev/null | tail -1
# EP 실제 활성 여부는 이 라인으로 판정한다 (NoDPEP 문자열은 지표가 아님)
grep -a "Expert parallelism is enabled" "$LOG" 2>/dev/null | tail -1
# all-reduce 백엔드 선택 (--disable-custom-all-reduce 효과 확인용)
grep -a "all-reduce backends" "$LOG" 2>/dev/null | head -2
curl -s "http://localhost:$PORT/v1/models" | head -c 200; echo

# ── 3. 벤치 ───────────────────────────────────────────────────────
echo ""
echo "=== 벤치 sweep=$SWEEP, requests=$REQUESTS, max_tokens=$MAX_TOKENS ==="
"$PY" -u "$REPO/bench/bench_sweep_b200.py" \
    --host localhost --port "$PORT" --model "$SERVED" \
    --device B200 --tp 8 --gpus 8 --precision FP8 \
    --gen-dir "$GEN_DIR" \
    --sweep "$SWEEP" \
    --requests "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --out "$OUT" 2>&1 | tee "$BENCH_LOG" | grep -vE "req/s.*%\|"

# ── 4. 결과 ───────────────────────────────────────────────────────
echo ""
echo "=== 결과 ==="
cat "$OUT/sweep_summary.md" 2>/dev/null || ls -la "$OUT"

echo ""
echo "=== MTP 수락률 (서버 로그) ==="
grep -a "acceptance rate" "$LOG" 2>/dev/null | tail -3

echo ""
echo "결과: $OUT/sweep_summary.md"
echo "벤치 로그: $BENCH_LOG"
echo "서버 로그: $LOG"
