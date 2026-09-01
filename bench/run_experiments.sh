#!/bin/bash
# =============================================================================
# experiments.conf 의 실험들을 위에서부터 순차 실행
#
# 사용:
#   bash bench/run_experiments.sh                 # bench/experiments.conf 사용
#   bash bench/run_experiments.sh my.conf         # 다른 설정 파일
#
# 백그라운드 권장:
#   setsid nohup bash bench/run_experiments.sh \
#     > /workspace/logs/experiments.out 2>&1 &
#
# 실험마다 남는 것 (/workspace/results/<이름>_sweep/):
#   sweep_summary.md   결과 표
#   server_args.txt    vLLM 이 받은 실제 서빙 인자
#   config.txt         실행 옵션 + 클라이언트 요청 설정
#   run_output.txt     기동~벤치 전체 출력
#   fingerprint.txt    하드웨어 지문
# 마지막에 전체 요약: /workspace/results/experiments_summary.md
# =============================================================================
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONF="${1:-$REPO/bench/experiments.conf}"
LOGDIR="${LOGDIR:-/workspace/logs}"
RESULTS="${RESULTS:-/workspace/results}"
mkdir -p "$LOGDIR" "$RESULTS"

[ -f "$CONF" ] || { echo "설정 파일 없음: $CONF" >&2; exit 1; }

echo "======================================"
echo "  실험 설정 : $CONF"
echo "  시작      : $(date +%F' '%T)"
echo "======================================"

PASS=(); FAIL=()

while IFS= read -r line || [ -n "$line" ]; do
    # 주석/빈 줄 제거
    line="${line%%#*}"
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$line" ] && continue

    name="${line%%[[:space:]]*}"
    opts="${line#"$name"}"
    if ! echo "$name" | grep -qE '^[A-Za-z0-9_]+$'; then
        echo "!! 잘못된 실험 이름 '$name' — 건너뜀 (영문/숫자/밑줄만)"
        FAIL+=("$name(이름오류)"); continue
    fi

    OUT="$RESULTS/${name}_sweep"
    echo ""
    echo "########## [$name] 시작 $(date +%H:%M:%S)  옵션:${opts:- (기본값)} ##########"

    # EXTRA_ARGS 의 따옴표를 살리기 위해 eval 로 env 를 구성한다
    eval "env $opts \
        OUT='$OUT' \
        LOG='$LOGDIR/vllm_${name}.log' \
        BENCH_LOG='$LOGDIR/bench_${name}.log' \
        bash '$REPO/bench/run_mtp1_ep_sweep.sh'" 2>&1 \
        | tee "$LOGDIR/run_${name}.out"
    rc=${PIPESTATUS[0]}

    # 재현 로그 동봉
    mkdir -p "$OUT"
    cp -f "$LOGDIR/run_${name}.out" "$OUT/run_output.txt" 2>/dev/null
    grep -a "non-default args" "$LOGDIR/vllm_${name}.log" 2>/dev/null \
        | tail -1 > "$OUT/server_args.txt"
    {
        echo "name    : $name"
        echo "invoked : env $opts bash bench/run_mtp1_ep_sweep.sh"
        echo "date    : $(date -u +%FT%TZ)"
        echo "client  : bench_sweep_b200.py (sweep/requests/max-tokens 는 위 옵션 또는 기본값,"
        echo "          ignore_eos=1, temperature/top_p 미전송, thinking off)"
    } > "$OUT/config.txt"

    if [ "$rc" -eq 0 ]; then
        echo "########## [$name] 성공 $(date +%H:%M:%S) ##########"
        PASS+=("$name")
    else
        echo "########## [$name] 실패 rc=$rc — 다음 실험으로 진행 ##########"
        FAIL+=("$name")
    fi
done < "$CONF"

# ── 전체 요약 (4개 지표만) ─────────────────────────────────────────
SUM="$RESULTS/experiments_summary.md"
{
    echo "# 실험 요약 ($(date +%F' '%T))"
    echo ""
    for name in "${PASS[@]}"; do
        f="$RESULTS/${name}_sweep/sweep_summary.md"
        [ -f "$f" ] || continue
        echo "## $name"
        echo ""
        echo "| conc | TPOT(ms) | Interactivity | Input TPS/server | Output TPS/server |"
        echo "|---|---|---|---|---|"
        grep -E '^\|[0-9]' "$f" | awk -F'|' '{printf "| %s | %s | %s | %s | %s |\n", $9, $10, $11, $15, $16}'
        echo ""
    done
    [ ${#FAIL[@]} -gt 0 ] && echo "실패: ${FAIL[*]}"
} > "$SUM"

echo ""
echo "======================================"
echo "  성공: ${PASS[*]:-없음}"
echo "  실패: ${FAIL[*]:-없음}"
echo "  요약: $SUM"
echo "======================================"
cat "$SUM"
echo "EXPERIMENTS_DONE"
[ ${#FAIL[@]} -eq 0 ]
