#!/bin/bash
# =============================================================================
# gen8k 워크로드 생성 — 8k1k 벤치용 합성 프롬프트 1,024장
#
# 근거 (LG U+ 원본 워크플로 문서 = README_gen8k_원본.md):
#   - 총 1,024장. bench 의 --requests 기본값이 1024 이고, 그보다 적으면
#     sheet 가 재사용되어 prefix cache 에 히트해 Input TPS 가 부풀려진다.
#     (200장으로 돌렸다가 conc>=64 구간 Input TPS 가 7~8% 과대 측정됨)
#   - --n-uuid 37 --n-garbage 27 이 8k 토큰 예산에 맞춘 값이며
#     generate_workload.py 의 기본값이다. generate() 내부 기본값(90/65)은
#     ~15k 토큰이 되어 컨텍스트 초과로 HTTP 400 이 난다.
#   - GLM tokenizer 실측: 17,430 bytes → 8,211 토큰
#
# 사용: bash build_gen8k.sh [출력디렉터리] [개수]
# =============================================================================
set -euo pipefail

OUT="${1:-/workspace/gen8k}"
N="${2:-1024}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/venv/main/bin/python3}"
[ -x "$PY" ] || PY=python3

mkdir -p "$OUT"

echo "gen8k 생성: $OUT (seed 1~$N)"
made=0
for s in $(seq 1 "$N"); do
    prefix="s${s}_"
    if [ -f "$OUT/${prefix}target.txt" ]; then
        continue
    fi
    "$PY" "$HERE/generate_workload.py" \
        --seed "$s" \
        --out-dir "$OUT" \
        --prefix "$prefix" \
        --n-uuid 37 --n-garbage 27 > /dev/null
    made=$((made + 1))
    if [ $((s % 100)) -eq 0 ]; then
        echo "  $s/$N"
    fi
done

total=$(find "$OUT" -maxdepth 1 -name '*target*.txt' | wc -l | tr -d ' ')
echo "완료: 신규 $made 개, target 파일 총 $total 개"

first=$(find "$OUT" -maxdepth 1 -name '*target*.txt' | sort | head -1)
if [ -n "$first" ]; then
    echo "샘플: $(basename "$first") $(wc -c < "$first" | tr -d ' ') bytes (약 8,200 토큰)"
fi

if [ "$total" -lt 1024 ]; then
    echo "경고: 1,024 미만이면 --requests 1024 에서 sheet 재사용 → prefix cache 오염" >&2
fi
