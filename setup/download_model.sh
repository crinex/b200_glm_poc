#!/bin/bash
# =============================================================================
# GLM-5.2 FP8 모델 다운로드 (약 722GB, safetensors 141개)
# 사용: HF_TOKEN=hf_xxx bash setup/download_model.sh
#
# 주의: 맨 python3 는 /usr/bin/python3 로 잡힌다. venv 를 명시한다.
# =============================================================================
set -e

PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in /venv/main/bin/python3 /venv/bin/python3; do
        if [ -x "$c" ]; then PY="$c"; break; fi
    done
fi
[ -n "$PY" ] || { echo "venv 인터프리터 없음. PY= 로 지정." >&2; exit 1; }

MODEL_ID="${MODEL_ID:-zai-org/GLM-5.2-FP8}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"

echo "======================================"
echo "  GLM-5.2 FP8 다운로드"
echo "  소스   : $MODEL_ID"
echo "  저장   : $MODEL_DIR"
echo "  인터프리터: $PY"
echo "======================================"

mkdir -p "$MODEL_DIR"

# 디스크 여유 확인 (722GB + 여유)
avail=$(df -BG --output=avail "$(dirname "$MODEL_DIR")" | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -lt 800 ]; then
    echo "경고: 여유 공간 ${avail}GB. 모델이 722GB 이므로 800GB 이상 권장." >&2
fi

export HF_XET_HIGH_PERFORMANCE=1        # 고성능 전송 (huggingface_hub >= 1.28)
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

"$PY" - <<PYEOF
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$MODEL_DIR",
    resume_download=True,
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
print("완료:", path)
PYEOF

n=$(find "$MODEL_DIR" -name '*.safetensors' | wc -l | tr -d ' ')
echo ""
echo "safetensors: $n 개 (정상값 141)"
du -sh "$MODEL_DIR"
[ "$n" -eq 141 ] || { echo "미완료. 스크립트를 재실행하면 이어받습니다." >&2; exit 1; }
echo "다운로드 완료."
