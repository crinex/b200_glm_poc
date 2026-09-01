#!/bin/bash
# =============================================================================
# 빈 인스턴스 → 측정 가능 상태까지 한 번에
#
# 사용:
#   bash setup/full_setup.sh              # 세팅만
#   RUN_BENCH=1 bash setup/full_setup.sh  # 세팅 후 기본 측정까지
#
# 소요 (실측, 2026-09-01 기준):
#   vLLM 설치      5~10분
#   모델 704GB     2.5분  (HF Xet 고속 전송. 링크 느리면 20~30분)
#   gen8k 1,024장  0.5분
#   서버 기동      15~25분 (첫 기동은 FlashInfer cubin 다운로드 포함)
#   sweep 4~256    20분
# =============================================================================
set -euo pipefail

REPO="${REPO:-/workspace/b200_glm_poc}"
LOGDIR="${LOGDIR:-/workspace/logs}"
GEN_DIR="${GEN_DIR:-/workspace/gen8k}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/GLM-5.2-FP8}"
RUN_BENCH="${RUN_BENCH:-0}"

mkdir -p "$LOGDIR"
cd "$REPO"

step() { echo ""; echo "=========== $* ==========="; }

# ── 1. vLLM ───────────────────────────────────────────────────────
step "1/4  vLLM 설치"
bash setup/install.sh 2>&1 | tee "$LOGDIR/install.log" | tail -12

# torchaudio 는 vLLM 이 끌어오는 torch(cu130) 와 CUDA 버전이 어긋나
# `import vllm` 자체를 깨뜨린다:
#   RuntimeError: Detected that PyTorch and TorchAudio were compiled with
#   different CUDA versions. PyTorch has CUDA 13.0 whereas TorchAudio has 12.8
# vLLM 은 torchaudio 를 쓰지 않으므로 제거한다.
step "2/4  torchaudio 제거 (cu128/cu130 충돌)"
PY="${PY:-/venv/main/bin/python3}"
"$PY" -m pip uninstall -y torchaudio --root-user-action=ignore 2>&1 | tail -3 || true
"$PY" -c 'import vllm, torch; print("  vllm", vllm.__version__,
    "| torch", torch.__version__, torch.version.cuda)'

# ── 2. 모델 ───────────────────────────────────────────────────────
step "3/4  모델"
n=$(find "$MODEL_DIR" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -eq 141 ]; then
    echo "  이미 존재 (141 safetensors, $(du -sh "$MODEL_DIR" | cut -f1))"
else
    bash setup/download_model.sh 2>&1 | tee "$LOGDIR/download.log" | tail -6
fi

# ── 3. gen8k ──────────────────────────────────────────────────────
step "4/4  gen8k 워크로드"
g=$(find "$GEN_DIR" -maxdepth 1 -name '*target*.txt' 2>/dev/null | wc -l | tr -d ' ')
if [ "$g" -ge 1024 ]; then
    echo "  이미 존재 ($g 개)"
else
    bash bench/workload/build_gen8k.sh "$GEN_DIR" 1024 2>&1 | tail -4
fi

cat <<EOF

=========== 세팅 완료 ===========
  모델  : $(find "$MODEL_DIR" -name '*.safetensors' | wc -l | tr -d ' ') safetensors
  gen8k : $(find "$GEN_DIR" -maxdepth 1 -name '*target*.txt' | wc -l | tr -d ' ') 개
  GPU   : $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')

  측정 (기본: MTP spec-tokens 1 + Expert Parallel):
    bash bench/run_mtp1_ep_sweep.sh

  변형:
    NO_EP=1 bash bench/run_mtp1_ep_sweep.sh        # EP1 (expert parallel 끔)
    SPEC_TOKENS=2 bash bench/run_mtp1_ep_sweep.sh  # MTP 2 토큰
    EXTRA_ARGS="--gpu-memory-utilization 0.96" bash bench/run_mtp1_ep_sweep.sh

  주의: 서버를 SIGKILL 로 여러 번 죽이면 NCCL P2P transport 가 교착해
        이후 모든 기동이 실패한다 (FINDINGS.md 참조). 인스턴스 재시작만이
        복구 방법이므로, 가능하면 정상 종료를 쓸 것.
=================================
EOF

if [ "$RUN_BENCH" = "1" ]; then
    step "측정 시작"
    bash bench/run_mtp1_ep_sweep.sh
fi
