# B200 × GLM-5.2 추론 최적화 PoC

NVIDIA B200 (Blackwell sm_100) 8장에서 GLM-5.2 FP8 추론 성능 측정 및 최적화.

**측정값과 검증된 사실은 [FINDINGS.md](FINDINGS.md) 를 먼저 읽을 것.**
서버 인자 근거, 실패한 조합, 틀린 추론 기록이 모두 거기 있다.

## 서버 리셋 후 복원

```bash
# 1. 레포 + 환경
curl -fsSL https://raw.githubusercontent.com/crinex/b200_glm_poc/main/setup/bootstrap.sh | bash

# 2. 모델 (722GB)
HF_TOKEN=hf_xxx bash setup/download_model.sh

# 3. gen8k 워크로드 1,024장
bash bench/workload/build_gen8k.sh /workspace/gen8k 1024

# 4. 서버
bash serve/start_server.sh

# 5. 벤치
/venv/main/bin/python3 bench/bench_sweep_b200.py \
  --host localhost --port 8000 --model glm-5.2 \
  --gen-dir /workspace/gen8k \
  --sweep 4,8,16,32,64,128,256 \
  --max-tokens 1024 \
  --out /workspace/results/run1
```

서버 재기동 전에는 `bash serve/gpu_reset.sh` 로 GPU 를 비울 것.
`pkill -f vllm` 은 워커를 못 잡아 173GB/GPU 가 남는다.

## 디렉토리 구조

```
setup/
  bootstrap.sh          리셋 후 최초 실행 (레포 클론 + 환경 설치)
  install.sh            vLLM + 의존성
  download_model.sh     GLM-5.2-FP8 다운로드
serve/
  start_server.sh       vLLM 서버 (실측 검증 구성)
  gpu_reset.sh          GPU 메모리 강제 해제
bench/
  bench_sweep_b200.py   B200 벤치 (concurrency sweep)
  bench_sweep_h200.py   H200 원본 (참조용)
  bench_sweep_mi355x.py MI355x/Moreh 원본 (참조용)
  benchmark.py          단발 추론 테스트
  workload/
    build_gen8k.sh          gen8k 1,024장 생성
    generate_workload.py    LG U+ 방식 합성 프롬프트 생성기
    verify_answer.py        정답 키 채점 (벤치는 사용 안 함)
    README_gen8k_원본.md    원본 워크플로 문서
results/
  b200_osl1024_kvfp8.md   기준 측정 (OSL 1024)
  b200_osl512_kvfp8.md    참고 (OSL 512, 문서화 조건과 불일치)
report/
  make_xlsx.py            비교표 Excel 생성
  compare.html            B200/H200/Moreh 비교 페이지
FINDINGS.md               검증된 사실 + 실패 기록
```

## 환경 (실측)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 × 8, sm_100, 178.34 GiB/GPU |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.28.0 |
| 모델 | `zai-org/GLM-5.2-FP8` (141 shards, 722GB) |
| 인터프리터 | `/venv/main/bin/python3` |

## 측정 결과 (B200 × 8, TP=8, KV cache fp8, MTP 미적용)

ISL ~8,200 / OSL 1,024 / gen8k 1,024장

| conc | TPOT(ms) | Interactivity | Input TPS | Output TPS |
|---|---|---|---|---|
| 4 | 11.48 | 87.09 | 2,660 | 333 |
| 16 | 15.97 | 62.60 | 7,523 | 940 |
| 32 | 20.70 | 48.32 | 11,724 | 1,465 |
| 64 | 35.57 | 28.11 | 13,883 | 1,734 |
| 128 | 62.69 | 15.95 | 15,810 | 1,975 |
| 256 | 83.75 | 11.94 | 16,138 | 2,016 |

전체는 `results/b200_osl1024_kvfp8.md`.

## 적용된 최적화

| 항목 | 상태 |
|---|---|
| FlashInfer Sparse MLA (`FLASHINFER_MLA_SPARSE`) | 자동 활성 (sm_100) |
| DEEPSEEK_V32_INDEXER (sparse attention) | 자동 활성, KV block size 64 |
| FP8 KV Cache | **적용** — 디폴트 아님. 명시 필요 |
| Prefix Caching + Chunked Prefill | V1 엔진 자동 활성 |
| MTP Speculative Decoding | **미적용** |

## 다음 작업

1. `mnbt` 상향 재측정 — `MAX_BATCHED=32768 bash serve/start_server.sh`
   현재 8192 가 ISL 8,200 보다 작아 고concurrency 병목 의심 (FINDINGS.md §7)
2. MTP speculative decoding 적용
3. bf16 KV cache 대조 측정 (디폴트 baseline)
4. 커스텀 커널

### 커스텀 커널 계획

| 우선순위 | 커널 | 언어 |
|---|---|---|
| P0 | DSA + IndexShare fusion | Mojo |
| P0 | Sparse MLA (gather + attention) | Mojo |
| P0 | MoE router fusion | Triton |
| P0 | MoE dispatch/combine | Triton + CUDA |
| P1 | Grouped FP8 GEMM + SwiGLU | Mojo |
| P1 | MLA projection/KV fusion | Triton |
| P2 | Sampling fusion | Triton |
