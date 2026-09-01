# B200 × GLM-5.2 추론 최적화 PoC

NVIDIA B200 (Blackwell sm_100) 8장에서 GLM-5.2 FP8 추론 성능 측정 및 최적화.

**측정값과 검증된 사실은 [FINDINGS.md](FINDINGS.md) 를 먼저 읽을 것.**
서버 인자 근거, 실패한 조합, 틀린 추론 기록이 모두 거기 있다.

## 서버 리셋 후 복원

빈 인스턴스에서 한 줄:

```bash
git clone https://github.com/crinex/b200_glm_poc.git /workspace/b200_glm_poc && \
cd /workspace/b200_glm_poc && bash setup/full_setup.sh
```

`full_setup.sh` 가 vLLM 설치 → torchaudio 제거 → 모델 704GB → gen8k 1,024장을
순서대로 처리하고, 이미 있는 단계는 건너뛴다. 세팅 후 바로 측정까지 하려면
`RUN_BENCH=1` 을 붙인다.

측정:

```bash
bash bench/run_mtp1_ep_sweep.sh                    # MTP spec-tokens 1 + EP8
NO_EP=1 bash bench/run_mtp1_ep_sweep.sh            # EP1 (expert parallel 끔)
SPEC_TOKENS=2 bash bench/run_mtp1_ep_sweep.sh      # MTP 2 토큰
EXTRA_ARGS="--gpu-memory-utilization 0.96" bash bench/run_mtp1_ep_sweep.sh
SKIP_BOOT=1 bash bench/run_mtp1_ep_sweep.sh        # 서버가 이미 떠 있으면
```

### 반드시 지킬 것

**서버를 SIGKILL 로 죽이지 말 것.** 죽이면 NCCL P2P transport 가 교착해
이후 **모든 기동이 실패**하고, 인스턴스 재시작 외에는 복구 방법이 없다.
벤치가 실패하면 서버는 살려두고 `SKIP_BOOT=1` 로 벤치만 재실행할 것.
자세한 증상과 소거된 원인은 [FINDINGS.md](FINDINGS.md) §0 참조.

정상 정리는 `bash serve/gpu_reset.sh` (SIGTERM 우선, 잔여물 정리 포함).

## 디렉토리 구조

```
setup/
  full_setup.sh         빈 인스턴스 → 측정 가능 상태까지 한 번에
  bootstrap.sh          레포 클론 + 환경 설치
  install.sh            vLLM 0.28.0 (버전 고정) + 의존성
  download_model.sh     GLM-5.2-FP8 704GB
serve/
  start_server.sh       vLLM 서버 (KV_DTYPE / MAX_BATCHED 등 환경변수)
  gpu_reset.sh          SIGTERM 우선 정리 + GPU/잔여물 해제
bench/
  run_mtp1_ep_sweep.sh  MTP·EP 측정 (SPEC_TOKENS / NO_EP / EXTRA_ARGS / SKIP_BOOT)
  run_mtp_conc256.sh    단일 concurrency 측정
  bench_sweep_b200.py   B200 벤치
  bench_sweep_h200.py   H200 원본 (참조)
  bench_sweep_mi355x.py MI355x/Moreh 원본 (참조)
  benchmark.py          단발 추론 테스트
  workload/
    build_gen8k.sh          gen8k 1,024장 생성
    generate_workload.py    LG U+ 방식 합성 프롬프트 생성기
    verify_answer.py        정답 키 채점 (벤치는 사용 안 함)
    README_gen8k_원본.md    원본 워크플로 문서
results/
  b200_baseline_h200cfg_sweep.md   기준선 (MTP·EP 미적용)
  b200_mtp1_ep8_sweep.md           MTP spec-tokens 1 + EP8
  b200_mtp1_ep8_serverlog.txt      해당 측정의 서버 로그 발췌
  b200_osl1024_kvfp8.md            초기 측정 (mnbt 8192, max-len 32768)
  b200_osl512_kvfp8.md             참고 (OSL 512 — 조건 불일치)
report/
  compare.html          B200 / H200 / Moreh 비교 페이지
  make_xlsx.py          비교표 Excel 생성
FINDINGS.md             검증된 사실 + 실패 기록 (먼저 읽을 것)
```

## 측정 결과 (B200 × 8, TP=8, 가중치 FP8 · KV cache FP8)

gen8k 1,024 고유 sheet · ISL ≈ 8,200 · OSL 1,024 · sweep 4~256

| conc | 기준선 Output TPS | MTP1+EP8 Output TPS | 개선 |
|---|---|---|---|
| 4 | 304 | 372 | +22.4% |
| 8 | 569 | 721 | +26.7% |
| 16 | 920 | 1,173 | +27.5% |
| 32 | 1,408 | 1,764 | +25.3% |
| 64 | 1,772 | 1,831 | +3.3% |
| 128 | 2,010 | **2,118** | +5.4% |
| 256 | 2,007 | 2,065 | +2.9% |

최고 처리량은 **conc=128, MTP1+EP8 에서 2,118 tok/s**.
낮은 concurrency 에서 MTP 효과가 크고 (+22~27%), 높은 구간에서 줄어든다 (+3~5%).
전체 수치는 `results/` 참조.

주의: Input TPS 는 Output TPS × (ISL/OSL) 과 오차 0.1% 내로 일치하며
prefill 성능 지표가 아니다 (FINDINGS.md §4).

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

측정 완료: 기준선, MTP1+EP8

미측정:

| 구성 | 목적 |
|---|---|
| `NO_EP=1 SPEC_TOKENS=1` | EP 기여도 분리 (MTP1+EP8 − MTP1+EP1 = EP 몫) |
| `SPEC_TOKENS=2` | accept length 2.76 vs 1.83 — 처리량으로 확인 |
| `EXTRA_ARGS="--gpu-memory-utilization 0.96"` | KV cache 증량 |
| `EXTRA_ARGS="--moe-backend DEEPGEMM"` | MoE 가 SM 140/148 점유 |

시도했으나 불가:

- `--disable-custom-all-reduce` — 기동 교착. 다만 이 인자가 원인이 아니라
  SIGKILL 후 NCCL P2P 교착이 원인이었다 (FINDINGS.md §0). 재검증 필요
- Expert Parallel 크기 지정 — vLLM 에 옵션이 없다. DP × TP 로 파생

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
