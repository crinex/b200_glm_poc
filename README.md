# B200 × GLM-5.2 추론 최적화 PoC

NVIDIA B200 (Blackwell sm_100) 환경에서 GLM-5.2 FP8 모델 추론 최적화.

## 서버 리셋 후 복원 (매일)

```bash
# 원라이너 — 레포 클론 + 환경 설치 자동화
curl -fsSL https://raw.githubusercontent.com/crinex/b200_glm_poc/main/setup/bootstrap.sh | bash
```

## 디렉토리 구조

```
setup/
  bootstrap.sh       # 리셋 후 최초 실행 (레포 클론 + 환경 설치)
  install.sh         # vLLM + 의존성 설치
  download_model.sh  # GLM-5.2-FP8 모델 다운로드
serve/
  start_server.sh    # vLLM 서버 시작 (최적화 옵션 포함)
bench/
  benchmark.py       # 추론 벤치마크
kernels/             # 커스텀 커널 (Triton / Mojo)
```

## 환경

| 항목 | 버전 |
|---|---|
| GPU | NVIDIA B200 × 8 |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| vLLM | 0.28.0 |
| 모델 | zai-org/GLM-5.2-FP8 |

## 핵심 최적화 (vLLM)

- **FlashInfer Sparse MLA** — B200(sm_100)에서 자동 활성화
- **FlashAttention 3** — sm_100 자동 감지
- **MTP Speculative Decoding** — GLM-5.2 내장 헤드 활용 (accept length ~2.76)
- **FP8 KV Cache** — weight FP8와 별개로 추가 적용
- **Prefix Caching + Chunked Prefill**

## 커스텀 커널 계획

| 우선순위 | 커널 | 언어 |
|---|---|---|
| P0 | DSA + IndexShare fusion | Mojo |
| P0 | Sparse MLA (gather + attention) | Mojo |
| P0 | MoE router fusion | Triton |
| P0 | MoE dispatch/combine | Triton + CUDA |
| P1 | Grouped FP8 GEMM + SwiGLU | Mojo |
| P1 | MLA projection/KV fusion | Triton |
| P2 | Sampling fusion | Triton |
