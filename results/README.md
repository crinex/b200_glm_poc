# 측정 결과 인덱스

모든 측정: gen8k 1,024 고유 sheet · ISL ≈ 8,200 · OSL 1,024 · sweep 4~256
· 가중치 FP8 · KV cache fp8 · TP=8 · vLLM 0.28.0
서빙 인자는 FINDINGS.md §6-1 의 지정 구성.

**서버가 다르면 수치를 직접 비교하지 말 것.** 같은 구성이 서버 간 −16% 차이났다
(FINDINGS.md §6-2).

## 서버 C — deploygpu (드라이버 570 + cuda-compat, 2026-09-01)

동일 서버 3구성. 분해가 유효한 세트.

| 파일 | 구성 | conc=128 Output | conc=256 Output |
|---|---|---|---|
| `b200_deploygpu_baseline_sweep.md` | MTP·EP 없음 | 1,854 | 1,843 |
| `b200_mtp1_ep1_sweep.md` (+serverlog) | **MTP1, EP 끔 — 최적** | **2,013** | 1,994 |
| `b200_mtp1_ep8_run2_sweep.md` | MTP1, EP8 | 1,783 | 1,729 |

분해: MTP +8~35%, EP8 은 고concurrency −11~13%.

## 서버 B — Vast.ai (드라이버 595, 2026-09-01, 소멸)

| 파일 | 구성 | 비고 |
|---|---|---|
| `b200_baseline_h200cfg_sweep.md` | MTP·EP 없음 | conc=128: 2,010 |
| `b200_mtp1_ep8_sweep.md` (+serverlog) | MTP1, EP8 | conc=128: 2,118 |

서버 C 보다 전반적으로 8~16% 빠른 인스턴스였다. 서버 C 결과와 섞어 분해하면
정반대 결론이 나온다 — 실제로 그랬다.

## 서버 A — Vast.ai (2026-08-31, 소멸)

서빙 구성이 다르다 (max-model-len 32768, mnbt 8192, max-num-seqs 미지정).

| 파일 | 구성 | 비고 |
|---|---|---|
| `b200_osl1024_kvfp8.md` | MTP·EP 없음 | mnbt 8192 가설 검증에 사용 (기각됨) |
| `b200_osl512_kvfp8.md` | 〃, OSL 512 | 문서화 조건(OSL 1024)과 불일치 — 참고용 |

## 비교 대상 (외부)

H200 (MTP 유/무), Moreh MI355X — `H200_GLM5.2_Measure.pdf` 인용.
서버 구성·ISL/OSL 미기재. Input/Output 비 8:1 로 ISL:OSL=8:1 은 역산 확인됨.
Moreh 는 원본 스크립트 기본값이 MXFP4 라 "FP8" 표기와 상충.
