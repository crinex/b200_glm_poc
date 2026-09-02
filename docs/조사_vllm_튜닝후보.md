# 조사 — TP=8 · EP=1 · MTP(spec 1~2) 고정 하에서 vLLM 옵션 튜닝 후보

목표: 위 고정 조건에서 **vLLM 옵션만으로** 최대 성능(Output TPS)·효율을 낸다.
기준 구성 = 서버 D 실측 최적 (MTP1+EP1, conc=128 에서 2,251 tok/s).

각 후보의 근거 표기:
- **[로그 확인]** 이번 세션에서 argparse 덤프 또는 엔진 로그로 존재·값을 확인함
- **[요검증]** 이름/의미를 서버에서 재확인 후 실행 (기동 시 무효 인자는 즉시 실패하므로 안전)

---

## 0. 우리가 아는 병목 구조 (실측)

| 관찰 | 수치 | 시사점 |
|---|---|---|
| KV cache 상주 한계 | 1,390,848 tok ÷ 9,224 tok/req ≈ **150개** | conc≥150 은 큐잉. KV 를 늘리면 포화점 상승 |
| decode-bound | prefill 여력 24.5k tok/s vs 소비 ~2.2k×8 | prefill 쪽 옵션은 후순위 |
| MoE 가 SM 독점 | TRT-LLM MoE 가 **SM 140/148** | MoE 커널 백엔드가 최대 소비처 |
| MTP 이득 감쇠 | conc 4→256 에서 +38%→+8% | 고conc 에서 draft 비용·KV 잠식이 상쇄 |
| attention backend | `FLASHINFER_MLA_SPARSE` 자동 선택 | 대안 백엔드 존재 |
| all-reduce | `['CUSTOM','SYMM_MEM','PYNCCL']`, 후보에 `FLASHINFER` 도 있음 | 통신 백엔드 교체 여지 |
| prefix cache hit 17~25% | sweep 레벨 간 sheet 재사용으로 히트 발생 | **측정 오염 가능성 — 별도 §5** |

---

## 1. 후보 A — Speculative (핵심 축, 사용자 지정 범위)

### A1. `--spec-tokens 2` **[로그 확인]** — 최우선
- 근거: spec=1 실측 accept length 1.75 / spec=2 실측(서버 B) 2.76
- 기대: 저~중 conc 에서 추가 이득. 단 GLM 은 MTP 레이어 1개를 재사용하므로
  draft 비용 2배 + KV 잠식 증가 → 고conc 에서 역전 가능성. **실측만이 답**
- 리스크: 낮음 | conf: `mtp2_ep1  NO_EP=1 SPEC_TOKENS=2`

### A2. spec 토큰 수별 KV/성능 트레이드오프 지도 (1 vs 2, conc 축 교차)
- A1 결과에 따라 "저conc 는 2, 고conc 는 1" 같은 운영 정책 도출 가능
- vLLM 에 배치 크기 기준 spec 자동 비활성 옵션이 있으면 활용 **[요검증]**
  (과거 버전의 `--speculative-disable-by-batch-size` 계열)

---

## 2. 후보 B — 메모리/KV (포화점 끌어올리기)

### B1. `--gpu-memory-utilization 0.93 → 0.95` **[로그 확인]**
- 근거: KV +2%p ≈ 상주 +3~4개. conc 128~256 대기 감소
- 리스크: CUDA graph 캡처 단계 OOM (기동 실패로 즉시 판정됨)
- conf: `kvup  NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--gpu-memory-utilization 0.95"`

### B2. `--no-enable-prefix-caching` **[로그 확인]**
- 근거: gen8k 1,024장은 공유 프리픽스가 없다. APC 는 해시·블록 관리 오버헤드만
  얹을 가능성 + §5 의 측정 오염 제거 효과
- 기대: 소폭(±1~3%) — 그러나 측정 순수성 확보 가치가 큼
- conf: `noapc  NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--no-enable-prefix-caching"`

### B3. `--max-num-seqs 256 → 128 / 384` **[로그 확인]**
- 근거: KV 상주 한계(~150)보다 상한(256)이 커서 초과 admission → 선점/스래싱
  가능성. 128 로 내리면 conc=256 에서 안정, 384 는 반대 방향 대조군
- conf: `seq128  NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--max-num-seqs 128"`

### B4. `--kv-cache-dtype fp8_ds_mla` + `--attention-backend FLASHMLA_SPARSE` **[로그 확인]**
- 근거: choices 에 존재. DeepSeek 전용 656B 패킹 레이아웃 — 여전히 FP8 계열.
  KV 용량·attention 커널이 동시에 바뀌는 **묶음 변경**
- 기대: 미지 (용량↑ 가능, 커널 성능은 실측 필요) | 리스크: 기동 실패 가능
- conf: `dsmla  NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--kv-cache-dtype fp8_ds_mla --attention-backend FLASHMLA_SPARSE"`

---

## 3. 후보 C — MoE 백엔드 (최대 SM 소비처)

### C1. `--moe-backend` 교체 **[로그 확인]**
- 자동 선택 = `FLASHINFER_TRTLLM`. 로그의 후보 목록:
  `['AITER','FLASHINFER_TRTLLM','FLASHINFER_CUTLASS','DEEPGEMM','TRITON','MARLIN','HUMMING','BATCHED_DEEPGEMM','BATCHED_TRITON',...]`
- B200(sm_100)에서 유효 후보: `DEEPGEMM`, `FLASHINFER_CUTLASS`, `TRITON`(대조군)
- 기대: MoE 가 SM 140/148 을 쓰므로 여기서 몇 %p 가 갈리면 전체에 직결
- conf:
  - `moe_dg   NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--moe-backend DEEPGEMM"`
  - `moe_cut  NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--moe-backend FLASHINFER_CUTLASS"`

### C2. `--linear-backend` **[로그 확인 — kernel_config 에 'auto']**
- dense GEMM 백엔드. MoE 다음 소비처 | **[요검증]** 선택지 목록
- 우선순위 중하

---

## 4. 후보 D — 통신/컴파일/기타

### D1. `VLLM_ALLREDUCE_USE_FLASHINFER=1` (환경변수) **[로그 확인 — 기본 False]**
- all-reduce 후보 목록에 `FLASHINFER` 가 있으나 기본 비활성.
  TP=8 all-reduce 는 decode 스텝마다 도니 몇 % 여지
- conf 는 env 를 그대로 전달하므로:
  `ar_fi  NO_EP=1 SPEC_TOKENS=1 VLLM_ALLREDUCE_USE_FLASHINFER=1`
- 주의: `--disable-custom-all-reduce` 는 기각됨 (교착 이력 + 역방향)

### D2. CUDA graph 캡처 범위 **[로그 확인 — cudagraph_mode FULL_AND_PIECEWISE]**
- `--compilation-config` 로 `max_cudagraph_capture_size`, capture sizes 조정.
  관찰: 서버 D 기동마다 512/1024 로 다르게 잡힌 사례 있음 → 의미 조사 필요 **[요검증]**
- `--enforce-eager` 는 음성 대조군(그래프 이득 정량화)으로만 1회 가치

### D3. 스케줄러 미세 옵션 **[로그 확인 — 이름 존재]**
- `--scheduler-reserve-full-isl` / `--no-...`: KV 예약 정책으로 추정.
  덜 보수적이면 admission 증가 가능 → 의미 확인 후 시험 **[요검증]**
- `--prefill-schedule-interval`, `--max-num-scheduled-tokens`: prefill/decode
  배분 관련 추정 **[요검증]**
- 참고: mnbt(8192→16384)는 이미 **효과 없음 실측** — 이 계열 기대치는 낮게

### D4. API/클라이언트 오버헤드 (효율 관점)
- conc=256 스트리밍 1,024개를 API 서버 프로세스 1개가 처리 — CPU 병목 여부를
  기동 중 `top` 으로 관찰 후, `--api-server-count` 류 옵션 존재 시 시험 **[요검증]**

---

## 5. 방법론 이슈 (실험 전 처리)

### 5-1. 노이즈 바닥 미측정
- 같은 서버 같은 구성 반복이 아직 없다. **P0 로 mtp1_ep1 1회 재실행** →
  이후 모든 ±% 판독의 기준. (서버 B↔D 교차 재현 ±1% 이 힌트)

### 5-2. sweep 레벨 간 prefix cache 오염 의심
- 관찰: 고유 sheet 1,024장인데 hit rate 17~25% — 레벨을 넘어가며 같은 sheet 가
  재사용되기 때문(레벨 내 중복은 없음). 벤치가 레벨 사이 `/reset_prefix_cache`
  를 시도하지만 실제 동작 여부 미확인 **[요검증]**
- 처리: B2(`--no-enable-prefix-caching`) 실험이 자연스러운 통제. 만약 유의미한
  차이가 나오면 기존 수치 전체에 각주 필요

### 5-3. 효율 지표 추가 제안
- "성능"은 Output TPS, "효율"은 **tok/s/GPU** 와 **tok/J**.
  측정 중 `nvidia-smi dmon`(전력) 샘플링을 러너에 추가하면 구성별
  전력·에너지 비교 가능 — B200 1000W 에서 MoE 백엔드 간 효율 차가 갈릴 수 있음

---

## 6. 제안 실험 매트릭스

### Phase 0 — 기반 (각 ~35분)
| 이름 | 목적 |
|---|---|
| `repeat_ref` (mtp1_ep1 재실행) | 노이즈 바닥 |

### Phase 1 — 스크리닝 (`SWEEP=32,128,256` 로 단축, 각 ~20분)
| conf 라인 | 가설 |
|---|---|
| `mtp2      NO_EP=1 SPEC_TOKENS=2` | accept 2.76 활용 |
| `kvup     NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--gpu-memory-utilization 0.95"` | 상주 +3~4 |
| `noapc    NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--no-enable-prefix-caching"` | 오버헤드/오염 제거 |
| `seq128   NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--max-num-seqs 128"` | 선점 감소 |
| `moe_dg   NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--moe-backend DEEPGEMM"` | MoE 커널 |
| `moe_cut  NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--moe-backend FLASHINFER_CUTLASS"` | 〃 |
| `ar_fi    NO_EP=1 SPEC_TOKENS=1 VLLM_ALLREDUCE_USE_FLASHINFER=1` | all-reduce |
| `dsmla    NO_EP=1 SPEC_TOKENS=1 EXTRA_ARGS="--kv-cache-dtype fp8_ds_mla --attention-backend FLASHMLA_SPARSE"` | KV 레이아웃+어텐션 |

8건 × ~25분(기동 포함) ≈ **3.5시간**. 노이즈 바닥(P0) 초과 개선만 승자로.

### Phase 2 — 확정 (각 ~35분)
- 승자 구성 full sweep (4~256)
- 승자들 조합 스택 (예: spec2 + kvup + moe 승자) — 교호작용 확인
- spec=1 vs 2 를 승자 스택 위에서 재대조

### 기각/보류 (재실험 금지 목록)
- EP8 (두 서버에서 −4~−13%), mnbt 상향 (무효과),
  `--disable-custom-all-reduce` (교착 이력·역방향), TP<8 (모델 미적재)

---

## 7. 실행 전 체크리스트

1. 서버 엔드포인트 복구 확인 (현재 137.175.28.108:35316 응답 없음)
2. **[요검증]** 표시 항목의 인자 이름·의미를 argparse 덤프로 재확인
   (무효 인자는 기동 즉시 실패하므로 실험 자체는 안전)
3. `bench/experiments.conf` 에 Phase 0→1 순서로 기입, 러너 실행
4. 결과 폴더 지문 자동 동봉 확인 후 수치 비교
