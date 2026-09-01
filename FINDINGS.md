# 검증된 사실 (2026-08-31 세션)

실측으로 확인한 것만 적는다. 추론은 "미검증"으로 표시한다.
다음 세션에서 같은 것을 다시 파헤치지 않기 위한 문서다.

---

## 1. 환경 (실측)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 × 8, sm_100, 178.34 GiB/GPU |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.28.0 |
| 모델 | `zai-org/GLM-5.2-FP8`, 141 safetensors, 722GB |
| venv | `/venv/main/bin/python3` |

`README.md` 에 한때 CUDA 12.8 / PyTorch 2.11.0+cu128 로 적혀 있었으나 틀렸다.

### 모델 config.json 요약

```
architectures        = ["GlmMoeDsaForCausalLM"]
model_type           = glm_moe_dsa
dtype                = bfloat16          ← 가중치만 FP8, 연산은 bf16
max_position_embeddings = 1048576
index_topk           = 2048              ← sparse attention indexer
kv_lora_rank         = 512
num_nextn_predict_layers = 1             ← MTP 헤드
quantization_config  = {quant_method: fp8, activation_scheme: dynamic,
                        fmt: e4m3, weight_block_size: [128,128]}
```

---

## 2. 서버 인자

### 필수 (빼면 기동 불가)

| 인자 | 이유 |
|---|---|
| `--tensor-parallel-size 8` | 모델 ~700GB. 1장(178GB)에 안 들어감. TP=4(712GB)도 KV cache 여유 부족 |
| `--max-model-len 32768` | native 1,048,576 그대로면 KV cache 가 최대 길이 요청 하나를 못 담아 기동 거부 |
| `--trust-remote-code` | GLM 커스텀 모델링 코드 |

### 선택적 최적화

**`--kv-cache-dtype fp8`** — 실측 A/B 확인:

| 구성 | KV cache | 32k 요청 기준 수용 | 기동 | 추론 |
|---|---|---|---|---|
| 미지정 (디폴트 = bf16) | 796,160 tokens | 24.30x | 정상 | 정상 |
| `fp8` 명시 | 1,546,112 tokens | 47.18x | 정상 | 정상 |

정확히 2배. **디폴트는 bf16 이고, fp8 은 자동 적용되지 않는다.**
`results/` 의 모든 측정값은 `fp8` 구성에서 나왔다 = 디폴트가 아니다.

### 무효값 / 함정

- **`--dtype float8_e4m3fn` 은 vLLM 0.28 에서 무효.** 허용값은
  `auto|bfloat16|float|float16|float32|half`. `auto` 를 쓸 것.
- **`--gpu-memory-utilization 0.92` 는 vLLM 기본값과 동일.** 명시해도 변화 없음.
- **`--enable-prefix-caching`, `--enable-chunked-prefill` 은 V1 엔진에서 자동 활성.**
  플래그는 사실상 no-op.
- **`--data-parallel-size 8` + TP=1 은 불가능.** DP 는 복제본마다 전체 모델을
  요구하므로 8×700GB 가 필요하다. CUDA OOM.
- **`pgrep -f 'openai.api_server'` 로 서버 생존을 판정하면 안 된다.**
  `vllm serve` 로 띄우면 기동 초기 cmdline 이 `vllm serve ...` 뿐이라 매칭되지
  않는다. 살아있는 서버를 죽었다고 오판한다. `$!` 로 받은 PID 에 `kill -0` 을
  쓰거나 패턴을 `'vllm serve|openai\.api_server|VLLM::'` 로 넓힐 것.
- **맨 `python3` 로 실행하면 안 된다.** `/usr/bin/python3` 로 잡혀
  `ModuleNotFoundError: No module named 'vllm'`. `/venv/main/bin/python3` 명시.

---

## 3. 내가 틀렸던 추론 (반복 방지)

### `--kv-cache-dtype fp8` 이 필수라고 결론냈다 → 틀렸다

근거로 삼은 코드:
```
vllm/models/deepseek_v32/attention.py:162   require_fp8_kv_cache: bool = True
vllm/models/deepseek_v32/attention.py:273   assert is_quantized_kv_cache(...)
```

**GLM 은 이 파일을 타지 않는다.** 레지스트리가 명시한다:
```
vllm/model_executor/models/registry.py:118
  "GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM")
```

vLLM 0.28 에는 DeepSeek 계열 구현이 두 벌 있다:
- `vllm/model_executor/models/deepseek_v2.py` ← **GLM 이 실제로 타는 경로. assertion 없음**
- `vllm/models/deepseek_v32/` ← 별개의 신규 구현. 여기에만 assertion 존재

교훈: grep 히트만으로 호출 경로를 단정하지 말 것. import 체인을 확인하거나 실측할 것.

### vLLM docstring 이 부정확하다

`vllm/config/cache.py:81` 의 서술:
> Some models (namely DeepSeekV3.2) default to fp8

이 버전 코드와 맞지 않는다. `auto` 해석은
`vllm/utils/torch_utils.py:491 resolve_kv_cache_dtype_string()` 이 담당하고,
그 안의 `get_kv_cache_quant_algo_string()` (line 427) 은
`quant_method.startswith("modelopt")` 일 때만 KV cache 양자화를 읽는다.
GLM 은 `quant_method = "fp8"` 이라 modelopt 이 아니므로 `auto` 로 남고,
`kv_cache_dtype_str_to_dtype()` 에서 `model_config.dtype` = bf16 이 된다.

---

## 4. 벤치마크 방법론

### gen8k 워크로드

LG U+ 원본 방식. `bench/workload/` 참조.

| 항목 | 값 | 근거 |
|---|---|---|
| sheet 수 | **1,024** | 원본 워크플로 문서. `--requests` 기본값이 1024 |
| 생성 인자 | `--n-uuid 37 --n-garbage 27` | `generate_workload.py` argparse 기본값 |
| 파일 크기 | 17,430 bytes | 실측 |
| ISL | **8,211 tokens** | GLM tokenizer `/tokenize` 실측 |
| OSL | **1,024** | "8k1k 벤치" 명칭 + `--max-tokens` 기본값 1024 |
| sweep | 4,8,16,32,64,128,256 | 원본 문서 명령. PDF 표와 일치 |

`generate()` 내부 기본값 90/65 는 ~15k 토큰이 되어 컨텍스트 초과 → HTTP 400.

### sheet 수가 requests 보다 적으면 측정이 오염된다

200장으로 conc=256(1,024 요청)을 돌리면 sheet 당 ~5회 재사용 → prefix cache 히트.
Input TPS 는 명목 ISL 로 계산되므로 **고concurrency 구간이 7~8% 과대 측정**된다.
Interactivity 는 더 크게 틀어진다 (conc=64 에서 28.62 vs 실제 21.35, +34%).

벤치 스크립트가 경고를 띄운다:
```
WARNING: 200 sheets < 1024 requests -- reused sheets hit the prefix cache
```
이 경고가 나오면 결과를 버릴 것.

### `--conc` 는 `--sweep ""` 없이는 무시된다

```python
concs = [int(c) for c in args.sweep.split(",") if c.strip()] or [args.conc]
```

`bench_sweep_b200.py` 의 `--sweep` 기본값이 `"1,4,8,16,32,64,128,256"` 이므로
`--sweep` 을 생략하면 `--conc` 가 무시되고 전체 sweep 이 돌아간다.
단일 concurrency 만 재려면 `--sweep "" --conc N` 을 함께 줘야 한다.
(`bench_sweep_h200.py` 기본값은 `"12,64,128,200,256"`, `bench_sweep_mi355x.py` 는 `""`)

단일 conc 실행은 wave 축소(conc*4)가 적용되지 않아 `--requests` 전량을 보낸다.

### 벤치 스크립트의 `--device` / `--precision` / `--tp` 는 라벨이다

```python
# Table metadata. These do not change what is sent -- they label the row and
# set what the throughput is divided by.
```

전송 payload 를 바꾸지 않는다. 결과 표의 "FP8" 은 측정값이 아니라
`--precision` 기본값이 찍힌 문자열이다. **서버 실제 구성과 무관하다.**

### 벤치 스크립트 세 종의 차이

| | `bench_sweep_mi355x.py` | `bench_sweep_h200.py` | `bench_sweep_b200.py` |
|---|---|---|---|
| 원본/파생 | 원본 | 파생 | 파생 |
| `--device` 기본 | MI355x | H200 | B200 |
| `--precision` 기본 | **MXFP4** | FP8 | FP8 |
| `--model` 기본 | `/models/GLM-5.2-MXFP4` | 자동 조회 | 자동 조회 |
| `--tp` 기본 | 0 | 8 | 8 |
| `--api-key` | 없음 | 있음 | 있음 |

측정 로직 차이는 한 줄뿐이다:
```python
- tpl = None if args.think else {"enable_thinking": False}   # mi355x
+ tpl = {"enable_thinking": bool(args.think)}                # h200/b200
```
thinking 기본 off 이므로 **기본값으로 돌리면 세 스크립트 결과가 동일**하다.

---

## 5. B200 측정 결과

서버: TP=8, `--kv-cache-dtype fp8`, mnbt 8192, gpu-util 0.92, MTP 미적용
워크로드: gen8k 1,024장, ISL ~8,200, OSL 1,024

| conc | TPOT(ms) | Interactivity | Input TPS | Output TPS |
|---|---|---|---|---|
| 1 | 9.45 | 105.77 | 815 | 102 |
| 4 | 11.48 | 87.09 | 2,660 | 333 |
| 8 | 13.04 | 76.72 | 4,614 | 577 |
| 16 | 15.97 | 62.60 | 7,523 | 940 |
| 32 | 20.70 | 48.32 | 11,724 | 1,465 |
| 64 | 35.57 | 28.11 | 13,883 | 1,734 |
| 128 | 62.69 | 15.95 | 15,810 | 1,975 |
| 256 | 83.75 | 11.94 | 16,138 | 2,016 |

OSL 512 로 돌린 결과는 `results/b200_osl512_kvfp8.md` 에 있으나
문서화된 조건(OSL 1024)과 다르므로 참고용이다.

---

## 6. H200 / Moreh 비교의 한계

`H200_GLM5.2_Measure.pdf` 에서 얻을 수 있는 것과 없는 것.

**있음**: concurrency, TPOT, Interactivity, Input/Output TPS, 장비가, 원가, MTP 유무

**없음** (서버 구성 전부):
- `--max-num-batched-tokens`
- `--max-num-seqs` (continuous batching 시퀀스 상한)
- `--kv-cache-dtype`
- `--gpu-memory-utilization`, `--max-model-len`
- 실제 TP, 실제 precision (표의 값은 라벨)
- attention backend, vLLM 버전
- MTP `num_speculative_tokens`

서버 어디에도 H200 기동 명령의 흔적이 없다 (쉘 히스토리, nohup.out, 스크립트 모두 없음).

**Moreh precision 은 상충한다.** PDF 표는 Moreh 열까지 "FP8" 로 병합해놨으나,
원본 워크플로 문서의 예시 명령은 `--precision MXFP4 --tp 4` 이고
`bench_sweep_mi355x.py` 기본 모델 경로도 `/models/GLM-5.2-MXFP4` 다.
MXFP4(4bit)면 FP8(8bit) 대비 decode 가 빨라 Output TPS 우위(2,777)의 원인일 수 있다.

**결론**: 현재 비교는 "우리 구성 vs 상대의 미상 구성"이다.
순수 하드웨어 비교로 제시하면 과한 주장이다.
상대측 서버 실행 명령 한 줄을 받으면 대부분 해소된다.

---

## 6-1. 2026-09-01 측정 (지정 서빙 구성)

서버 인자 (포트 30269 는 bench_sweep_h200.py 기본 포트와 동일 — H200 구성으로 추정):
```
--tensor-parallel-size 8 --async-scheduling --kv-cache-dtype fp8
--max-model-len 16384 --max-num-seqs 256 --gpu-memory-utilization 0.93
--enable-chunked-prefill --enable-prefix-caching --trust-remote-code
```
`--ignore-eos 1` 은 vllm serve 인자가 아니므로 제외했다 (argparse 386 개 옵션 중
eos 포함 옵션 0 개). 벤치가 요청별로 전송하므로 동작은 동일.

KV cache 1,571,008 tokens. mnbt 는 미지정 시 max_model_len 과 같은 값(16384)이 된다.

| conc | TPOT(ms) | Interactivity | Input TPS | Output TPS |
|---|---|---|---|---|
| 4 | 11.27 | 88.73 | 2,437 | 304 |
| 8 | 13.11 | 76.29 | 4,553 | 569 |
| 16 | 15.78 | 63.37 | 7,365 | 920 |
| 32 | 20.14 | 49.65 | 11,274 | 1,408 |
| 64 | 34.24 | 29.21 | 14,181 | 1,772 |
| 128 | 60.58 | 16.51 | 16,094 | 2,010 |
| 256 | 84.96 | 11.77 | 16,069 | 2,007 |

### Expert Parallel 은 기본적으로 꺼져 있다

로그 근거:
```
[fp8.py:713] Using MoEPrepareAndFinalizeNoDPEPMonolithic
rank 0 ... DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
```
`NoDPEP` = expert parallel 미사용. 전문가를 EP 로 분산하지 않고 TP 로 샤딩한다.
world size 8 이 전부 TP 로 소진되고 DP=1.

### MoE 백엔드는 자동 선택된다

```
[fp8.py:411] Using FLASHINFER_TRTLLM Fp8 MoE backend out of potential backends:
  ['AITER','FLASHINFER_TRTLLM','FLASHINFER_CUTLASS','DEEPGEMM','TRITON','MA...']
TRT-LLM fused MoE cooperative launch SM allocation: 140 SMs used for MoE,
  8 SMs reserved (total SMs: 148)
```
`--moe-backend` 로 강제 가능. SM 148 중 140 을 MoE 가 쓰므로 튜닝 여지가 있다.

### MTP 구성

`--spec-method mtp --spec-tokens 2` (JSON `--speculative-config` 보다 인용이 안전).
MTP 활성 시 KV cache 가 1,571,008 → 1,472,768 tokens 로 줄어든다 (MTP 레이어 메모리).
vLLM 경고: `num_speculative_tokens > 1` 은 같은 MTP 레이어를 여러 번 forward 하므로
acceptance rate 가 낮아질 수 있다 → 1 도 함께 시험할 가치가 있다.

---

## 7. 미검증 가설

### [기각] mnbt 8192 < ISL 8,200 이 고concurrency 병목이다

프롬프트 하나가 스텝당 토큰 예산보다 크다. chunked prefill 에서 prefill 과 decode 가
같은 8192 예산을 나눠 쓰므로, 8,200 토큰 프롬프트를 채우면 그 스텝의 decode 예산이
사실상 0 이 된다.

정황 증거: conc=32 → 64 에서 TPOT 이 20.70 → 35.57 로 급증하는데
이 구간 KV cache 여유는 충분하다 (1,546,112 토큰 / 요청당 ~9,224 = 약 167 요청 수용).
또 OSL 을 512 → 1024 로 늘리면 conc≥16 전 구간에서 Output TPS 가 개선되는데,
출력이 길어져 prefill 압력이 절반으로 줄어든 것과 부합한다.

**결과: 기각.** 2026-09-01 에 mnbt 16384 (+ `--async-scheduling`) 로 재측정했으나
전 구간 차이가 1~3% 이내이고 방향도 일관되지 않았다.

| conc | TPOT (mnbt 8192 → 16384) | Output TPS |
|---|---|---|
| 32 | 20.70 → 20.14 | 1,465 → 1,408 |
| 64 | 35.57 → 34.24 | 1,734 → 1,772 |
| 128 | 62.69 → 60.58 | 1,975 → 2,010 |
| 256 | 83.75 → 84.96 | 2,016 → 2,007 |

**실제 원인은 KV cache 용량으로 보인다.** 요청당 9,224 토큰(ISL 8,200 + OSL 1,024)에
KV cache 1,571,008 토큰이므로 약 170 개만 상주 가능하다. conc=256 이면 86 개가
대기하므로 사용자별 체감 속도가 떨어진다. 어제 "conc=64 구간은 KV 여유가 충분하다"는
관찰만으로 예산 문제로 단정한 것이 오류였다.

conc=64 구간의 TPOT 증가는 배치 확대에 따른 decode 스텝당 연산량 증가로 보인다.

주의: 이 측정은 mnbt 와 `--async-scheduling` 을 동시에 바꿨다. 두 효과가 상쇄됐을
가능성은 낮다고 본다 (전 구간 차이가 1~3% 로 균일).

### MTP speculative decoding 효과

H200 은 MTP 로 Output TPS 653 → 702 (+7.5%), Interactivity 9.82 → 11.23 (+14%) 를 얻었다.
B200 은 미적용. 모델에 `num_nextn_predict_layers = 1` 이 있어 MTP 헤드는 존재한다.

**검증 방법**: `start_server.sh` 의 `--speculative-config` 주석 해제.
bash nohup 안에서 single-quote 이스케이프가 깨지기 쉬우므로 배열로 전달할 것.

---

## 8. 서버 리셋 후 복원 순서

```bash
# 1. 레포 + 환경
curl -fsSL https://raw.githubusercontent.com/crinex/b200_glm_poc/main/setup/bootstrap.sh | bash

# 2. 모델 (722GB, 오래 걸림)
HF_TOKEN=hf_xxx bash setup/download_model.sh

# 3. 워크로드
bash bench/workload/build_gen8k.sh /workspace/gen8k 1024

# 4. 서버 (첫 기동은 FlashInfer JIT cubin 다운로드로 ~25분,
#    이후 캐시되어 ~15분. "No available shared memory broadcast block
#    found in 60 seconds" 는 정상 메시지)
bash serve/start_server.sh

# 5. 벤치 (전 구간 약 15분)
/venv/main/bin/python3 bench/bench_sweep_b200.py \
  --host localhost --port 8000 --model glm-5.2 \
  --gen-dir /workspace/gen8k \
  --sweep 4,8,16,32,64,128,256 \
  --max-tokens 1024 \
  --out /workspace/results/run1
```

재기동 시에는 반드시 먼저: `bash serve/gpu_reset.sh`
