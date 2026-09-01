# 검증된 사실 (2026-08-31 ~ 09-01)

실측으로 확인한 것만 적는다. 추론은 "미검증"으로 표시한다.
다음 세션에서 같은 것을 다시 파헤치지 않기 위한 문서다.

---

## 0. 가장 먼저 읽을 것 — NCCL P2P transport 교착

**서버를 SIGKILL 로 죽이면 이후 모든 기동이 교착한다. 인스턴스 재시작만이 복구 방법이다.**

2026-09-01 에 실측으로 확인. 이 인스턴스의 **첫 기동만 성공**하고,
SIGKILL 이후 6회 기동이 전부 동일 지점에서 멈췄다.

### 증상

- 예외·에러 로그 없음. 무증상 정지
- 로그가 `[pynccl.py:113] vLLM is using nccl==2.29.7` 직후 멈추고
  `[cuda_communicator.py] Using [...] all-reduce backends ... 'tp:0'` 에 도달하지 않음
  (정상이면 4초 만에 도달)
- 워커 8개가 `State: R`, CPU 99.9% 로 무한 회전. 스레드 63개
- GPU 는 1,074 MiB (CUDA 컨텍스트만), 사용률 0%. 가중치 로드 안 됨
- `/tmp/vllm_dist_*` 파일이 4,160 bytes 에서 멈춤 (성공 시 16,352 bytes)

### NCCL_DEBUG=INFO 로 특정한 지점

```
Bootstrap timings total 0.286458 (...)          ← 부트스트랩 정상 완료
New proxy send connection 6 from local rank 6, transport 0
Connected to proxy localRank 0 -> 0x7aedf40052d0   ← 마지막 줄. 이후 정지
```

rendezvous 는 0.29초에 끝난다. 멈추는 곳은 **P2P transport 설정
(transport 0 = NVLink/PCIe) 의 CUDA IPC 핸들 교환** 구간이다.

부수적으로 IB 심볼 오류가 16건 찍히지만 교착의 직접 원인인지는 미확정:
```
mlx5dv_reg_dmabuf_mr - libmlx5.so.1: undefined symbol, version MLX5_1.25
```

### 원인이 아닌 것들 (전부 소거됨)

| 후보 | 검증 |
|---|---|
| vLLM 인자 | `--disable-custom-all-reduce` 없이도, EP 꺼도 재현 |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `=0` 으로도 재현 |
| `/dev/shm` 고갈 | 998G 중 8.0K 사용 |
| 공유메모리/세마포어 누수 | `ipcs -m`, `ipcs -s` 모두 비어 있음 |
| 좀비 프로세스 | 0개 |
| GPU 메모리 누수 | kill 하면 0 으로 복귀 |
| 잔여 파일 | `/tmp/vllm_dist_*`, `/dev/shm` 삭제 + 60초 대기 후에도 재현 |

### 대처

- **예방**: 서버를 SIGKILL 하지 말 것. `serve/gpu_reset.sh` 가 SIGTERM 을
  먼저 시도한다. 벤치 실패 시 서버를 살려두고 벤치만 재실행할 것
  (`SKIP_BOOT=1`)
- **복구**: 인스턴스 재시작. 프로세스/파일 정리로는 풀리지 않는다
- `NCCL_P2P_DISABLE=1` 로 기동될 가능성은 있으나 GPU 간 통신이 느려져
  성능 측정값이 무의미해진다. 벤치마크에는 쓸 수 없다
- py-spy 로 스택을 뜨려 했으나 컨테이너에서 ptrace 차단
  (`/proc/sys` 읽기 전용). `NCCL_DEBUG=INFO` 가 대안이다

---

## 0-1. deploygpu 인스턴스 초기화 (2026-09-01 확립)

Vast.ai 이미지와 달리 빈 서버다. **`setup/setup_deploygpu.sh` 를 먼저 실행할 것.**
접속은 `ssh b200` (로컬 `~/.ssh/config` 등록, 키 `~/.ssh/deploygpu_b200_openssh`.
같은 이름의 `.pem` 파일은 형식이 깨져 있어 사용 불가).

에러 → 원인 매핑 (겪은 순서대로):

| 증상 | 원인 | 해결 |
|---|---|---|
| venv 생성 시 ensurepip 오류 | python3.12-venv 미설치 (+ apt 캐시 낡음) | `apt update && apt install python3.12-venv` |
| `torch.cuda.is_available()=False` | 드라이버 570 = CUDA 12.8 브랜치, torch 는 cu130 | `cuda-compat-13-0` + ld.so.conf 최우선 등록 |
| `Error 802: system not yet initialized` | Fabric Manager 미기동 (NVL5 필수) | 아래 FM 체인 |
| FM: "ibstat not found" | 프로바이더의 FM 시작 스크립트가 요구 | `infiniband-diags` |
| FM: "ib_umad not loaded" | NVL5 관리 경로 | `modprobe ib_umad` + modules-load.d |
| FM: "/opt/nvidia/nvlsm/sbin/nvlsm does not exist" | NVLink Subnet Manager 미설치 | `apt install nvlsm` |
| 기동 중 `FileNotFoundError: 'ninja'` (profile 단계) | FlashInfer 샘플링 커널 JIT 에 빌드 도구 필요 | `apt install ninja-build build-essential` |

FM 패키지는 커널 드라이버와 **정확히 같은 버전**이어야 한다
(`nvidia-fabricmanager-570=570.195.03-1`). forward-compat 은 사용자 공간
libcuda(580.178.04)만 교체하므로 재부팅이 필요 없다.

성공 로그: `OpenSM: Entering MASTER state` →
`Successfully configured all the available GPUs and NVSwitches` →
`torch.cuda.is_available() = True`, B200 × 8.

주의: 이 인스턴스의 nvswitch 는 `/dev/nvidia-nvswitch*` 로 보이지 않고
lspci 에도 안 잡히지만 (가상화), FM 없이는 CUDA 가 열리지 않는다.

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

### Input TPS 는 prefill 성능 지표가 아니다 — Output TPS × (ISL/OSL) 이다

벤치의 Input TPS = 전체 입력 토큰 / 전체 벽시계 시간이고, 벽시계 시간에는
decode 가 포함된다. OSL 1,024 면 시간의 대부분이 decode 이므로 Input TPS 가
희석된다. 실제 prefill 속도는 vLLM 서버 로그의 `Avg prompt throughput`
(측정 중 24,577 tok/s) 을 봐야 한다. 벤치가 보고한 값은 16,956 이었다.

수식으로 Input TPS ≈ Output TPS × (ISL/OSL). 우리 측정 (ISL/OSL = 8.0):

| conc | Output TPS | × 8.0 | 실측 Input TPS |
|---|---|---|---|
| 4 | 372 | 2,976 | 2,979 |
| 64 | 1,831 | 14,648 | 14,660 |
| 128 | 2,118 | 16,944 | 16,956 |
| 256 | 2,065 | 16,520 | 16,534 |

오차 0.1% 이내. **Input TPS 는 독립 정보가 없다. Output TPS 만 보면 된다.**

### H200 / Moreh 의 ISL:OSL 비율은 8:1 로 확정된다

위 관계를 PDF 수치에 적용하면 비율을 역산할 수 있다.

| 측정 | Input TPS | Output TPS | 비율 |
|---|---|---|---|
| H200 MTP 미사용 @256 | 5,227 | 653 | 8.00 |
| H200 MTP 사용 @256 | 5,618 | 702 | 8.00 |
| H200 MTP 사용 @4 | 2,357 | 294 | 8.02 |
| Moreh @256 | 21,996 | 2,777 | 7.92 |

전부 8:1 이다. PDF 에 ISL/OSL 이 기재되지 않았지만 **비율은 확정**되며,
gen8k 실측 ISL 8,211 과 "8k1k 벤치" 명칭을 합치면 OSL 1,024 로 볼 근거가 된다.
(§6 의 "ISL/OSL 없음" 항목은 이 계산으로 상당 부분 해소된다.)

처음에 `--max-tokens 512` 로 돌린 측정이 왜 조건 불일치인지도 이것으로 설명된다.
그 경우 비율이 16:1 이 되어 다른 측정과 성립하지 않는다.

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

### Expert Parallel 은 기본적으로 꺼져 있고, `--enable-expert-parallel` 로 켜진다

**EP 활성 여부를 판정하는 로그는 이 두 개다:**
```
[expert_map_manager.py:245] [EP Rank 0/8] Expert parallelism is enabled.
  Expert placement strategy: linear. Local/global number of experts: 32/256.
[cuda_communicator.py:266] Using ['PYNCCL'] all-reduce backends for group 'ep:0'
```
켜면 전문가 256 개가 GPU 8 장에 32 개씩 통째로 배분되고 `ep:0` 통신 그룹이 생긴다.
DP=1 이어도 TP 그룹 8 장에 걸쳐 EP 가 성립한다.

**주의 — 다음 라인은 EP 지표가 아니다:**
```
[fp8.py:713] Using MoEPrepareAndFinalizeNoDPEPMonolithic
```
EP 를 켜도 끄도 동일하게 출력된다. `NoDP`(데이터 병렬 없음) + `Monolithic`(단일 노드)
조건을 가리키며 EP 유무와 무관하다. 이것을 "No DP, No EP" 로 오독해
EP 가 꺼졌다고 판단한 적이 있다 (2026-09-01).

### EP 크기는 직접 지정할 수 없다 — DP × TP 로 파생된다

expert 관련 인자는 네 개뿐이고 크기 지정 옵션이 없다:
```
--enable-expert-parallel / --no-enable-expert-parallel
--expert-placement-strategy
--enable-return-routed-experts
```

DP=1, TP=8 에서 켜면 **무조건 EP8** 이 된다 (전문가 256개 → GPU 8장에 32개씩).
**"EP1" 은 expert parallel 을 끄는 것**과 같다. 전문가를 EP 로 분산하지 않고
TP 로 샤딩하는 상태이며, 이것이 기본값이다.
스크립트에서는 `NO_EP=1`.

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


## 6-2. 2026-09-01 deploygpu 서버 — 동일 서버 3구성 분해 (확정)

서버: deploygpu (드라이버 570 + cuda-compat-13-0, FM 구성). 서빙 인자는 §6-1 과 동일.
워크로드 동일 (gen8k 1,024장, ISL ~8,200, OSL 1,024).

### Output TPS (tok/s per server)

| conc | 기준선 | MTP1+EP1 | MTP1+EP8 | MTP 몫 | EP 몫 |
|---|---|---|---|---|---|
| 4 | 259 | 307 | 297 | +18.5% | -3.3% |
| 8 | 463 | 601 | 559 | +29.8% | -7.0% |
| 16 | 725 | 788 | 845 | +8.7% | +7.2% |
| 32 | 985 | 1,327 | 1,328 | +34.7% | +0.1% |
| 64 | 1,387 | 1,712 | 1,587 | +23.4% | -7.3% |
| 128 | 1,854 | **2,013** | 1,783 | +8.6% | **-11.4%** |
| 256 | 1,843 | 1,994 | 1,729 | +8.2% | **-13.3%** |

### 결론 (이 서버 기준)

- **MTP(spec-tokens 1)는 전 구간 이득**: +8~35%. 수락률 74~79%, accept length ~1.75
- **EP8 은 중립~해로움**: 고concurrency 에서 -11~-13%. 최적은 **MTP1 + EP1 (EP 끔)**
- 최고 처리량: conc=128 에서 2,013 tok/s
- KV cache: 기준선 1,495,168 / MTP1+EP1 1,390,848 / MTP1+EP8 1,402,880
  (MTP 가 약 7% 잠식)

### 하드웨어 지문을 반드시 수집할 것

서버 A/B/C 의 성능차(-8~-16%) 원인 판정에 필요했던 값들 — 클럭·전력 상한,
NVLink 토폴로지, 호스트 CPU, 가상화 형태, FM 파티션 — 을 수집하지 않아
인스턴스 소멸 후 소급 확인이 불가능했다.

`setup/fingerprint.sh` 가 이를 수집한다. `full_setup.sh` 가 세팅 끝에 자동
실행하고, `bench/run_mtp1_ep_sweep.sh` 는 결과 폴더에 사본
(`<OUT>/fingerprint.txt`)을 남긴다. **측정 결과를 레포로 회수할 때 지문도
함께 가져올 것.**

### 서버 간 편차가 구성 효과를 압도한다

같은 구성(MTP1+EP8)이 Vast 서버에서 2,065~2,118, deploygpu 에서 1,729~1,783
(-16%). 기준선도 -8%. **교차 서버 분해는 무효**이며, 실제로 교차 데이터로는
"EP 몫이 크고 MTP 몫이 0" 이라는 정반대 결론이 나왔었다.
NVLink 토폴로지는 정상(전 쌍 NV18, 링크당 50GB/s)이므로 forward-compat
드라이버 경로나 호스트/가상화 차이로 추정 (미확정).
**구성 비교는 반드시 같은 서버에서 할 것.**


## 6-3. 2026-09-01 서버 D (풀스펙) — 3구성 재검 (최종)

서버 D: Vast 계열, 드라이버 595.91.07, **전력 1000W/1000W, SM max 1965MHz**
(지문: `results/fingerprint_serverD_20260901.txt`), docker.

Output TPS 요약 (전체 표는 results/b200_D_*.md):

| conc | 기준선 | MTP1+EP1 | MTP1+EP8 | MTP 몫 | EP8 몫 |
|---|---|---|---|---|---|
| 4 | 309 | 427 | 403 | +38% | -6% |
| 32 | 1,410 | 1,874 | 1,778 | +33% | -5% |
| 64 | 1,767 | 2,045 | 1,915 | +16% | -6% |
| 128 | 2,003 | **2,251** | 2,133 | +12% | -5% |
| 256 | 1,999 | 2,159 | 2,078 | +8% | -4% |

- **MTP spec=1: +8~38%** (수락률 75.2%, accept length 1.75). 두 서버(C, D)에서
  방향 일치, 풀스펙에서 이득이 더 큼
- **EP8: -4~-6% 균일 열세.** C(-3~-13%)에 이어 재확인 → **EP8 은 이 워크로드에서
  쓰지 말 것**
- 최고 기록: **MTP1+EP1, conc=128, 2,251 tok/s** (H200 MTP 대비 3.23×)
- 기준선이 서버 B 와 ±1% 재현 → 풀스펙 인스턴스 간 재현성 확인.
  C 의 -8~-16% 열세는 하드웨어 등급/드라이버 차이 방향 (C 의 전력 상한은
  미기록 — 이후 지문 자동 수집으로 해결)

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
