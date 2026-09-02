# 학습 노트 — MTP · Expert Parallel · 하드웨어 지문 · GPU 토폴로지

GLM-5.2 × B200 실험(2026-08-31 ~ 09-01)에서 실제로 측정하고 부딪힌 것들을
근거로 정리한다. 모든 수치는 이 레포 `results/` 의 실측값이다.

측정 조건 공통: gen8k 1,024장, ISL ≈ 8,200 / OSL 1,024 고정, Weight FP8,
KV cache fp8, TP=8, vLLM 0.28.0.

---

## 1. MTP (Multi-Token Prediction) — 무엇이 빨라지고, 무엇은 안 빨라지나

### 1-1. 동작 원리

일반 디코딩은 forward 1회 = 토큰 1개다. MTP 는 모델에 내장된 **draft 헤드**
(GLM-5.2 는 `num_nextn_predict_layers=1`, 레이어 1개)가 다음 토큰을 미리
추측하고, 본 모델이 다음 forward 에서 **추측을 검증하면서 동시에 그 다음
토큰도 생성**한다. 추측이 맞으면 forward 1회에 토큰 2개가 나온다.

- 스텝당 기대 토큰 수(accept length) = 1 + p₁ (spec-tokens 1)
- 실측: p₁ ≈ 75% → accept length ≈ 1.75
- spec-tokens 2 (서버 B 실측): p₁=93.8%, p₂=88.1% → 1 + 0.938 + 0.826 ≈ 2.76
  - 단 GLM 은 MTP 레이어가 1개뿐이라 2개를 추측하려면 같은 레이어를 두 번
    forward 한다 (vLLM 이 수락률 저하 경고를 냄)

### 1-2. 실측 효과 (서버 D, MTP 켜기 전후, EP 끔)

| conc | 기준선 Out | MTP1 Out | 이득 |
|---|---|---|---|
| 4 | 309 | 427 | +38% |
| 32 | 1,410 | 1,874 | +33% |
| 64 | 1,767 | 2,045 | +16% |
| 128 | 2,003 | 2,251 | +12% |
| 256 | 1,999 | 2,159 | +8% |

**왜 이득이 accept length(+75%)보다 작고, concurrency 가 오를수록 줄어드나:**

1. **MTP 는 디코딩만 가속한다. prefill 은 전혀 건드리지 못한다.**
   ISL 8,200 워크로드에서는 전체 시간의 상당분이 prefill 이라 이득이 희석된다.
2. draft 레이어 forward 자체가 추가 연산이다. 낮은 concurrency 에서는
   디코딩이 메모리 대역폭에 묶여 연산 여유가 있으므로 draft 가 거의 공짜지만,
   높은 concurrency 에서는 배치가 이미 연산을 채우고 있어 실비용이 된다.
3. **MTP 는 KV cache 를 잠식한다** (draft 레이어의 KV). 실측 약 7%:
   1,495,168 → 1,390,848 tokens (서버 D). 요청당 ~9,224 토큰이므로 상주 가능
   요청이 ~162 → ~150 개로 줄어, 포화 구간에서 큐잉이 늘어난다.

### 1-3. "MTP 를 쓰면 Input 처리량이 줄어드나?" — 지표의 함정

먼저 사실관계: **우리 실측에서 같은 조건이면 MTP 를 켰을 때 Input TPS 도
올라갔다** (기준선 16,034 → MTP1 18,019 @conc128). 줄지 않았다.

혼란의 근원은 이 벤치의 Input TPS 정의다:

```
Input TPS = 전체 입력 토큰 수 / 전체 벽시계 시간
```

벽시계 시간에 prefill 과 decode 가 모두 들어가고, OSL 1,024 워크로드에서는
시간의 대부분이 decode 다. 그 결과 수학적으로:

```
Input TPS ≈ Output TPS × (ISL / OSL) = Output TPS × 8.0
```

실측 오차 0.1% 이내로 성립한다 (21개 측정 지점 전부). 즉 **이 벤치의
Input TPS 는 prefill 성능 지표가 아니라 Output TPS 의 8배 복사본**이다.
MTP 로 decode 가 빨라지면 벽시계가 줄어 Input TPS 도 같이 오른다.

그래서 "Input 처리량이 줄어드는" 경우는 이런 것들이다:

- **OSL 을 늘렸을 때** (512→1024 실측: In 19,990 → 15,810 @128).
  decode 체류가 길어져 단위시간당 prefill 횟수가 줄어든 것 — MTP 와 무관.
- **실제 prefill 경합**: chunked prefill 에서 prefill 과 decode 가 스텝당
  토큰 예산(mnbt)을 나눠 쓰므로, decode 쪽 일이 늘면(검증+draft) 포화
  구간에서 prefill 수용이 미세하게 밀릴 수 있다. 실측으로는 KV cache 7%
  잠식 효과가 더 컸다.
- 진짜 prefill 속도를 보려면 서버 로그의 `Avg prompt throughput` 을 봐야
  한다 (실측 ~24,500 tok/s — 벤치 Input TPS 16,000대와 전혀 다른 값).

**교훈: 지표의 계산식을 모르면 개선/악화를 거꾸로 읽는다.**

---

## 2. Expert Parallel — 왜 EP8 이 이득이 없거나 손해였나

### 2-1. MoE 를 8장에 나누는 두 방식

GLM-5.2 는 레이어마다 전문가(FFN) 256개를 두고 토큰마다 일부만 쓴다.

| | TP 샤딩 (EP1, 기본) | EP8 |
|---|---|---|
| 배치 방식 | 모든 전문가의 **가중치 행렬을 8조각**으로. 전 GPU 가 전 전문가의 1/8 씩 보유 | **전문가를 통째로** 32개씩 배분 (256/8) |
| 통신 | 레이어마다 **all-reduce** | 토큰을 담당 GPU 로 보내는 **all-to-all** (dispatch) + 회수 (combine) |
| 부하 분포 | 항상 균등 (모두가 같은 조각 연산) | **전문가 인기도에 따라 불균등** — 몰리는 GPU 가 병목 |

판정 로그 (이걸로 EP 활성을 확인한다):
```
[EP Rank 0/8] Expert parallelism is enabled. Local/global number of experts: 32/256
Using [...] all-reduce backends ... for group 'ep:0'
```
주의: `MoEPrepareAndFinalizeNoDPEPMonolithic` 라인은 EP 를 켜도 끄도 똑같이
나온다. EP 지표가 아니다 (실수했던 부분).

### 2-2. 실측 (서버 D, MTP1 고정)

| conc | EP1 Out | EP8 Out | EP8 효과 |
|---|---|---|---|
| 4 | 427 | 403 | −5.6% |
| 32 | 1,874 | 1,778 | −5.1% |
| 128 | 2,251 | 2,133 | −5.2% |
| 256 | 2,159 | 2,078 | −3.8% |

서버 C 에서도 같은 방향(−3~−13%). **두 인스턴스에서 재확인된 결론.**

### 2-3. 왜 손해인가 — 단일 노드 8-GPU 의 특수성

1. **비교 대상인 all-reduce 가 이미 너무 싸다.** NVLink 로 전 쌍이 직결된
   단일 노드(NV18 = 900GB/s급)에서 vLLM 은 자체 CUSTOM all-reduce 까지 쓴다.
   EP 가 절감해 주는 통신비 자체가 작다.
2. **all-to-all 은 공짜가 아니다.** MoE 레이어마다 토큰을 흩뿌리고(dispatch)
   다시 모으는(combine) 두 번의 재배치가 든다. 게다가 attention 부분은
   여전히 TP 라 all-reduce 도 없어지지 않는다 — 통신이 줄지 않고 종류만 는다.
3. **부하 불균형.** 배치 전략이 `linear`(0~31번 → GPU0 식)이고 EPLB(부하
   재배치)는 미사용이었다. 인기 전문가가 몰린 GPU 가 스텝마다 straggler 가
   되고, 나머지 7장은 그를 기다린다.
4. **GEMM 효율.** TP 샤딩은 활성화된 전문가들의 연산을 큰 배치로 뭉치지만,
   EP 는 GPU 별로 자기 전문가에 온 토큰만 계산하므로 GEMM 이 잘게 쪼개진다.

**그럼 EP 는 언제 이기나:** 노드를 넘어갈 때다. 노드 간 all-reduce 는
비싸므로(NVLink 밖), 전문가를 노드별로 쪼개고 토큰만 이동시키는 편이 낫다.
또 DP 와 결합해 수백 GPU 로 넓히는 wide-EP(DeepEP 류), EPLB 로 부하를
맞추는 구성에서 진가가 나온다. **단일 노드 8-GPU 추론에서는 TP 샤딩이
기본적으로 유리하다** — 이번 실측이 그 교과서적 사례다.

---

## 3. 하드웨어 지문 — 같은 "B200 8장"이 같지 않다

### 3-1. 실제 사건

동일 코드·동일 구성·동일 워크로드인데 인스턴스마다 이랬다:

| 서버 | 기준선 @128 | MTP1+EP8 @128 | 특징 |
|---|---|---|---|
| B (Vast) | 2,010 | 2,118 | 드라이버 595, docker |
| C (deploygpu) | 1,854 (−8%) | 1,783 (**−16%**) | 드라이버 570+compat, **VM** |
| D (Vast 계열) | 2,003 (B와 ±1%) | 2,133 | 드라이버 595, docker, **1000W 확인** |

서버 C 만 느렸고, **통신이 많은 구성(EP8)일수록 격차가 커졌다.** 이 차이는
우리가 튜닝으로 얻는 이득(±5~10%)보다 크다. 즉 **서버가 다르면 구성 비교가
불가능하다** — 실제로 교차 서버 데이터로 분해했다가 정반대 결론("MTP 효과
없음, EP 가 전부")을 낸 적이 있다.

### 3-2. 성능차를 만들 수 있는 하드웨어 특성 (지문 수집 항목)

| 항목 | 왜 성능이 갈리나 | 확인 명령 |
|---|---|---|
| **전력 상한 / 클럭** | B200 은 SKU 가 700W~1000W. 상한이 낮으면 클럭이 눌려 전 구간 균일 감속 | `nvidia-smi --query-gpu=power.limit,clocks.max.sm` |
| **드라이버/GSP 세대** | 신형 GPU 는 최신 브랜치에 최적화가 몰림. GSP 펌웨어는 커널 드라이버에 묶임 — forward-compat 은 사용자공간만 교체 | `nvidia-smi`, `/proc/driver/nvidia/version` |
| **가상화 형태** | VM 은 커널 런치·인터럽트·메모리 pinning 경로에 오버헤드. PCI 토폴로지가 평탄화되어 NUMA 정보도 사라짐 | `systemd-detect-virt`, `/.dockerenv` |
| **NVSwitch 파티션** | FM 이 "partition id N activated" 를 찍으면 패브릭을 파티션으로 분할해 쓰는 것 — 다른 테넌트와 스위치 공유 시 all-to-all 이 먼저 맞는다 | `journalctl -u nvidia-fabricmanager` |
| **NUMA 가시성** | 매핑이 보이면 스케줄러가 GPU 근처 CPU/메모리를 쓴다. VM 에서 N/A 면 최적화 불가 | `nvidia-smi topo -m` 의 CPU/NUMA Affinity |
| **호스트 CPU** | 디토크나이즈, 스케줄러, NCCL proxy 스레드가 CPU 를 탄다. 고concurrency 일수록 민감 | `lscpu` |

서버 C 의 감속 원인은 이 중 무엇인지 **확정하지 못했다** — 당시 C 의 전력
상한을 안 재서다. 그래서 지금은 `setup/fingerprint.sh` 가 모든 세팅과 모든
측정 결과 폴더에 지문을 자동 동봉한다. **수치는 반드시 지문과 짝으로
보관한다**는 것이 이 사건의 교훈이다.

---

## 4. `nvidia-smi topo -m` 읽는 법 — 측정 전에 미리 알 수 있는 것

서버 D 의 실제 출력(발췌):

```
        GPU0  GPU1 ... GPU7   NIC0  CPU Affinity   NUMA Affinity
GPU0     X    NV18     NV18   PHB   0-47,96-143    0
GPU1    NV18   X       NV18   PHB   0-47,96-143    0
```

### 4-1. GPU↔GPU 칸 — 연결 등급 (좋은 순서)

| 표기 | 의미 | 대략적 함의 |
|---|---|---|
| `NV#` | NVLink 직결, # = 링크 수 | **NV18 × 50GB/s = 900GB/s 급.** all-reduce/all-to-all 이 빠름 |
| `PIX` | 같은 PCIe 스위치 아래 | P2P 가능하나 PCIe 대역폭(~64GB/s)에 갇힘 |
| `PXB` | PCIe 스위치 여러 개 경유 | 조금 더 나쁨 |
| `PHB` | PCIe Host Bridge(CPU) 경유 | CPU 를 거침 — P2P 성능 크게 하락 |
| `NODE` | 같은 NUMA 노드 내, CPU 경유 | |
| `SYS` | **NUMA 노드 사이**(QPI/UPI)를 건너감 | 최악. GPU 간 통신이 CPU 인터커넥트를 탐 |

미리 알 수 있는 것: 전 쌍이 `NV#` 로 균일하면 TP all-reduce 가 싸고 GPU
배치 순서를 고민할 필요가 없다. 반대로 `SYS` 가 섞인 4+4 구조라면 TP=8 의
all-reduce 가 느려지므로 TP=4×DP=2 같은 분할이나 GPU 선택이 성능을 좌우한다.

### 4-2. NIC 열 — GPU↔네트워크 카드 거리

멀티노드 학습/추론에서 GPUDirect RDMA 는 GPU 와 NIC 가 같은 PCIe 스위치
(`PIX`)에 있을 때 최적이다. 우리 서버 D 는 전부 `PHB` — 단일 노드 추론이라
무관하지만, 노드 간 EP/DP 를 하려면 이 열이 병목 예고가 된다.

### 4-3. CPU / NUMA Affinity 열

`0-47,96-143 / NUMA 0` = 이 GPU 는 소켓 0 근처다. 의미:

- vLLM 워커·NCCL proxy 스레드를 그 코어들에 두면 host↔GPU 복사와 런치
  지연이 줄어든다 (`numactl`, taskset)
- pinned memory 를 반대쪽 NUMA 에 잡으면 H2D 복사가 UPI 를 건너간다
- **VM 에서 이 열이 N/A 로 나오면** (서버 C 가 그랬다) 토폴로지가 숨겨진
  것 — 스케줄러가 최적화를 못 하고, 성능 편차의 원인 후보가 하나 늘어난
  것으로 읽으면 된다

### 4-4. 실전 요약 — 새 서버를 받으면 3분 안에 볼 것

```bash
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv
nvidia-smi topo -m          # GPU 간 등급, NIC 거리, NUMA
systemd-detect-virt         # VM 인지 docker 인지
systemctl is-active nvidia-fabricmanager   # NVL5 는 FM 없으면 CUDA Error 802
```

이 네 줄이면 "이 서버에서 낼 수 있는 성능의 상한"과 "이전 측정과 비교해도
되는지"를 측정 전에 판단할 수 있다. 전부 `setup/fingerprint.sh` 에 들어 있다.

---

## 5. 한 줄 요약

- **MTP**: 디코딩만 가속(+8~38%). prefill 은 그대로, KV cache 7% 잠식.
  Input TPS 는 이 벤치에서 Output×8 인 파생 지표라 따로 읽지 말 것
- **EP8**: 단일 노드 NVLink 8-GPU 추론에서는 all-reduce 가 이미 싸서
  all-to-all 비용·부하 불균형만 남는다 → −4~−6%. EP 는 멀티노드용 도구
- **하드웨어 지문**: 같은 GPU 이름이라도 전력 상한·드라이버·가상화·패브릭
  공유로 −16%까지 갈린다. 수치는 지문과 짝으로만 의미가 있다
- **topo -m**: 측정 전에 통신 상한(NV#/SYS), RDMA 적합성(NIC 열),
  NUMA 최적화 가능성을 예고해 주는 지도다
