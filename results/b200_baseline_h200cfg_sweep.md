# B200 기준선 — 지정 서빙 구성, MTP·EP 미적용

측정일 2026-09-01. 이전 인스턴스에서 측정하여 수치만 보존한 것이다
(인스턴스가 사라져 원본 result.json 은 없다).

## 서버 구성

```
vllm serve /workspace/models/GLM-5.2-FP8
  --host 0.0.0.0 --port 30269
  --tensor-parallel-size 8
  --async-scheduling
  --kv-cache-dtype fp8
  --served-model-name glm-5.2-fp8
  --max-model-len 16384
  --max-num-seqs 256
  --gpu-memory-utilization 0.93
  --enable-chunked-prefill
  --enable-prefix-caching
  --trust-remote-code
```

- `--ignore-eos 1` 은 vllm serve 인자가 아니므로 제외 (벤치가 요청별로 전송)
- MTP 미적용, Expert Parallel 미적용 (기본값 = EP1)
- KV cache 1,571,008 tokens
- mnbt 는 미지정 → `max_model_len` 과 같은 16384 로 자동 설정됨
- all-reduce (tp:0): `['CUSTOM', 'SYMM_MEM', 'PYNCCL']`
- MoE 백엔드: FLASHINFER_TRTLLM (자동), SM 140/148 점유

## 워크로드

gen8k 1,024 고유 sheet · ISL ≈ 8,200 · OSL 1,024 · sweep 4~256
요청 수는 conc×4 (상한 1,024): 16/32/64/128/256/512/1024

## 결과

|ISL|OSL|GPU|Precision|TP|Concurrency|TPOT(ms)|Interactivity|Input TPS/server|Output TPS/server|Total TPS/GPU|
|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|4|11.27|88.73|2,437|304|343|
|8200|1024|B200 (8)|FP8|8|8|13.11|76.29|4,553|569|640|
|8197|1024|B200 (8)|FP8|8|16|15.78|63.37|7,365|920|1,036|
|8197|1024|B200 (8)|FP8|8|32|20.14|49.65|11,274|1,408|1,585|
|8197|1024|B200 (8)|FP8|8|64|34.24|29.21|14,181|1,772|1,994|
|8198|1024|B200 (8)|FP8|8|128|60.58|16.51|16,094|2,010|2,263|
|8198|1024|B200 (8)|FP8|8|256|84.96|11.77|16,069|2,007|2,259|

Output TPS 최고점: conc=128 에서 2,010 tok/s

## 참고

Input TPS 는 Output TPS × (ISL/OSL) = Output TPS × 8.0 과 오차 0.1% 이내로
일치한다. prefill 성능 지표가 아니다 (FINDINGS.md §4 참조).
실제 prefill 속도는 서버 로그의 `Avg prompt throughput` (측정 중 24,577 tok/s).
