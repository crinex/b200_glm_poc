# [원문] SGLang × GLM-5.2-FP8 8×B200 조사 (r-sglang, 2026-09-02)

요약: GLM-5.2는 SGLang 공식 지원·튠 완료. 우리 워크로드(디코드·KV 용량 바운드)의 직접 레버는 DP attention(KV 중복 제거)과 MTP 스텝 확대. 공개 벤치 기준 SGLang FP8 TP8 단순 구성은 vLLM 결과(2,251)를 확실히 넘지 못함.

## 1. 지원 현황 [확인됨]
- GLM-5 day-0(2026-02), GLM-5.2(glm_moe_dsa) v0.5.14(06-26), v0.5.15(07-10) Blackwell 튠, v0.5.15.post1 DSA 패치. 최신 v0.5.18(08-22): torch 2.13, flashinfer 0.6.17, sgl-kernel 0.4.6.post1.
- 모델카드 요구: sglang ≥ v0.5.13.post1, transformers ≥ 5.3.
- (G)B200 트래킹 #19380: FP8/NVFP4 × MTP × Agg/Disagg 전 조합 Functional+Baseline 완료, Round-1 최적화 진행 중. B200 FP8 agg prefill host-bound(piecewise CUDA graph for NSA로 완화). FP8 GSM8K 0.950.
- GLM-5 쿡북 B200 FP8: `--tp 8 --ep 1 --quantization fp8 --attention-backend dsa --dsa-decode-backend trtllm --dsa-prefill-backend trtllm --moe-runner-backend flashinfer_trtllm --enable-flashinfer-allreduce-fusion --mem-fraction-static 0.9`
- GLM-5.2 쿡북: Blackwell DSA KV fp8_e4m3 자동. 저지연(tp8)/균형(tp8+dp2 dp-attn)/고처리량(tp4 ep2 또는 dp-attn+DeepEP). MTP 권장 `EAGLE --speculative-num-steps 5 --speculative-num-draft-tokens 6`(accept 5+).

## 2. Blackwell 커널 기본값 [확인됨]
- DSA: BF16 KV → prefill flashmla_sparse + decode trtllm; FP8 KV → 둘 다 trtllm. top-k `--dsa-topk-backend sgl-kernel`. <2048 토큰은 dense MHA 자동 전환.
- MoE: flashinfer_trtllm(FP8/NVFP4), 대안 flashinfer_cutlass/deep_gemm/triton. v0.5.18 "MoE deferred finalize" 기본 ON.
- GLM-5.2 전용(v0.5.15, 7월 블로그): Spec V2 zero-overhead(+11% TPS), TopK-V2(80K ISL 2.33×), indexer prologue fusion(디코드 커널 12→4, BS128 ~5%), IndexShare MTP(드래프트 비용 1.9×↓), v0.5.18 kv_len ≤ 2048 인덱서 스킵. 블로그 실측은 NVFP4·8×B300.
- FA4는 MLA/DSA 무관. torch.compile "out of maintenance".

## 3. vLLM과 다른 점
| 후보 | 기대효과 | 근거 | 노력/리스크 |
|---|---|---|---|
| DP attention (`--enable-dp-attention --dp-size 2~8`) | TP8 MLA는 KV가 8 GPU 중복 → DP=8이면 GPU당 자기 요청 KV만 → 상주 ~150 한계 이론상 최대 8× 완화 [추정: 실측 없음] | DP-attn 문서, V3.2 `--tp 8 --dp 8 --enable-dp-attention` 레시피 | 중/중상. 저동시성 지연 악화. GLM-5.2 DP-attn+EAGLE 첫 배치 데드락 #34582(2노드, 미해결) |
| EP + DeepEP (`--ep 8 --moe-a2a-backend deepep --deepep-mode auto`) | GPU당 32 expert → MoE 가중치 읽기 8×↓. DeepEP는 ep==tp 필수 | EP 문서; GB200 대규모 EP 13.4k tok/s/GPU(멀티노드) | 중. 단일노드 이득 미공개. EPLB 가능 |
| MTP 스텝 확대 (EAGLE=NEXTN, steps 3→5, draft 4→6) | MTP 3스텝만으로 conc128 +16%(1,519→1,764). accept 5+면 추가 | InferenceX B200 FP8 SGLang v0.5.12 8k/1k; 쿡북 | 낮음. **spec ON 시 `--max-running-requests` 48 강제 → 명시 필수**. Spec V2/overlap은 topk=1만 |
| Overlap scheduler + Spec V2 | 기본 ON, +11% | v0.5.15 | 없음 |
| KV fp8_e4m3 + trtllm | 메모리 절반 | | fp8 KV 시 prefill trtllm 강제(#20163) → 정확도 스팟체크 |
| HiCache | 멀티턴만 유효, 8.2k 단발 랜덤엔 없음 | | |
| PD disaggregation | 단일노드 손해, 버그 다수 | | 비권장 |
| NVFP4 체크포인트 | 동일 interactivity에서 FP8 대비 2× tok/s/GPU(GLM-5) | InferenceX | FP4+EAGLE IMA #30209, flashinfer_trtllm NaN #30989 이력 |

## 4. 공개 벤치 (B200, 2026)
- InferenceX 2026-05-20 GLM-5-FP8 SGLang v0.5.12 TP8 8k/1k: MTP(3스텝) conc128 13.78 tok/s/user, TPOT 72.6ms → 노드 출력 ≈1,764 tok/s; conc256 ≈3,040(TPOT 84ms); 비MTP conc128 ≈1,519. **사이트의 tok/s/GPU는 입력+출력 합산.** → vLLM 0.28 2,251@128은 당시 SGLang FP8 TP8보다 이미 높음.
- Lambda GLM-5.2-FP8 HGX B200 8k/2k conc32 spec 없음: SGLang 1,454(TPOT 19.4) vs vLLM 1,264(TPOT 24.4) → +15%, TTFT 5.4s vs 1.9s로 SGLang 나쁨.
- 쿡북 B200 FP8 플레이그라운드 수치(conc64 TPOT 17.7, conc256 32.6)는 "simulated accept length" → 상한 참고 [추정].
- 커뮤니티 8×B200 GLM-5-FP8(1k/1k, conc100, EAGLE 3): 1,370 out tok/s, TPOT 34ms, accept 3.52; "EAGLE 없으면 디코드 2.6× 느림", DeepGEMM JIT 콜드스타트 10분+.
- Dynamo GLM-5.2 레시피: SGLang+NVFP4+FP8 KV, MTP draft 3(accept 2.69).
- 8k/1k DP-attn/EP GLM-5.2 FP8 단일노드 공개 수치 없음 → 자체 측정 필요.

## 5. 마이그레이션 리스크
- 빌드: 현재 `lmsysorg/sglang:latest`(cu12) / `dev-cu13`·`nightly-cu134` 이미지. v0.5.18 torch 2.13.
- 핀: sglang ≥ 0.5.15.post1, transformers ≥ 5.3, 가급적 v0.5.18.
- FP8 KV + DSA: trtllm prefill 강제(#20163). flashmla_kv 정확도 버그 04-15 수정(#22723). 회귀 테스트 권장.
- MTP + DP attention: 멀티노드 데드락 미해결(#34582). dp2부터 단계 검증.
- Spec ON 시 max-running-requests 48 강제, DeepGEMM JIT 10분+, `SGLANG_CACHE_DIR` 재컴파일(v0.5.18 breaking).

## 6. 권장 런치 (단계별)
Step A — 베이스라인(TP8 + MTP 3):
```
docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
  -v $HF_HOME:/root/.cache/huggingface lmsysorg/sglang:dev-cu13 \
  sglang serve --model-path zai-org/GLM-5.2-FP8 --tp 8 --ep 1 \
  --quantization fp8 --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
  --moe-runner-backend flashinfer_trtllm --enable-flashinfer-allreduce-fusion \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 256 --cuda-graph-max-bs 256 \
  --chunked-prefill-size 32768 --mem-fraction-static 0.88 \
  --context-length 32768 --tool-call-parser glm47 --reasoning-parser glm45 \
  --host 0.0.0.0 --port 30000
```
Step B — KV 병목 해소: `--dp-size 8 --enable-dp-attention --ep 8 --moe-a2a-backend deepep --deepep-mode auto` (안 되면 dp 2→4→8). 필요 시 `--enable-dp-lm-head`, `--enable-eplb`.
Step C — MTP 확대: accept 로그 확인 후 `--speculative-num-steps 5 --speculative-num-draft-tokens 6`.
Step D — NVFP4: `--model-path nvidia/GLM-5.2-NVFP4 --quantization modelopt_fp4`, `--speculative-moe-runner-backend triton` 병행.

## 출처
- https://github.com/sgl-project/sglang/releases
- https://huggingface.co/zai-org/GLM-5.2-FP8 , https://huggingface.co/nvidia/GLM-5.2-NVFP4
- https://github.com/sgl-project/sglang/issues/19380
- https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5 , https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2 , https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3
- https://www.lmsys.org/blog/2026-07-13-glm52-optimization/
- https://docs.sglang.io/basic_usage/deepseek_v32.html , https://docs.sglang.io/docs/advanced_features/attention_backend , https://docs.sglang.io/docs/advanced_features/expert_parallelism , https://docs.sglang.io/docs/advanced_features/server_arguments
- https://sgl-project-sglang-93.mintlify.app/distributed/data-parallelism
- https://github.com/sgl-project/sglang/issues/34582 , /issues/18595 , /issues/20163 , /issues/22723 , /issues/30209 , /issues/30989
- https://www.lmsys.org/blog/2025-10-14-sa-inference-max/
- https://inferencex.semianalysis.com/blog/mi355x-glm5-fp8-sglang-40-cheaper-than-b200 , https://inferencex.semianalysis.com/compare-precision/glm-5-1-b200-fp4-vs-fp8
- https://lambda.ai/inference-models/zai-org/glm-5.2
- https://gist.github.com/BenHamm/dcd09f595fef141567a39582f502cef4
- https://docs.nvidia.com/dynamo/v1.4.0/recipes/glm-5-2
- https://hub.docker.com/r/lmsysorg/sglang/tags
