# [원문] vLLM·SGLang 외 서빙 스택 조사 (r-stacks, 2026-09-02)

## 0. 핵심 결론
- 베이스라인(vLLM 0.28 TP8, MTP spec-1, 2,251 tok/s/노드 = **281 tok/s/GPU @conc128, 17.6 tok/s/user**)은 공개 프론티어 대비 4~6배 낮음. 같은 8k/1k·FP8·B200에서 SGLang+MTP 1,300~1,750 tok/s/GPU, TRT-LLM(DSv3.2 FP8, TP8/EP8/ADP/MTP1, batch 256) 1,077 tok/s/GPU [확인됨]. "프레임워크 교체"보다 **spec 토큰 수(1→5~6)·ADP/EP·chunked-prefill·batch 크기** 재설정만으로 큰 폭 개선 여지 [추정].
- 베이크오프: (1) TensorRT-LLM 1.3.0rc25+ (FP8, DEEPGEMM MoE, TP8/EP8, MTP1, ADP 시험), (2) SGLang + NVFP4 + EAGLE-MTP. Dynamo는 단일노드 PD(4+4) 옵션.

## 1. 프레임워크별 준비도
| 스택 | GLM-5.2 지원 | Blackwell 커널 | MTP | EP / Attention-DP | 성숙도 | 기대 |
|---|---|---|---|---|---|---|
| TensorRT-LLM (PyTorch, trtllm-serve) | ✅ GLM-5/5.2/5.3 명시 [확인됨]. 배포 가이드는 GLM-5까지 | FP8: DEEPGEMM MoE(기본 CUTLASS override 필수), NVFP4: TRTLLM-gen. DSA: CuTe DSL radix/GVR top-k(topk=2048 전용), FP8 sparse-MLA [확인됨] | ✅ MTP nextn 1. **NVFP4에서는 MTP 미지원** [확인됨] | EP8 ✅. GLM 계열 ADP·disagg·KV reuse "Untested" [확인됨] | rc. GLM-5.1-FP8 툴콜 출력 붕괴 #15295(H200/CUTLASS) 미해결 | DSv3.2 동형 1,077 tok/s/GPU → 우리 대비 ~3~4× [추정] |
| NVIDIA Dynamo | 엔진 비의존. GLM-5.2/5.3 레시피 = SGLang NVFP4, B200 4~12 GPU, DTP4/DEP4, MTP depth 3, HiCache [확인됨] | 위임 | 위임 | PD 분리 + KV-aware 라우팅. 단일노드 4+4 예시 | 1.0 GA. Baseten GLM-5.2 PD 분리 "2× tok/s" 자사 주장 | ±0~30% [추정] |
| TokenSpeed (LightSeek, 2026-05) | GLM-5/5.2/5.3, DSv3.2/V4 레시피 | B200/B300/GB200, FP8·NVFP4·MXFP4 | ✅ | 존재 | 신생. vLLM `TOKENSPEED_MLA` 백엔드로 흡수 중. "TRT-LLM 대비 +11%" 자사 주장 | 3순위 |
| LMDeploy | GLM-5 미등재 → 제외 | | | | | |
| KTransformers | GLM-5.2 Day-0이나 CPU 오프로드 전용 → 제외 | | | | | |
| llm-d / Aphrodite / xLLM | 단일노드 무의미 / 열세 / Ascend 중심 → 제외 | | | | | |

## 2. Z.ai 공식 권고
- 지원: SGLang ≥0.5.13.post1, vLLM ≥0.23.0, Transformers, KTransformers, Unsloth, xLLM. **TRT-LLM 언급 없음** [확인됨].
- MTP: 학습·추론 MTP steps 7, IndexShare+KVShare, 평균 수락 길이 5.47 [확인됨]. vLLM B200 레시피: `--kv-cache-dtype fp8_e4m3 --tensor-parallel-size 8 --speculative-config.num_speculative_tokens 5`, DeepGEMM 필수, MTP 수락률 버그 PR #45895 수정 [확인됨]. → 우리 spec-1은 권고(5)의 1/5.
- 아키텍처: IndexShare = 4레이어당 인덱서 1개(`indexer_types` 21 full + 57 shared). DSv3.2 코드 경로 그대로 쓰는 엔진은 이 필드 처리 확인 필요.
- Blackwell 주의: vLLM GB300(aarch64) seq>4096 출력 붕괴 #47827 "not planned" 종결. x86 B200 재현 미확인.

## 3. 공개 수치 (8k/1k)
| 소스 | 모델/정밀도 | 스택·config | 결과 |
|---|---|---|---|
| InferenceX 2026-05 | GLM-5 FP8 B200 TP8 SGLang+EAGLE MTP | conc 32/64 | 1,297 / 1,619 tok/s/GPU (38/24 tok/s/user) |
| InferenceX spec 비교 | GLM-5/5.1 FP8 B200 8k/1k | MTP on/off | 55 tok/s/user: 1,249 vs 520; 27 tok/s/user: off 우세(2,246 vs 1,757) → 고동시성에서 MTP 이득 감소 |
| InferenceX 2026-05 | GLM-5 NVFP4 B200 TP4 SGLang MTP | conc 128 | **4,116 tok/s/GPU** (17.6 tok/s/user) — 우리와 같은 interactivity에서 ~14× |
| TRT-LLM 블로그 | DSv3.2 (FP8 MoE+NVFP4 o_proj) B200 | TP8/EP8/ADP/MTP1, batch 256 | 1,077 tok/s/GPU, 11.3 tok/s/user; min-latency TP4/MTP3 312 tok/s/user |
| InferenceX v2 2026-02 | DS R1 B200 8k/1k | TRT-LLM vs SGLang FP8 단일노드 | TRT-LLM 우세(정성) |
| InferenceX 2025-12 | DS R1 NVFP4 B200 TP4/EP4 SGLang 0.5.6 | conc 128 | 5,145 tok/s/GPU |
| MLPerf v6.0 | DS-R1 8×B200 TRT-LLM NVFP4 ADP EP | offline/server | ~7.3k / 6.5k tok/s/GPU (형상 상이) |
| SGLang 블로그 2026-07 | GLM-5.2-NVFP4 8×B300 | EAGLE 5 steps/6 draft, 80k ISL | BS1 500+ tok/s/user |

## 4. 양자화
- nvidia/GLM-5.2-NVFP4 (2026-06-25, ModelOpt 0.46, MoE 전문가 선형층만 NVFP4; shared expert·attention·dense·MTP는 FP8/BF16). 정확도: GPQA-D 89.5→89.4, SciCode 49.9→49.0, IFBench/AA-LCR/τ² 소폭 상승 [확인됨].
- 반론: umans.ai(2026-07) 장기 agentic 체인 CoT 붕괴 관찰(PTQ). Baseten은 자체 NVFP4로 BFCL 동등 확인 후 프로덕션. → agentic이면 자체 롱호라이즌 평가 필수 [추정].
- FP8→NVFP4 재양자화는 2중 손실 위험 → 공식 nvidia 체크포인트 권장 [추정].
- KV: FP8 KV 표준. NVFP4 KV는 미성숙 → FP8 KV 유지.
- TRT-LLM NVFP4 경로 MTP 불가 → NVFP4 실험은 SGLang이 적합 [확인됨].

## 5. 권고 베이크오프
1. TRT-LLM FP8 1.3.0rc25+: `--tp_size 8 --ep_size 8 --max_batch_size 128~256 --max_num_tokens 8192`, YAML `moe_config.backend: DEEPGEMM`, `kv_cache_config.dtype: fp8`, `speculative_config: {decoding_type: MTP, num_nextn_predict_layers: 1}`, `enable_attention_dp` on/off. 목표 1,000+ tok/s/GPU. 리스크: IndexShare 처리·FP8 출력 붕괴 → 툴콜 정확도 스모크 선행.
2. SGLang + nvidia/GLM-5.2-NVFP4 + EAGLE-MTP(5 steps/6 draft), TP4 또는 DTP4/DEP4. 목표 3,000~4,000 tok/s/GPU. 리스크: 품질.
3. 보조: Dynamo 4+4 PD 분리 A/B. TokenSpeed 3순위.
- 당장 vLLM에서: spec tokens 1→5, EP, chunked-prefill/batch 상향, DeepGEMM 확인, PR #45895 포함 여부.

## 출처
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md
- https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-glm-5-on-trtllm.html
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs
- https://github.com/NVIDIA/TensorRT-LLM/releases ; https://github.com/NVIDIA/TensorRT-LLM/issues/15295
- https://huggingface.co/zai-org/GLM-5.2-FP8 ; https://huggingface.co/blog/zai-org/glm-52-blog ; https://recipes.vllm.ai/zai-org/GLM-5.2
- https://huggingface.co/nvidia/GLM-5.2-NVFP4 ; https://blog.umans.ai/blog/glm-5-2-nvfp4-not-worth-serving/ ; https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/
- https://inferencex.semianalysis.com/blog/b200-glm5-nvfp4-vs-h200-fp8-3-6x-perf-per-dollar ; https://inferencex.semianalysis.com/compare-spec-decode/glm-5-1-b200-fp8-mtp-vs-none ; https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs ; https://inferencex.semianalysis.com/blog/sglang-0-5-6-b200-deepseek-r1-fp4-up-to-1-8x
- https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.2 ; https://docs.nvidia.com/dynamo/dev/recipes/glm-5-nvfp4
- https://www.lmsys.org/blog/2026-07-13-glm52-optimization/ ; https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2
- https://nebius.com/blog/posts/mlperf-inference-v6-0-results
- https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html ; https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.2-Tutorial.md ; https://lightseek.org/tokenspeed/recipes/models
- https://github.com/vllm-project/vllm/issues/47827
