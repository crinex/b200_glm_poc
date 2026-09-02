# [원문] Blackwell 커널/시스템 최적화 서베이 (r-kernels, 2026-09-02)

## 0. 요약 결론
- 가장 큰 레버: (a) NVFP4 expert 전환, (b) MTP 다단계/EAGLE화, (c) DP-attention+EP 재구성. 커널 신규 작성보다 vLLM/FlashInfer 기존 경로를 켜는 쪽이 투자 대비 효율 높음.
- DSA 모델은 attention 1스텝 커널 시간이 MLA 대비 2.7배 [확인됨, vLLM GB300 블로그]. IndexCache(레이어 간 top-k 재사용)가 커널 작성 없는 가장 직접적 완화책.

## 1. 우선순위 표
| # | 기법 | 지금 시간이 가는 곳 | 기대 이득(근거) | vLLM 구현 경로 | 노력 |
|---|---|---|---|---|---|
| 1 | NVFP4 experts (nvidia/GLM-5.2-NVFP4) + FP8 KV | MoE GEMM 가중치 대역폭 (~740GB→~459GB) | [확인됨] SemiAnalysis: GLM-5 B200 FP8 TP8 1,947 → NVFP4 2.4~3.07× tok/s/GPU (8k/1k, conc256, SGLang MTP). 메모리 3× 축소 → resident 150→300+ | 체크포인트 교체. `VLLM_USE_FLASHINFER_MOE_FP4=1`. TP=4 로드 가능 → 2 replica/DP2 | 낮음. 온라인 quant 불가[추정] (ModelOpt 캘리브레이션 필요). GLM-5 NVFP4는 MoE-only 양자화, attention FP8/BF16 유지 [확인됨] |
| 2 | DP-attention(DP8)+EP8+DeepEP low-latency+DBO | TP=8 AR 2회/레이어 + attention 8-way 분할 | [확인됨] SGLang 문서 DSv3.2 B200 TP8/EP8/DP8 ~2,237 tok/s @88 conc (우리 2,251@128 유사 → 단순 전환 이득 [추정] 0~+20%). DBO는 device당 64~128 토큰 이상에서만 이득, 32에서 −27% | `--data-parallel-size 8 --enable-expert-parallel --all2all-backend deepep_low_latency --enable-dbo` (full CUDA graph 필수) | 중. KV는 DP당 독립이라 용량 병목 완화 없음 |
| 3 | MTP 다단계 + 수용률 개선 | 수용률 75%(τ≈1.75) | GLM-5는 3 MTP 레이어 파라미터 공유 학습 → accept 2.76 [확인됨 arXiv 2602.15763]. Dynamo GLM-5.2 레시피 draft 3, acceptance 2.69 [확인됨]. [추정] decode-bound +25~40% | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. 이슈 #28312(k=2 IMA)는 PIECEWISE 그래프로 우회 | 낮음 |
| 4 | IndexCache (DSA top-k 레이어 간 재사용) | 인덱서 logits+top-k(2048) 매 레이어 | [확인됨] DSA 1스텝=MLA 2.7×; TRT-LLM 커스텀 top-k 7.41×; GVR top-k 1.88× → TPOT −7.5%@100K. index_topk_freq=4 → [추정 −5~10% TPOT, 정확도 검증 필수] | `--hf-overrides '{"use_index_cache":true,"index_topk_freq":4}'` | 낮음 |
| 5 | AllReduce+RMSNorm(+quant) 퓨전, 저지연 AR | TP=8 레이어당 AR 2회 | [확인됨] vLLM 0.28 AR fusion 1.5~3× 커널(#51070); GB200 소메시지 AR 11→2.37µs 시 ITL −7~13% (arXiv 2607.16100). 8×B200 symm AR 비효율 #33459 | `--compilation-config '{"pass_config":{"enable_fi_allreduce_fusion":true}}'`; `VLLM_ALLREDUCE_USE_SYMM_MEM` 벤치 후 결정 | 낮음 |
| 6 | MoE 백엔드/SM 튠 | trtllm-gen FP8 MoE 140/148 SM | [확인됨] autotuner 오선택 이슈 #3537; SGLang GLM flashinfer_trtllm ≈+10%, CuTeDSL FP4 GEMM +4%. SM 수 조정 파라미터 [미확인] | trtllm-gen vs `VLLM_FLASHINFER_MOE_BACKEND=cutlass|throughput` 스윕 | 낮음~중 |
| 7 | Attention 커널 선택 | FLASHINFER_MLA_SPARSE | [확인됨] FlashMLA sparse decode B200 350 TFLOPS "미최적화"; FlashInfer 휴리스틱 부족 #35807. 기대 [추정] ±5% | `VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE` 비교 | 낮음 |
| 8 | KV 오프로드 (LMCache/offloading connector) | resident 150 한계 | [추정] 멀티턴/agentic만 유효; fp8_ds_mla+LMCacheMP 레시피 존재 | `--kv-transfer-config` | 중 |
| 9 | P/D 분리 (Dynamo) | 프리필이 디코드 배치를 끊음 | [확인됨] Dynamo GLM-5.2 B200 agg 176 → disagg 321 tok/s/GPU (NVFP4, 12 GPU) | 4+8 GPU 필요 → 8 GPU에 [추정] 부적합 | 높음 |
| 10 | 커널 신규 작성 (인덱서+top-k 퓨전, SwiGLU epilogue, permute 퓨전) | 인덱서 K 저장/양자화 소커널 | [확인됨] TRT-LLM: blockwise FP8 quant 퓨전 32~64%(op), 인덱서 K 저장 퓨전 e2e +3.5~13.4%, MQA MTP-3 커널 +10% | Gluon(Triton 3.6+ tcgen05/TMEM), CuTe DSL(sm_100a) → custom op | 높음 |

## 2. 도구 성숙도 (sm_100, 2026-09)
| 도구 | 상태 |
|---|---|
| Triton 3.6/3.7 (Gluon) | tcgen05 mma, TMEM, warp specialization [확인됨]; 일반 Triton 경로는 wgmma급 [추정] |
| CUTLASS 4 / CuTe DSL | sm_100a/103a, block-scaled grouped GEMM [확인됨]; FlashInfer 신규 커널이 CuTe DSL 추세 |
| DeepGEMM | SM100 전 레이아웃, UE8M0, grouped GEMM, MegaMoE(dispatch+GEMM+SwiGLU+combine 퓨전), 인덱서 MQA logits 커널 [확인됨]; vLLM이 인덱서 커널을 DeepGEMM에서 사용 |
| FlashMLA | sm90/sm100 sparse decode FP8 KV, B200 350 TFLOPS(prefill 1450) [확인됨] |
| FA4 | 2026-03 릴리스, 알파; MLA/DSA decode 경로 없음 [확인됨] |
| ThunderKittens 2.0 | Blackwell, MXFP8/NVFP4 [확인됨] |
| TileLang 0.1.12 | tcgen05 GEMM 추가 [확인됨], MoE/attention 성숙도 낮음 [추정] |
| cuTile DSL | CUDA 13.2 stable [확인됨] |
| FlashInfer JIT | vLLM/SGLang 기본 Blackwell 백엔드; trtllm-gen FP8/FP4 MoE, AR+RMSNorm 퓨전, sparse MLA [확인됨] |

## 3. 참고 수치 (유사 ISL/OSL)
- TRT-LLM DSv3.2 B200 NVFP4 8k/1k batch256 TP8/EP8: 1,077 tok/s/GPU, TPOT 98.5ms [확인됨]. 우리 FP8 TP8: 281 tok/s/GPU @55ms.
- SemiAnalysis GLM-5 B200: FP8 TP8 1,947 tok/s/GPU, NVFP4 TP4 4,091 (conc256, SGLang+MTP) [확인됨].
- Dynamo GLM-5.2 B200 NVFP4 agg 176 tok/s/GPU @57 tok/s/user [확인됨].
- vLLM GB300 DSv3.2 NVFP4 TP2 2k/1k: 2,816 tok/s/GPU [확인됨].

## 4. 커널 작성 없는 Quick Wins
1. nvidia/GLM-5.2-NVFP4 + `VLLM_USE_FLASHINFER_MOE_FP4=1`, KV fp8 유지
2. MTP num_speculative_tokens 2/3 스윕 (PIECEWISE cudagraph)
3. `--max-num-seqs` 256~512 + cudagraph_capture_sizes 큰 배치 포함
4. enable_fi_allreduce_fusion + custom AR vs symm_mem
5. IndexCache index_topk_freq 2/4 + 정확도 확인
6. MoE 백엔드 스윕 + FlashInfer 0.6.17+ autotuner 재실행
7. NVFP4 여유 시 TP4×2 replica 또는 DP8+EP8+DBO

## 출처
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs
- https://vllm.ai/blog/2026-02-13-gb300-deepseek
- https://inferencex.semianalysis.com/blog/b200-glm5-nvfp4-vs-h200-fp8-3-6x-perf-per-dollar
- https://docs.nvidia.com/dynamo/v1.4.0/recipes/glm-5-2
- https://huggingface.co/nvidia/GLM-5.2-NVFP4
- https://docs.sglang.io/basic_usage/deepseek_v32.html
- https://github.com/sgl-project/sglang/issues/17526
- https://github.com/vllm-project/vllm/issues/31473
- https://docs.vllm.ai/en/latest/features/index_cache/
- https://arxiv.org/abs/2604.22312
- https://docs.vllm.ai/en/latest/design/dbo/
- https://docs.vllm.ai/en/stable/design/fusions/
- https://arxiv.org/html/2607.16100v1
- https://github.com/vllm-project/vllm/issues/33459
- https://github.com/vllm-project/vllm/issues/35807
- https://github.com/flashinfer-ai/flashinfer/issues/3537
- https://github.com/deepseek-ai/FlashMLA
- https://github.com/deepseek-ai/DeepGEMM
- https://arxiv.org/pdf/2602.15763
- https://github.com/vllm-project/vllm/issues/28312
- https://vllm.ai/blog/2026-04-22-fp8-kvcache
- https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis
- https://github.com/triton-lang/triton/releases
- https://tridao.me/blog/2026/flash4/
- https://github.com/HazyResearch/ThunderKittens
- https://github.com/tile-ai/tilelang
- https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- https://docs.lmcache.ai/recipes/deepseek_v4_flash.html
- https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/
