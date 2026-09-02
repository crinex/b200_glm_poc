# [원문] Modular MAX / Mojo × GLM-5.2-FP8 on 8×B200 조사 (r-mojo, 2026-09-02)

결론: GLM-5.2-FP8 8×B200은 MAX 26.5+에서 공식 검증됨(대체 서빙 후보로 시험 가치). 커스텀 커널 언어로는 Mojo보다 CuTe DSL/Gluon이 vLLM 통합에 현실적.

## 1. 상태 요약
### (a) MAX 서빙 프레임워크
- [확인됨] MAX 26.5(2026-08-11) GLM-5.2(`GlmMoeDsaForCausalLM`, cross-layer index sharing) 지원 + NextN MTP 자동 활성. 나이틀리(26.6.0.dev, 09-02) `zai-org/GLM-5.2-FP8`·`nvidia/GLM-5.2-NVFP4` 8×B200 검증 명시. DSA indexer 개선(B200 레이어당 0.89ms→0.10ms, batch 8/76k ctx). https://max.modular.com/releases/v26.5/ , https://max.modular.com/releases/nightly
- [확인됨] 26.3: SM100 sparse MLA decode(qbf16/FP8 KV), EP dispatch/combine, grouped dynamic NVFP4 quant, "NVFP4 grouped matmul B200에서 FlashInfer보다 빠름(Kimi K2.5)". 26.4: DSv3.2 long-context sparse MLA, GLM-5/5.1 FP8/NVFP4, TP+EP MoE. 26.5: GEMM+Bias+SwiGLU 퓨전(1.06~1.12x), low-latency all-reduce(1.1~1.68x), FP8 latent KV.
- [확인됨] 나이틀리 DP-EP/TP+EP 추가, EP dispatch 토큰 절반 드롭→NaN 버그 수정(B200당 ~74개 초과 expert) — 256 expert 모델 직접 해당. **EP 경로 성숙 중**.
- [확인됨] MAX vs vLLM/SGLang/TRT-LLM 공개 비교 **MoE/B200 없음**. GTC 2026 블로그 DSv3 B200 데모만. 제3자(Spheron 2026-05) Llama-3.1-8B/H100 +16%이나 "MoE는 MAX 미최적화, vLLM 권장" 명시.
- [확인됨] AA에서 Modular Cloud GLM-5.2 median 167 tok/s(Databricks 313, Baseten 260) — 하드웨어 불명.
- [확인됨] 라이선스: ModCon(2026-08-18) Mojo 1.0 Apache 2.0. MAX Community License(source-available), NVIDIA 상용 프로덕션 무료, 지원 의무 없음.
- **판정**: `max serve zai-org/GLM-5.2-FP8` 8×B200 공식 동작·검증 [확인됨]. vLLM 0.28 대비 성능 우위 **미입증**, EP 정합성 버그 최근까지. A/B 후보로 충분(하루 셋업), 즉시 교체 근거 없음.

### (b) Mojo GPU (sm_100)
- [확인됨] stdlib `gpu/compute/arch/tcgen05`, TMEM, TMA, cluster, 2-CTA. Blackwell matmul Part 3: BF16 GEMM 1,429 TFLOPS(SOTA ~81%)→85%. Structured Mojo Kernels Part 1(2026-03): ~1,770 TFLOPS, 코드 48% 감소.
- [추정] "cuBLAS 초과" 주장은 원문 직접 비교 없음.
- [확인됨] GTC 2026: CUTLASS conv2d Mojo 포팅 130.7 TFLOPS 동등, 770줄 vs 3k줄.
- [확인됨] 오픈소스 커널(modular/modular `max/kernels`) SM100: blockwise FP8 matmul, MXFP4/MXFP8/NVFP4 grouped matmul(W4A8), sparse MLA decode, DSA indexer(bitonic top-k), EP dispatch/combine, GEMM+SwiGLU. **후보 커널 5종이 Mojo로 이미 존재**.
- [확인됨] FA4/FlashMLA/DeepGEMM 대비 공개 벤치 없음.

### (c) vLLM/PyTorch 통합
- [확인됨] `max.experimental.torch.CustomOpLibrary`: Mojo 소스→`.mojoc` 컴파일, torch 호출. `pip install max[all]` 무거움.
- [확인됨] 한계(포럼 2025-05~07): 동적 shape 미숙→shape별 재컴파일, 스칼라 미지원, illegal address. `experimental` 유지.
- [확인됨] vLLM: `torch.library.custom_op` ~36µs/호출(직접 `torch.Library` ~10µs). `@CustomOp.register`/`register_oot`, attention backend는 platform plugin `get_attn_backend_cls`. CUDA graph 호환 언급 없음 [추정: 호스트측 MAX 런타임 호출 시 capture 실패 가능].
- **판정**: decode 경로(CUDA graph 필수)에 넣으려면 MAX 런타임 우회해 Mojo→PTX/cubin 직접 로드 필요 가능성 [추정].

### (d) 경쟁 언어
- Triton 3.7 Gluon: tcgen05/TMEM/TMA multicast/2CTA, mxfp8/nvfp4 block-scaled. 코어 Triton은 tcgen05 제한적.
- CUTLASS 4 / CuTe DSL: Blackwell grouped GEMM(FP8 blockwise), block-scaled MoE 예제. FlashInfer가 CuTe DSL로 SM100 커널(router GEMM, DSA indexer top-k radix, sparse MLA) JIT — **vLLM 0.28 기본 경로와 동일 스택**.
- TRT-LLM 블로그(2026-08-27): DSA sparse MLA TMALDG.Gather4(+47% TPS/GPU), indexer top-k radix-select 7.41x, e2e 25~40%(저지연). 참조 설계.
- ThunderKittens 2.0(2026-01) C++ 템플릿. TileLang tcgen05/2SM, SM100 FA·block-scaled grouped GEMM 예제.
- MLSys 2026 FlashInfer 콘테스트 DSA 트랙 1위 **Triton** 구현(37x). https://github.com/Dogacel/DeepSeek-Sparse-Attention-Kernels

## 2. 결정 표
| 후보 커널 | 추천 언어 | 이유 | 성숙도 |
|---|---|---|---|
| DSA indexer + top-k gather 퓨전 | Triton/Gluon(1), CuTe DSL(2) | 메모리바운드·정수 로직, tcgen05 불필요. 콘테스트 1위 Triton. `torch.library` 즉시 삽입 | 높음 |
| Sparse MLA decode attention | CuTe DSL(FlashInfer JIT 포크) | FlashInfer sparse MLA가 CuTe DSL·기본 백엔드. Gather4/TMA 필요 | 중상 |
| MoE router/dispatch/combine 퓨전 | Triton(+CUDA) | 소형 GEMM+softmax+top-k+scatter | 높음 |
| Grouped FP8(block-128) GEMM + SwiGLU | CuTe DSL 또는 DeepGEMM 패치 | tcgen05·TMEM·2CTA·블록 스케일 — CUTLASS 레퍼런스 완성 | 중 |
| Sampling 퓨전 | Triton | 단순, CUDA graph 친화 | 높음 |
| Mojo 전반 | — | 커널 자산 우수하나 vLLM 통합 experimental, 동적 shape·CUDA graph 미검증. **MAX 통째 사용 시만** 가치 | 통합 낮음 |

순위: Triton(Gluon) ≥ CuTe DSL > TileLang > ThunderKittens > Mojo(vLLM 플러그인).

## 3. 최소 PoC (indexer+top-k 퓨전 1개, vLLM 0.28 e2e) — 7~10 영업일
1. 베이스라인 측정(0.5일): nsys/vLLM profiler로 decode step 중 indexer 커널 시간
2. 커널 작성(3~5일): Triton `index score(MQA) → causal clamp → radix/bitonic top-k(2048) → block-table gather`. 참조: Dogacel, TRT-LLM radix-select, MAX bitonic scorer
3. 등록(1일): `torch.library.custom_op`+`register_fake`, `vllm/v1/attention/backends/mla/indexer.py` top-k 호출 env flag 교체. top-k 집합 일치율 검증
4. CUDA graph/torch.compile 검증(1일)
5. 벤치(1일): conc 1/8/32, ctx 8k/76k/256k. 목표 레이어당 indexer ≥3x, e2e TPOT 5~10% [추정]
6. (병행 1일) MAX 대조군: `max serve zai-org/GLM-5.2-FP8` TP8(+EP) 동일 워크로드
Mojo로 동일 PoC 시 CustomOpLibrary 이슈 조사만 +1주 [추정].

## 4. 커뮤니티 사례
- [확인됨] DSv3급 MoE MAX B200 제3자 재현 벤치 없음. Modular Cloud 자체(MiniMax M3, GLM 5.3) 외 없음.
- [확인됨] vLLM/SGLang/TRT-LLM은 InferenceX·Lambda 다수 수치(GLM-5.2 HGX B200 1,264 tok/s @32 conc). https://lambda.ai/inference-models/zai-org/glm-5.2

## 출처
- https://max.modular.com/releases/nightly
- https://max.modular.com/releases/v26.5/
- https://max.modular.com/releases/v26.3/
- https://www.modular.com/blog/modcon-announcements
- https://www.modular.com/legal/community
- https://www.modular.com/blog/modular-26-5-mojo-1-0-is-here
- https://max.modular.com/develop/custom-kernels-pytorch
- https://www.spheron.network/blog/modular-max-mojo-gpu-cloud-llm-inference/
- https://github.com/triton-lang/triton/releases/tag/v3.7.0
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs
- https://github.com/Dogacel/DeepSeek-Sparse-Attention-Kernels
- https://docs.vllm.ai/en/stable/design/custom_op/
- https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance
- https://www.modular.com/blog/structured-mojo-kernels-part-1-peak-performance-half-the-code
- https://docs.modular.com/max/api/kernels/linalg/matmul/gpu/sm100_structured/blockwise_fp8/blockwise_fp8_matmul_kernel/
- https://forum.modular.com/t/no-active-mlir-context-with-new-customoplibrary-torch-integration/1491
