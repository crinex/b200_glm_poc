"""
GLM-5.2 FP8 추론 벤치마크
사용: python3 bench/benchmark.py --base-url http://localhost:8000
"""
import argparse
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "openai", "-q"])
    from openai import OpenAI

def run_single(client, prompt: str, max_tokens: int = 256) -> dict:
    start = time.perf_counter()
    first_token_time = None
    tokens = 0

    stream = client.chat.completions.create(
        model="glm-5.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if chunk.choices[0].delta.content:
            tokens += 1

    total = time.perf_counter() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else 0

    return {
        "total_s": total,
        "ttft_ms": ttft,
        "tokens": tokens,
        "tps": tokens / total if total > 0 else 0,
    }


def benchmark(base_url: str, concurrency: int = 1, n: int = 10):
    client = OpenAI(base_url=f"{base_url}/v1", api_key="none")
    prompts = [
        "B200 GPU의 Blackwell 아키텍처가 H100 대비 어떤 점에서 뛰어난지 설명해줘.",
        "Multi-head Latent Attention(MLA)의 KV 캐시 압축 원리를 설명해줘.",
        "vLLM에서 Speculative Decoding이 어떻게 동작하는지 알려줘.",
        "FP8 양자화가 모델 추론 속도에 미치는 영향을 설명해.",
        "FlashAttention 3의 핵심 개선점이 뭔지 설명해줘.",
    ]

    print(f"\n{'='*50}")
    print(f"  GLM-5.2 FP8 벤치마크")
    print(f"  동시 요청: {concurrency} | 총 요청: {n}")
    print(f"{'='*50}")

    results = []
    prompt_cycle = (prompts * (n // len(prompts) + 1))[:n]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_single, client, p) for p in prompt_cycle]
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            results.append(r)
            print(f"  [{i+1:2d}/{n}] TTFT: {r['ttft_ms']:6.1f}ms | "
                  f"TPS: {r['tps']:6.1f} tok/s | "
                  f"총: {r['total_s']:4.2f}s")

    ttfts = [r["ttft_ms"] for r in results]
    tpss  = [r["tps"] for r in results]

    print(f"\n{'='*50}")
    print(f"  결과 요약")
    print(f"{'='*50}")
    print(f"  TTFT   avg: {statistics.mean(ttfts):7.1f}ms  "
          f"p50: {statistics.median(ttfts):7.1f}ms  "
          f"p95: {sorted(ttfts)[int(len(ttfts)*0.95)]:7.1f}ms")
    print(f"  TPS    avg: {statistics.mean(tpss):7.1f}     "
          f"p50: {statistics.median(tpss):7.1f}     "
          f"max: {max(tpss):7.1f}")
    print(f"  처리량: {sum(r['tokens'] for r in results) / sum(r['total_s'] for r in results):.1f} tok/s (전체)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    benchmark(args.base_url, args.concurrency, args.n)
