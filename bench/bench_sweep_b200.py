#!/usr/bin/env python3
"""
bench_sweep_b200.py — B200 × GLM-5.2 FP8 벤치마크
bench_sweep_h200.py와 동일한 측정 방식, B200/GLM-5.2 기본값으로 변경.

사용 예:
  # 단일 concurrency 실행
  python3 bench_sweep_b200.py --out /workspace/results/b200_c64 --conc 64

  # sweep (여러 concurrency 순차 실행)
  python3 bench_sweep_b200.py --out /workspace/results/b200_sweep --sweep 1,4,8,16,32,64,128,256
"""
import argparse
import contextlib
import glob
import html
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def request_headers(api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    return headers


def get_served_model(base, api_key):
    """Return the first model id advertised by the OpenAI-compatible endpoint."""
    req = urllib.request.Request(
        base + "/v1/models",
        headers=request_headers(api_key))
    with urllib.request.urlopen(req, timeout=30) as r:
        obj = json.load(r)
    data = obj.get("data") or []
    if not data:
        raise SystemExit("no model returned by %s/v1/models" % base)
    return data[0]["id"]


def post_stream(url, payload, timeout, api_key=None):
    """One streaming chat completion; returns the text and the server's usage."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers=request_headers(api_key))
    t0 = time.perf_counter()
    first = None
    chunks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                _d = ch.get("delta", {})
                piece = _d.get("content") or _d.get("reasoning")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(piece)
    return {"wall": time.perf_counter() - t0,
            "gen_s": (time.perf_counter() - first) if first else None,
            "text": "".join(chunks), "usage": usage}


@contextlib.contextmanager
def progress(total):
    try:
        from tqdm import tqdm
    except ImportError:
        state = {"n": 0}
        t0 = time.perf_counter()
        def tick():
            state["n"] += 1
            el = time.perf_counter() - t0
            eta = el / state["n"] * (total - state["n"])
            sys.stderr.write("\r  %d/%d  %3.0f%%  elapsed %.0fs  eta %.0fs   "
                             % (state["n"], total, 100 * state["n"] / total, el, eta))
            sys.stderr.flush()
        yield tick
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        bar = tqdm(total=total, unit="req", dynamic_ncols=True)
        try:
            yield lambda: bar.update(1)
        finally:
            bar.close()


COLS = ["ISL", "OSL", "GPU", "Precision", "TP", "p", "d", "Concurrency",
        "TPOT(ms)", "Interactivity (Token/sec/user)",
        "Input Token Throughput per GPU (Token/sec/gpu)",
        "Output Token Throughput per GPU (Token/sec/gpu)",
        "Total Token Throughput per GPU (Token/sec/gpu)",
        "Input Token Throughput per server (Token/sec/server)",
        "Output Token Throughput per server (Token/sec/server)"]

WAVES = 4
MIN_REQUESTS = 0


def requests_for(conc, cap):
    return min(cap, max(MIN_REQUESTS, conc * WAVES))


def jobs_for(sheets, requests):
    n = len(sheets)
    if requests >= n:
        return [sheets[i % n] for i in range(requests)]
    return [sheets[(i * n) // requests] for i in range(requests)]


def per_unit(m, gpus, servers):
    g, s = max(1, gpus or 0), max(1, servers or 0)
    return {
        "gpus": gpus, "servers": servers,
        "input_throughput_per_gpu": m["input_throughput"] / g,
        "output_throughput_per_gpu": m["output_throughput"] / g,
        "total_throughput_per_gpu": m["total_throughput"] / g,
        "input_throughput_per_server": m["input_throughput"] / s,
        "output_throughput_per_server": m["output_throughput"] / s,
    }


def md_table(runs):
    def row(m):
        u = m if "total_throughput_per_gpu" in m else {**m, **per_unit(m, m.get("gpus", 0), m.get("servers", 0))}
        return ["%.0f" % m.get("mean_isl", 0), "%.0f" % m.get("mean_osl", 0),
                "%s (%s)" % (m.get("device", "?"), m.get("gpus", "?")),
                str(m.get("precision", "")),
                str(m.get("tp", "") or ""), str(m.get("p", "") or ""),
                str(m.get("d", "") or ""), str(m.get("concurrency", "")),
                "%.2f" % m.get("median_tpot_ms", 0.0),
                "%.2f" % m.get("interactivity_median", 0.0),
                "%.0f" % u["input_throughput_per_gpu"],
                "%.0f" % u["output_throughput_per_gpu"],
                "%.0f" % u["total_throughput_per_gpu"],
                "%.0f" % u["input_throughput_per_server"],
                "%.0f" % u["output_throughput_per_server"]]
    out = ["|" + "|".join(COLS) + "|", "|" + "|".join(["---"] * len(COLS)) + "|"]
    out += ["|" + "|".join(row(m)) + "|" for m in runs]
    return "\n".join(out)


def report(m):
    w = 41
    def row(label, val):
        print("%-*s%s" % (w, label + ":", val))
    print("=" * 15 + " Serving Benchmark Result " + "=" * 15)
    row("Successful requests", m["completed"])
    row("Failed requests", m["failed"])
    row("Total input tokens (tokens)", m["total_input_tokens"])
    row("Total generated tokens (tokens)", m["total_output_tokens"])
    row("Mean ISL (tokens)", "%.1f" % m["mean_isl"])
    row("Mean OSL (tokens)", "%.1f" % m["mean_osl"])
    row("Request throughput (req/s)", "%.2f" % m["request_throughput"])
    row("Input token throughput (tok/s)", "%.2f" % m["input_throughput"])
    row("Output token throughput (tok/s)", "%.2f" % m["output_throughput"])
    row("Total Token throughput (tok/s)", "%.2f" % m["total_throughput"])
    row("Median TPOT (ms)", "%.2f" % m.get("median_tpot_ms", 0.0))
    row("Interactivity (tok/s/user)", "%.2f" % m.get("interactivity_median", 0.0))
    if m.get("gpus"):
        row("Total token throughput per GPU (tok/s)",
            "%.2f" % m.get("total_throughput_per_gpu", 0.0))
    row("Sampling", "temperature %s, top_p %s" % (
        "server default" if m.get("temperature") is None else "%g" % m["temperature"],
        "server default" if m.get("top_p") is None else "%g" % m["top_p"]))
    print("=" * 56)


VIEWER = """<!doctype html><meta charset=utf-8><title>%(title)s</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:16px 20px;background:#161a22;border-bottom:1px solid #262b36}
 h1{margin:0 0 8px;font-size:17px}
 .sum{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:13px;color:#9aa4b2}
 .sum b{color:#e6e6e6;font-weight:600}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;padding:16px 20px}
 .card{background:#161a22;border:1px solid #262b36;border-radius:6px;padding:10px;cursor:pointer}
 .card:hover{border-color:#4c8dff}
 .card .id{font-weight:600;margin-bottom:4px}
 .card .kv{font-size:12px;color:#9aa4b2}
 .err{border-color:#7a2b2b}
 dialog{width:min(1000px,92vw);max-height:88vh;background:#161a22;color:#e6e6e6;
        border:1px solid #262b36;border-radius:8px;padding:0}
 dialog::backdrop{background:rgba(0,0,0,.6)}
 .dh{padding:14px 18px;border-bottom:1px solid #262b36;display:flex;justify-content:space-between;align-items:center}
 .db{padding:14px 18px;overflow:auto;max-height:70vh}
 h3{margin:16px 0 6px;font-size:13px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.5px}
 pre{white-space:pre-wrap;word-break:break-word;background:#0f1115;border:1px solid #262b36;
     border-radius:6px;padding:10px;font:12px/1.5 ui-monospace,monospace;margin:0}
 button{background:#262b36;color:#e6e6e6;border:0;border-radius:5px;padding:6px 12px;cursor:pointer}
</style>
<header>
 <h1>%(title)s</h1>
 <div class=sum id=sum></div>
 <details open style="margin-top:12px">
  <summary style="cursor:pointer;color:#9aa4b2;font-size:13px">Metrics table (paste into the sheet)
   <button style="margin-left:10px" onclick="cp(event)">Copy</button></summary>
  <pre id=md style="margin-top:8px;overflow-x:auto;white-space:pre">%(table)s</pre>
 </details>
</header>
<script>
function cp(e){e.preventDefault();const t=document.getElementById('md').textContent;
 const done=()=>{e.target.textContent='Copied';setTimeout(()=>e.target.textContent='Copy',1500)};
 if(navigator.clipboard){navigator.clipboard.writeText(t).then(done,sel)}else{sel()}
 function sel(){const r=document.createRange();r.selectNodeContents(document.getElementById('md'));
  const s=getSelection();s.removeAllRanges();s.addRange(r);
  try{document.execCommand('copy');done()}catch(_){e.target.textContent='Select + Ctrl-C'}}}
</script>
<div id=grid></div>
<dialog id=dlg><div class=dh><b id=dt></b><button onclick="dlg.close()">Close</button></div>
<div class=db id=dbody></div></dialog>
<script>
const D = %(data)s;
document.getElementById('sum').innerHTML = D.summary_html;
const grid = document.getElementById('grid');
D.requests.forEach((r,i)=>{
  const c = document.createElement('div');
  c.className = 'card' + (r.error ? ' err' : '');
  c.innerHTML = '<div class=id>#'+String(i).padStart(4,'0')+'</div>'
    + '<div class=kv>ISL '+r.isl+' tokens</div>'
    + '<div class=kv>OSL '+r.osl+' tokens</div>';
  c.onclick=()=>{
    document.getElementById('dt').textContent = '#'+String(i).padStart(4,'0')+'  '+r.sheet;
    document.getElementById('dbody').innerHTML =
      '<h3>tokens</h3><pre>'+r.meta+'</pre>'
      + '<h3>input'+(r.input_truncated?' (middle elided -- full text in result.json)':'')+'</h3><pre>'+r.input+'</pre>'
      + '<h3>output</h3><pre>'+(r.output||'(empty)')+'</pre>';
    dlg.showModal();
  };
  grid.appendChild(c);
});
</script>
"""


def build_viewer(path, title, metrics, rows, full_input=False, table=None):
    head, tail = 3000, 1200
    cards = []
    for r in rows:
        text = r["input"]
        clipped = (not full_input) and len(text) > head + tail
        shown = (text[:head] + "\n\n...(%d chars elided)...\n\n" % (len(text) - head - tail)
                 + text[-tail:]) if clipped else text
        cards.append({
            "sheet": os.path.basename(r["sheet"]),
            "isl": r["isl"], "osl": r["osl"],
            "error": r["error"],
            "input_truncated": clipped,
            "input": html.escape(shown),
            "output": html.escape(r["output"] or (r["error"] or "")),
            "meta": html.escape("ISL %d tokens   OSL %d tokens" % (r["isl"], r["osl"])),
        })
    summary = (
        "<span>Completed <b>%d/%d</b></span>"
        "<span>Input <b>%.0f tok/s</b></span><span>Output <b>%.0f tok/s</b></span>"
        "<span>Total <b>%.0f tok/s</b></span><span>Mean ISL <b>%.0f tokens</b></span>"
        "<span>Mean OSL <b>%.0f tokens</b></span>"
        "<span>Interactivity <b>%.2f tok/s/user</b></span>" % (
            metrics["completed"], metrics["completed"] + metrics["failed"],
            metrics["input_throughput"],
            metrics["output_throughput"], metrics["total_throughput"],
            metrics["mean_isl"], metrics["mean_osl"],
            metrics.get("interactivity_median", 0.0)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(VIEWER % {"title": html.escape(title),
                          "table": html.escape(table or md_table([metrics])),
                          "data": json.dumps({"summary_html": summary, "requests": cards})})


def title_for(m):
    return "%s - conc %s, thinking %s" % (
        m.get("label") or m.get("endpoint", ""), m.get("concurrency", "?"),
        "on" if m.get("think") else "off")


def run_one(args, sheets, conc, out_dir, all_metrics_ref=None, table_ref=None):
    """Run one concurrency level and save results."""
    n_requests = requests_for(conc, args.requests)
    jobs = jobs_for(sheets, n_requests)
    base = "http://%s:%d" % (args.host, args.port)

    tpl = {"enable_thinking": bool(args.think)}
    effort = args.reasoning_effort if "gpt-oss" in args.model.lower() else None

    def one(sheet):
        prompt = open(sheet, encoding="utf-8").read()
        payload = {"model": args.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": args.max_tokens,
                   "stream": True, "stream_options": {"include_usage": True}}
        if args.ignore_eos:
            payload["ignore_eos"] = True
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if tpl:
            payload["chat_template_kwargs"] = tpl
        if effort:
            payload["reasoning_effort"] = effort
        rec = {"sheet": sheet, "input": prompt, "output": "", "error": None,
               "isl": 0, "osl": 0, "_gen_s": None}
        try:
            r = post_stream(base + "/v1/chat/completions", payload, args.timeout, args.api_key)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            rec["error"] = repr(e)
            return rec
        rec["output"] = r["text"]
        rec["isl"] = r["usage"].get("prompt_tokens", 0)
        rec["osl"] = r["usage"].get("completion_tokens", 0)
        rec["_gen_s"] = r["gen_s"]
        rec["_wall_s"] = r["wall"]
        return rec

    print("\n[conc=%d]  %d requests" % (conc, n_requests))
    results = []
    t_start = time.perf_counter()

    with progress(n_requests) as tick:
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = {pool.submit(one, j): j for j in jobs}
            for f in as_completed(futs):
                results.append(f.result())
                tick()

    t_wall = time.perf_counter() - t_start
    ok = [r for r in results if not r["error"]]
    fail = len(results) - len(ok)

    isl_list = [r["isl"] for r in ok]
    osl_list = [r["osl"] for r in ok]
    gen_s_list = [r["_gen_s"] for r in ok if r.get("_gen_s") is not None]
    wall_list = [r.get("_wall_s", 0) for r in ok]

    tpot_ms_list = []
    for r in ok:
        if r.get("_gen_s") and r["osl"] > 1:
            tpot_ms_list.append(r["_gen_s"] / (r["osl"] - 1) * 1000)

    total_in = sum(isl_list)
    total_out = sum(osl_list)

    m = {
        "endpoint": args.label or ("%s:%d" % (args.host, args.port)),
        "label": args.label,
        "model": args.model,
        "concurrency": conc,
        "completed": len(ok),
        "failed": fail,
        "wall_s": t_wall,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "mean_isl": statistics.mean(isl_list) if isl_list else 0,
        "mean_osl": statistics.mean(osl_list) if osl_list else 0,
        "request_throughput": len(ok) / t_wall,
        "input_throughput": total_in / t_wall,
        "output_throughput": total_out / t_wall,
        "total_throughput": (total_in + total_out) / t_wall,
        "median_tpot_ms": statistics.median(tpot_ms_list) if tpot_ms_list else 0.0,
        "interactivity_median": (1000.0 / statistics.median(tpot_ms_list)
                                  if tpot_ms_list else 0.0),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "think": args.think,
        "device": args.device,
        "precision": args.precision,
        "tp": args.tp,
        "p": args.p or None,
        "d": args.d or None,
        "gpus": args.gpus,
        "servers": args.servers,
        **per_unit({"input_throughput": total_in / t_wall,
                    "output_throughput": total_out / t_wall,
                    "total_throughput": (total_in + total_out) / t_wall},
                   args.gpus, args.servers),
    }

    os.makedirs(out_dir, exist_ok=True)

    # result.json
    rows_out = [{"sheet": r["sheet"], "input": r["input"], "output": r["output"],
                 "error": r["error"], "isl": r["isl"], "osl": r["osl"]} for r in results]
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "requests": rows_out}, f, ensure_ascii=False, indent=1)

    # viewer.html
    build_viewer(os.path.join(out_dir, "viewer.html"), title_for(m), m, rows_out, args.full_input)

    # console
    with open(os.path.join(out_dir, "client.txt"), "w") as f:
        with contextlib.redirect_stdout(f):
            report(m)
    report(m)

    if all_metrics_ref is not None:
        all_metrics_ref.append(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # B200 / GLM-5.2 기본값 (H200과 다른 부분)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="", help="empty = use first id from /v1/models")
    ap.add_argument("--gen-dir", default="/workspace/gen8k",
                    help="디렉토리 내 *target*.txt 파일을 프롬프트로 사용")
    ap.add_argument("--conc", type=int, default=256)
    ap.add_argument("--requests", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ignore-eos", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="B200 GLM-5.2 FP8")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--full-input", action="store_true")
    ap.add_argument("--sweep", default="1,4,8,16,32,64,128,256",
                    help="쉼표로 구분된 concurrency 목록. H200 기본: 12,64,128,200,256")
    # B200 메타데이터 (H200 -> B200, tp=8)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--servers", type=int, default=1)
    ap.add_argument("--device", default="B200")          # ← H200 -> B200
    ap.add_argument("--precision", default="FP8")
    ap.add_argument("--tp", type=int, default=8)         # ← TP=8
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--d", type=int, default=0)
    args = ap.parse_args()

    base = "http://%s:%d" % (args.host, args.port)
    if not args.model:
        args.model = get_served_model(base, args.api_key)
        print("Using model from /v1/models: %s" % args.model)

    # gen-dir 없으면 샘플 프롬프트 자동 생성
    if not os.path.isdir(args.gen_dir):
        print("⚠️  --gen-dir '%s' 없음 → 샘플 프롬프트 자동 생성" % args.gen_dir)
        os.makedirs(args.gen_dir, exist_ok=True)
        sample_prompts = [
            "NVIDIA B200 Blackwell 아키텍처가 H100 대비 어떤 점에서 뛰어난지 설명해줘.",
            "Multi-head Latent Attention(MLA)의 KV 캐시 압축 원리를 설명해줘.",
            "vLLM에서 Speculative Decoding이 어떻게 동작하는지 알려줘.",
            "FP8 양자화가 모델 추론 속도에 미치는 영향을 설명해줘.",
            "FlashAttention 3의 핵심 개선점이 뭔지 설명해줘.",
            "GLM-5.2의 MLA 아키텍처와 DeepSeek-V3의 차이점을 설명해줘.",
            "Tensor Parallel 8의 통신 오버헤드를 줄이는 방법을 설명해줘.",
            "B200에서 FP8 GEMM이 BF16 대비 얼마나 빠른지 설명해줘.",
            "vLLM의 Prefix Caching이 TTFT에 미치는 영향을 설명해줘.",
            "MoE (Mixture of Experts) 모델의 토큰 라우팅 메커니즘을 설명해줘.",
            "KV 캐시 오프로딩(offloading) 기법이 무엇인지 설명해줘.",
            "Chunked Prefill이 배치 처리 효율에 어떤 영향을 주는지 설명해줘.",
            "PagedAttention의 메모리 관리 방식을 설명해줘.",
            "B200의 NVLink 5.0 대역폭이 LLM 추론에 미치는 영향을 설명해줘.",
            "GLM-5.2의 Sparse Attention 메커니즘을 설명해줘.",
            "CUDA Graph가 LLM 추론 지연 시간을 줄이는 방법을 설명해줘.",
            "Triton 커널과 CUDA 커널의 성능 차이를 설명해줘.",
            "DeepSeek-V3 MTP(Multi-Token Prediction)의 accept rate를 높이는 방법을 설명해줘.",
            "B200 HBM3e 8TB/s 대역폭이 Decode 단계에서 중요한 이유를 설명해줘.",
            "vLLM에서 FlashInfer 백엔드가 기본 어텐션 구현보다 빠른 이유를 설명해줘.",
        ]
        for i, p in enumerate(sample_prompts):
            fname = os.path.join(args.gen_dir, "sample_target_%03d.txt" % i)
            with open(fname, "w", encoding="utf-8") as f:
                f.write(p)
        print("  샘플 프롬프트 %d개 생성됨: %s" % (len(sample_prompts), args.gen_dir))

    sheets = sorted(glob.glob(os.path.join(args.gen_dir, "*target*.txt")))
    if not sheets:
        raise SystemExit("no sheets in %s" % args.gen_dir)
    if args.requests > len(sheets):
        print("WARNING: %d sheets < %d requests -- reused sheets hit the prefix cache"
              % (len(sheets), args.requests))

    concs = [int(x) for x in args.sweep.split(",")]
    all_metrics = []

    for conc in concs:
        sub = os.path.join(args.out, "c%d" % conc) if len(concs) > 1 else args.out
        run_one(args, sheets, conc, sub, all_metrics)

    if len(concs) > 1:
        # 전체 sweep 요약 테이블
        print("\n" + "=" * 56)
        print("SWEEP SUMMARY (B200 × GLM-5.2 FP8, TP=%d)" % args.tp)
        print("=" * 56)
        print(md_table(all_metrics))
        summary_path = os.path.join(args.out, "sweep_summary.md")
        with open(summary_path, "w") as f:
            f.write("# B200 × GLM-5.2 FP8 벤치마크 결과\n\n")
            f.write(md_table(all_metrics))
        print("\n결과 저장: %s" % summary_path)


if __name__ == "__main__":
    main()
