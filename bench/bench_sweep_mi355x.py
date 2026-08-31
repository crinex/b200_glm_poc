#bench_sweep.py

"""
  - thinking is OFF by default.
  - no grading.
  - every request's prompt and completion are dumped, with its own ISL/OSL.

Reported metrics are wall clock, total and mean ISL/OSL, and the throughputs
derived from them. Per-request, only ISL and OSL vary.
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


def post_stream(url, payload, timeout):
    """One streaming chat completion; returns the text and the server's usage."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
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
    """Yield a tick() that advances a progress bar.

    tqdm when it is installed, otherwise a one-line fallback -- a benchmark
    should not fail to run because a box lacks a progress bar.
    """
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


WAVES = 4          # was 8          # request waves per concurrency level
MIN_REQUESTS = 0   # was 128: the floor made conc 4 send 32 waves, not 4


def requests_for(conc, cap):
    """How many requests one sweep level should send.

    Sending the full set at every level is dominated by the low ones: 1024
    requests at concurrency 4 is 256 waves and takes over an hour for a single
    row, while adding nothing -- the rate is steady long before that. Eight
    waves is enough to read it, and the high levels still get the full set.
    """
    return min(cap, max(MIN_REQUESTS, conc * WAVES))


def jobs_for(sheets, requests):
    """Pick `requests` prompts, spread across the whole set.

    Taking the first N biases a short run toward whatever sorts first, so rows
    measured at different concurrencies would no longer share an input
    distribution and their ISL columns would not be comparable. Asking for more
    than exist cycles, which reuses prompts and hits the prefix cache.
    """
    n = len(sheets)
    if requests >= n:
        return [sheets[i % n] for i in range(requests)]
    return [sheets[(i * n) // requests] for i in range(requests)]


def per_unit(m, gpus, servers):
    """Throughput normalised per GPU and per server.

    A count of 0 means "not given" and divides by 1, so a run without the flags
    still produces a row instead of crashing.
    """
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
    """The SKPI-532 table, ready to paste into Jira.

    p and d are blank for aggregated runs -- the sheet leaves them empty there
    rather than writing 0, and a 0 would read as "zero prefill servers".
    """
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
    row("Interactivity (tok/s/user)",
        "%.2f" % m.get("interactivity_median", 0.0))
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
    """Self-contained HTML.

    Prompts are clipped by default. One sheet is ~35KB, so 1024 of them inline
    make a ~40MB page: it opens, but it loads slowly and is awkward to move
    around. full_input turns the clipping off. result.json keeps the full text
    either way, so nothing is lost by clipping.
    """
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
        "<span>Completed <b>%d/%d</b>"</span>
        "<span>Input <b>%.0f tok/s</b><span>Output <b>%.0f tok/s</b></span>"</span>
        "<span>Total <b>%.0f tok/s</b><span>Mean ISL <b>%.0f tokens</b></span>"</span>
        "<span>Mean OSL <b>%.0f tokens</b>"</span>
        "<span>Interactivity <b>%.2f tok/s/user</b>"</span> % (
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


def rebuild(out_dir, full_input=False):
    """Re-emit result.json and viewer.html from an existing run.

    Used to bring older runs in line with the current output: they were written
    when latency series were still collected, and those fields are dropped here
    rather than left in a file that goes out to a customer.
    """
    path = os.path.join(out_dir, "result.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    drop_m = ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
              "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "interactivity",
              "started_at_utc", "ended_at_utc"]
    drop_r = ["ttft_ms", "tpot_ms", "wall_s"]
    m = {k: v for k, v in data["metrics"].items() if k not in drop_m}
    rows = [{k: v for k, v in r.items() if k not in drop_r} for r in data["requests"]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "requests": rows}, f, ensure_ascii=False, indent=1)
    build_viewer(os.path.join(out_dir, "viewer.html"), title_for(m), m, rows, full_input)

    # client.txt is the console capture from the original run, so it still holds
    # whatever that run printed. Rewrite it from the cleaned metrics rather than
    # leaving a stale transcript beside a cleaned result.json.
    log = os.path.join(out_dir, "client.txt")
    if os.path.exists(log):
        with open(log, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            report(m)
    report(m)
    print("rebuilt: %s/{result.json,viewer.html,client.txt}" % out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="/models/GLM-5.2-MXFP4")
    ap.add_argument("--gen-dir", default="gen")
    ap.add_argument("--conc", type=int, default=256)
    ap.add_argument("--requests", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ignore-eos", type=int, default=1,
                    help="1 = generate exactly --max-tokens; fixes OSL so it stops confounding throughput comparisons")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omitted = not sent (server default, 1.0 for this model)")
    ap.add_argument("--top-p", type=float, default=None,
                    help="omitted = not sent (server default, 0.95 for this model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--reasoning-effort", default="low",
                    help="gpt-oss only: low|medium|high (other models ignore it)")
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--full-input", action="store_true",
                    help="embed full prompts in the viewer (~40MB for 1024 requests)")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-emit result.json and viewer.html in --out; sends no requests")
    ap.add_argument("--sweep", default="",
                    help="comma-separated concurrencies, e.g. 4,8,16,32,64,128,256. "
                         "Each lands in <out>/c<N>/; the table spans all of them")
    # Table metadata. These do not change what is sent -- they label the row and
    # set what the throughput is divided by.
    ap.add_argument("--gpus", type=int, default=8, help="GPUs behind the endpoint")
    ap.add_argument("--servers", type=int, default=1, help="server instances (node count)")
    ap.add_argument("--device", default="MI355x")
    ap.add_argument("--precision", default="MXFP4")
    ap.add_argument("--tp", type=int, default=0)
    ap.add_argument("--p", type=int, default=0, help="prefill instances (PD only)")
    ap.add_argument("--d", type=int, default=0, help="decode instances (PD only)")
    args = ap.parse_args()

    if args.rebuild:
        rebuild(args.out, args.full_input)
        return

    base = "http://%s:%d" % (args.host, args.port)
    sheets = sorted(glob.glob(os.path.join(args.gen_dir, "*target*.txt")))
    if not sheets:
        raise SystemExit("no sheets in %s" % args.gen_dir)
    if args.requests > len(sheets):
        print("WARNING: %d sheets < %d requests -- reused sheets hit the prefix cache"
              % (len(sheets), args.requests))

    tpl = None if args.think else {"enable_thinking": False}
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
            r = post_stream(base + "/v1/chat/completions", payload, args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            rec["error"] = repr(e)
            return rec
        rec["output"] = r["text"]
        rec["isl"] = r["usage"].get("prompt_tokens", 0)
        rec["osl"] = r["usage"].get("completion_tokens", 0)
        rec["_gen_s"] = r["gen_s"]
        if not r["text"]:
            rec["error"] = "empty response"
        return rec

    fmt = lambda v: "server default" if v is None else "%g" % v

    concs = [int(c) for c in args.sweep.split(",") if c.strip()] or [args.conc]
    runs = []
    for conc in concs:
        out_dir = args.out if len(concs) == 1 else os.path.join(args.out, "c%d" % conc)
        # A single run sends exactly what was asked for; a sweep scales the low
        # levels down so one row does not eat the whole session.
        want = requests_for(conc, args.requests) if len(concs) > 1 else args.requests
        jobs = jobs_for(sheets, want)
        if len(concs) > 1:
            print("\n" + "#" * 18 + "  concurrency %d, %d requests  " % (conc, want)
                  + "#" * 18)
        runs.append(run_once(args, base, jobs, one, conc, out_dir, fmt))

    if len(concs) > 1:
        table = md_table(runs)
        with open(os.path.join(args.out, "table.md"), "w", encoding="utf-8") as f:
            f.write(table + "\n")
        build_viewer(os.path.join(args.out, "index.html"),
                     "%s - sweep %s" % (args.label or base, args.sweep),
                     runs[-1], [], table=table)
        print("\n" + table)
        print("\nSaved: %s/{table.md,index.html} + c<N>/ per concurrency" % args.out)


def run_once(args, base, jobs, one, conc, out_dir, fmt):
    """One concurrency level: send everything, write result.json + viewer.html.

    Flushing per level, not once per sweep: without it the second level starts
    with the first level's prompts already in the prefix cache and reads faster
    than it should.
    """
    if not args.no_flush:
        # sglang calls it /flush_cache, vLLM calls it /reset_prefix_cache. Try both before
        # giving up, so the same bench can measure either engine on the same footing --
        # a run that silently skipped the flush would read faster than it should.
        errs = []
        for path in ("/flush_cache", "/reset_prefix_cache"):
            try:
                urllib.request.urlopen(
                    urllib.request.Request(base + path, data=b""), timeout=30).read()
                print("Cache flushed via %s" % path)
                break
            except (urllib.error.URLError, OSError) as e:
                errs.append("%s: %r" % (path, e))
        else:
            raise SystemExit("Cache flush failed -- check the server: %s" % "; ".join(errs))

    print("Starting: %d requests, concurrency %d, thinking %s, max_tokens %d, "
          "temperature %s, top_p %s"
          % (len(jobs), conc, "on" if args.think else "off", args.max_tokens,
             fmt(args.temperature), fmt(args.top_p)))

    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    t0 = time.perf_counter()
    rows = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(one, s): i for i, s in enumerate(jobs)}
        with progress(len(jobs)) as tick:
            for f in as_completed(futs):
                rows[futs[f]] = f.result()
                tick()
    t_wc = time.perf_counter() - t0
    ended_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    ok = [r for r in rows if not r["error"]]
    tin = sum(r["isl"] for r in ok)
    tout = sum(r["osl"] for r in ok)
    n = max(1, len(ok))
    tpots = [r["_gen_s"] / (r["osl"] - 1) * 1000
             for r in ok if r["_gen_s"] and r["osl"] > 1]
    mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0
    # InferenceX 는 Interactivity 를 median / P90 / P99 TPOT 로 비교한다 (mean 은 안 쓴다):
    # utils/compare_results.py METRIC_DEFS. 우리는 mean 만 내보내고 있었다 -- 더 불리한 통계다.
    # mean_tpot 와 interactivity 의 정의는 그대로 두어 지난 측정들과의 비교를 깨지 않는다.
    def _pct(vals, q):
        if not vals:
            return 0.0
        xs = sorted(vals)
        i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        return xs[i]
    median_tpot = statistics.median(tpots) if tpots else 0.0
    p90_tpot = _pct(tpots, 0.90)
    p99_tpot = _pct(tpots, 0.99)
    # 요청별 TPOT 을 남긴다. 이걸 버렸기 때문에 지난 런들의 median 을 소급할 수 없었다.
    for r in rows:
        g = r.pop("_gen_s", None)
        r["tpot_ms"] = (g / (r["osl"] - 1) * 1000) if (g and r["osl"] > 1) else None
    m = {
        "label": args.label, "endpoint": base, "model": args.model,
        "concurrency": conc, "requested": len(jobs),
        "think": args.think, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "completed": len(ok), "failed": len(rows) - len(ok),
        "duration_s": t_wc,
        "total_input_tokens": tin, "total_output_tokens": tout,
        "mean_isl": tin / n, "mean_osl": tout / n,
        "request_throughput": len(ok) / t_wc,
        "input_throughput": tin / t_wc,
        "output_throughput": tout / t_wc,
        "total_throughput": (tin + tout) / t_wc,
        "mean_tpot_ms": mean_tpot,
        "median_tpot_ms": median_tpot,
        "p90_tpot_ms": p90_tpot,
        "p99_tpot_ms": p99_tpot,
        # 1000 / mean TPOT. 지난 측정들과의 연속성 때문에 정의를 바꾸지 않는다.
        "interactivity": (1000.0 / mean_tpot) if mean_tpot else 0.0,
        # InferenceX 가 실제로 비교하는 값들.
        "interactivity_median": (1000.0 / median_tpot) if median_tpot else 0.0,
        "interactivity_p90": (1000.0 / p90_tpot) if p90_tpot else 0.0,
        "interactivity_p99": (1000.0 / p99_tpot) if p99_tpot else 0.0,
        "device": args.device, "precision": args.precision,
        "tp": args.tp, "p": args.p, "d": args.d,
    }
    m.update(per_unit(m, args.gpus, args.servers))
    report(m)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bench_window.txt"), "w", encoding="utf-8") as f:
        f.write("start_utc  %s\nend_utc    %s\nduration_s %.2f\n"
                % (started_at, ended_at, t_wc))
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "requests": rows}, f, ensure_ascii=False, indent=1)
    build_viewer(os.path.join(out_dir, "viewer.html"), title_for(m), m, rows,
                 args.full_input)

    try:
        code = urllib.request.urlopen(base + "/health", timeout=10).status
        print("\nServer alive: /health = %d" % code)
    except Exception as e:
        print("\nServer alive: /health FAILED %r  <-- server may be down" % e)
    print("Saved: %s/{result.json,viewer.html}" % out_dir)
    return m


if __name__ == "__main__":
    main()
 