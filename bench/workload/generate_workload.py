#generate_workload.py

"""Generate a LU G+ style benchmark prompt and its answer key.

Reproduces the U+ original the way U+ made it: fixed element counts and a byte
budget, no tokenizer in the loop. Their answer key says as much —
"exact BPE data could not be loaded in this offline runtime" — which is why a
file they call 8K actually costs ~14K tokens on GLM. Measure the real cost with
count_tokens.py; this script does not target tokens.

Usage: python3 generate_workload.py --seed 1 --out-dir ./gen
"""
import argparse
import os
import random
import string
import uuid

from verify_answer import rank  # one definition of the tie-break rule

# Vocabulary lifted from the two originals so regenerated noise reads the same.
NAMES = """Aquila Argon Aster Atlas Aurora Beacon Boreal Chroma Cinder Cipher Cosmos Delta
Drift Dynamo Echo Eclipse Ember Fjord Fusion Glacier Halo Helix Ion Isobar Jade Krypton
Kuiper Lagoon Lumen Lyric Mirage Mist Nadir Nebula Nimbus Nova Obsidian Omega Orion
Prairie Prism Pulsar Quantum Quartz Rivet Rubicon Sable Sigma Solar Terra Tundra Umbra
Vector Vertex Vesper Willow Xenon Yonder Zenith Zephyr""".split()
REGIONS = """고원 구릉 궤도 극지 도시권 분지 분화구 빙벽 빙원 사막 석호 심층 안개지대 운하
초소 초원 평원 항로 해안 해저 협곡 화산대""".split()
OBJECTS = """archive cache-slate calc-letter context-shard context-tile fake-index mock-ledger
noise-lattice random-panel routing-card shadow-note signal-bundle signal-jar temp-node
token-box token-frame trace-bucket trace-log vector-bag vector-map verify-table
virtual-queue""".split()
ACTIONS = "검산 검출 격리 동기화 분류 분할 압축 재배열 정렬 중계 탐색 표본화".split()
COLORS = """amber black blue bronze cobalt crimson cyan green indigo ivory lime magenta navy
olive pearl purple red silver teal violet white yellow""".split()

RULE = "------------------------------------------------"
FINAL_TASK = """1. LOW count
2. MEDIUM count
3. HIGH count
4. overall average
5. extract top 20
6. extract bottom 20
7. extract KEY-073 value
8. validate results
9. write 800~1200 token report"""


def _letters(rng, n=6):
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def _digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def _section1(rng, count):
    out = ["[SECTION 1]", "UUID/HASH/NONCE block"]
    for _ in range(count):
        out += [f"UUID={uuid.UUID(int=rng.getrandbits(128), version=4)}",
                f"HASH={_digits(rng, 15)}",
                f"NONCE={_letters(rng)}-{_digits(rng, 6)}"]
    return out


def _section2(rng, count):
    out = ["[SECTION 2]", "garbage context"]
    for i in range(count):
        tag = f"Project-{rng.choice(NAMES)}-{i:03d}-{rng.randint(1000, 9999)}"
        if rng.random() < 0.5:
            last = (f"{rng.choice(COLORS)} route {_digits(rng, 6)} is a test sentence "
                    "for long-context retrieval noise.")
        else:
            last = (f"temporary coordinate {_digits(rng, 5)}-{_letters(rng)}-{_digits(rng, 6)} "
                    "invalidates reusable prefix cache blocks.")
        out += [f"{tag}는 {rng.choice(REGIONS)} 영역의 {rng.choice(OBJECTS)}를 "
                f"{rng.choice(ACTIONS)}하는 허구 시스템이다.",
                f"{tag} marker {_letters(rng)}-{_digits(rng, 6)} is synthetic context "
                "unrelated to any real operation.",
                last, ""]
    return out


def generate(seed, n_uuid=90, n_garbage=65, n_dc=300, n_key=100, dc_range=(100, 700),
             hidden_key_index=73, hidden_key_value="CORE-88132-ZETA-XR9"):
    """Return (prompt_text, answer_text) for the given seed."""
    rng = random.Random(seed)
    lines = _section1(rng, n_uuid) + [RULE] + _section2(rng, n_garbage) + [RULE]

    gpu = {f"DC{i:03d}": rng.randint(*dc_range) for i in range(1, n_dc + 1)}
    lines += ["[SECTION 3]", "GPU usage data"] + [f"{k} : {v}" for k, v in gpu.items()] + [RULE]

    keys = {f"KEY-{i:03d}": "".join(rng.choice(string.ascii_uppercase + string.digits)
                                    for _ in range(16)) for i in range(1, n_key + 1)}
    keys[f"KEY-{hidden_key_index:03d}"] = hidden_key_value
    lines += ["[SECTION 4]", "hidden key information"]
    lines += [f"{k} = {v}" for k, v in keys.items()] + [RULE]

    lines += ["[SECTION 5]", "reasoning rules", "LOW : 0~199", "MEDIUM : 200~399",
              "HIGH : 400+", RULE, "[SECTION 6]", "final task"] + FINAL_TASK.splitlines()
    target = "\n".join(lines)

    vals = list(gpu.values())
    answer = "\n".join([
        "TOKEN_TARGET=7800~8200 GPT tokens",
        "TOKENIZER_TARGET=cl100k_base/o200k_base class",
        "NOTE=Regenerated with the same section structure and byte budget as the U+ original. "
        "TOKEN_TARGET and ESTIMATED_TOKENS_BY_4_BYTES carry the original's 4-bytes-per-token "
        "assumption and are not measured; use count_tokens.py for the real cost.",
        f"SEED={seed}",
        f"BYTES={len(target.encode('utf-8'))}",
        f"ESTIMATED_TOKENS_BY_4_BYTES={len(target.encode('utf-8')) // 4}",
        f"SECTION1_COUNT={n_uuid}",
        f"SECTION2_COUNT={n_garbage}",
        "FILLER_LINES=0",
        f"LOW_COUNT={sum(1 for v in vals if v <= 199)}",
        f"MEDIUM_COUNT={sum(1 for v in vals if 200 <= v <= 399)}",
        f"HIGH_COUNT={sum(1 for v in vals if v >= 400)}",
        f"AVERAGE={sum(vals) / len(vals):.2f}",
        f"TOP20={rank(gpu, True)}",
        f"BOTTOM20={rank(gpu, False)}",
        f"KEY-{hidden_key_index:03d}={hidden_key_value}",
    ])
    return target, answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="", help="filename prefix, e.g. run3_")
    # gen8k/ was built with 37/27, not the generate() defaults (90/65): the counts
    # were cut to land the prompt on the 8k-token budget. 90/65 costs ~15k tokens,
    # which a 9500-context server rejects with HTTP 400. The answer file records
    # SECTION1_COUNT / SECTION2_COUNT, so an existing set is self-describing.
    ap.add_argument("--n-uuid", type=int, default=37, help="SECTION 1 element count")
    ap.add_argument("--n-garbage", type=int, default=27, help="SECTION 2 element count")
    args = ap.parse_args()

    target, answer = generate(args.seed, n_uuid=args.n_uuid, n_garbage=args.n_garbage)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, text in (("target.txt", target), ("answer.txt", answer)):
        path = os.path.join(args.out_dir, args.prefix + name)
        open(path, "w", encoding="utf-8").write(text)
        print(f"wrote {path}  ({len(text.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
 