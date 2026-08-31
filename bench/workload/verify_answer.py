#verify_answer.py

"""Recompute SECTION 3/4 ground truth from a target file and diff against its answer file.

Usage: python3 verify_answer.py target.txt answer.txt
"""
import re
import sys


def parse_target(path):
    text = open(path, encoding="utf-8").read()
    gpu = dict(
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r"^(DC\d+)\s*:\s*(\d+)\s*$", text, re.M)
    )
    keys = dict(
        (m.group(1), m.group(2).strip())
        for m in re.finditer(r"^(KEY-\d+)\s*=\s*(\S+)\s*$", text, re.M)
    )
    uuids = re.findall(r"^UUID=", text, re.M)
    # SECTION 2 paragraphs are 3 lines each, headed by "Project-<Name>-<NNN>-<NNNN>는 ..."
    sec2 = text.split("[SECTION 2]")[1].split("[SECTION 3]")[0]
    paras = re.findall(r"^Project-\w+-\d{3}-\d+는 ", sec2, re.M)
    return gpu, keys, len(uuids), len(paras), text


def parse_answer(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def rank(gpu, top):
    # ties: answer files list DC ids; sort by value then id to get a deterministic order
    items = sorted(gpu.items(), key=lambda kv: (-kv[1], kv[0]) if top else (kv[1], kv[0]))
    return ", ".join(f"{k}:{v}" for k, v in items[:20])


def main(target, answer):
    gpu, keys, n_uuid, n_para, text = parse_target(target)
    ans = parse_answer(answer)
    vals = list(gpu.values())
    got = {
        "SECTION1_COUNT": str(n_uuid),
        "SECTION2_COUNT": str(n_para),
        "BYTES": str(len(text.encode("utf-8"))),
        "LOW_COUNT": str(sum(1 for v in vals if v <= 199)),
        "MEDIUM_COUNT": str(sum(1 for v in vals if 200 <= v <= 399)),
        "HIGH_COUNT": str(sum(1 for v in vals if v >= 400)),
        "AVERAGE": f"{sum(vals) / len(vals):.2f}",
        "TOP20": rank(gpu, True),
        "BOTTOM20": rank(gpu, False),
        "KEY-073": keys.get("KEY-073", "<missing>"),
    }
    print(f"== {target} vs {answer} ==")
    print(f"GPU entries={len(gpu)}  KEY entries={len(keys)}  UUID lines={n_uuid}  SEC2 lines={n_para}")
    ok = True
    for k, v in got.items():
        exp = ans.get(k)
        match = exp == v
        ok &= match
        mark = "OK  " if match else "MISMATCH"
        if match:
            print(f"  {mark} {k} = {v[:70]}")
        else:
            print(f"  {mark} {k}\n      answer : {exp}\n      computed: {v}")
    print("VERDICT:", "answer file fully reproducible from target" if ok else "mismatch present")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
 