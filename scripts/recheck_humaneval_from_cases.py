import argparse
import json
import os
import subprocess
import sys
import tempfile
import re
from typing import Dict, List


_FIM_MARKERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")


def sanitize_completion(completion: str) -> str:
    if not completion:
        return ""
    text = completion.replace("\r\n", "\n")
    for m in _FIM_MARKERS:
        i = text.find(m)
        if i != -1:
            text = text[:i]
    text = text.replace("\uFFFD", "")
    text = "".join(ch for ch in text if ch in ("\n", "\t") or 32 <= ord(ch) <= 126)

    out_lines = []
    for line in text.split("\n"):
        if re.match(r"^(def|class|from|import)\s+", line):
            break
        if line.startswith("<|") or line.startswith("#!"):
            break
        if line and not line.startswith((" ", "\t")) and re.match(r"^[A-Za-z_]\w*(\.\w+)+\s*[\+\-*/]?=", line):
            break
        out_lines.append(line)

    text = "\n".join(out_lines).rstrip()

    lines = text.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        min_indent = min(len(ln) - len(ln.lstrip(" \t")) for ln in non_empty)
        if min_indent == 0:
            lines = [("    " + ln) if ln.strip() else ln for ln in lines]
            text = "\n".join(lines).rstrip()
    return text


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_humaneval_check(code: str, test_code: str, entry_point: str, timeout_sec: int = 8) -> bool:
    script = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-json", required=True)
    ap.add_argument("--case-jsonl", required=True)
    args = ap.parse_args()

    subset = load_json(args.subset_json)
    by_task = {x["task_id"]: x for x in subset}
    rows = load_jsonl(args.case_jsonl)

    total = 0
    p1 = 0
    p2 = 0
    for r in rows:
        tid = r["task_id"]
        ex = by_task[tid]
        prompt = ex["prompt"]
        test = ex["test"]
        entry = ex["entry_point"]
        cands = r.get("candidate_samples", [])
        if not cands:
            continue
        total += 1
        full1 = prompt + sanitize_completion(cands[0])
        ok1 = run_humaneval_check(full1, test, entry)
        p1 += int(ok1)
        ok_any = ok1
        if len(cands) > 1:
            full2 = prompt + sanitize_completion(cands[1])
            ok_any = ok_any or run_humaneval_check(full2, test, entry)
        p2 += int(ok_any)

    out = {
        "count": total,
        "pass@1_from_saved": round(p1 / total, 4) if total else 0.0,
        "pass@2_from_saved": round(p2 / total, 4) if total else 0.0,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
