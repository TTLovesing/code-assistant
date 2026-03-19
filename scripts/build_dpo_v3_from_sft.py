import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_ok(prompt: str, body: str) -> bool:
    try:
        ast.parse((prompt or "") + (body or ""))
        return True
    except Exception:
        return False


def normalize_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction or ""
    input_text = input_text or ""
    if input_text.strip():
        if instruction.endswith("\n"):
            return instruction + input_text
        return instruction + "\n" + input_text
    return instruction


def mutate_body(body: str) -> str:
    text = body or ""
    if not text.strip():
        return text

    rules = [
        (r"\b==\b", "!="),
        (r"\b!=\b", "=="),
        (r"\b<=\b", "<"),
        (r"\b>=\b", ">"),
        (r"\b<\b", "<="),
        (r"\b>\b", ">="),
        (r"\bTrue\b", "False"),
        (r"\bFalse\b", "True"),
        (r"\breturn\s+0\b", "return 1"),
        (r"\breturn\s+1\b", "return 0"),
        (r"\+", "-"),
        (r"-", "+"),
        (r"\*", "//"),
        (r"//", "*"),
    ]
    for pat, repl in rules:
        new_text, n = re.subn(pat, repl, text, count=1)
        if n > 0 and new_text != text:
            return new_text

    # Fallback: append a parse-safe but likely wrong bias for numeric returns.
    if re.search(r"\breturn\b", text):
        lines = text.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            ln = lines[i]
            m = re.match(r"^(\s*)return\s+(.+)$", ln)
            if m:
                indent = m.group(1)
                expr = m.group(2).strip()
                lines[i] = f"{indent}return ({expr}) + 1"
                return "\n".join(lines)
    return text


def canonical(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-jsonl", required=True)
    ap.add_argument("--seed-dpo-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--target-pairs", type=int, default=10000)
    args = ap.parse_args()

    sft_path = Path(args.sft_jsonl)
    seed_path = Path(args.seed_dpo_jsonl)
    out_path = Path(args.out_jsonl)
    rep_path = Path(args.report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "seed_pairs": 0,
        "sft_rows": 0,
        "built_from_sft": 0,
        "skip_empty": 0,
        "skip_parse_fail": 0,
        "skip_no_change": 0,
        "skip_duplicate": 0,
        "final_pairs": 0,
        "target_pairs": args.target_pairs,
    }

    seen = set()
    records = []

    # Keep existing DPO pairs first.
    for row in iter_jsonl(seed_path):
        p = row.get("prompt", "")
        c = row.get("chosen", "")
        r = row.get("rejected", "")
        if not p.strip() or not c.strip() or not r.strip():
            continue
        key = sha(p + "\n@@\n" + c + "\n##\n" + r)
        if key in seen:
            continue
        seen.add(key)
        records.append({"prompt": p, "chosen": c, "rejected": r, "source": "seed_dpo"})
        stats["seed_pairs"] += 1
        if len(records) >= args.target_pairs:
            break

    if len(records) < args.target_pairs:
        for row in iter_jsonl(sft_path):
            stats["sft_rows"] += 1
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            chosen = row.get("output", "")
            if not chosen.strip():
                stats["skip_empty"] += 1
                continue
            prompt = normalize_prompt(instruction, input_text)
            rejected = mutate_body(chosen)
            if canonical(rejected) == canonical(chosen):
                stats["skip_no_change"] += 1
                continue
            if not parse_ok(prompt, chosen) or not parse_ok(prompt, rejected):
                stats["skip_parse_fail"] += 1
                continue

            key = sha(prompt + "\n@@\n" + chosen + "\n##\n" + rejected)
            if key in seen:
                stats["skip_duplicate"] += 1
                continue
            seen.add(key)
            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "sft_mutation",
                }
            )
            stats["built_from_sft"] += 1
            if len(records) >= args.target_pairs:
                break

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["final_pairs"] = len(records)
    stats["gate_ready"] = len(records) >= args.target_pairs

    with rep_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

