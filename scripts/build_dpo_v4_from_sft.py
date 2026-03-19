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


def canonical(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def is_clean_body(text: str) -> bool:
    if not text or not text.strip():
        return False
    banned = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "\uFFFD")
    if any(x in text for x in banned):
        return False
    if re.search(r"(?m)^\s*(import|from)\s+", text):
        return False
    return True


def mutate_logic(body: str):
    text = body or ""
    if not text.strip():
        return None, "empty"

    # Priority: keep syntax valid and body style similar, only alter logic.
    patterns = [
        (r"\b==\b", "!=", "cmp_eq_to_ne"),
        (r"\b!=\b", "==", "cmp_ne_to_eq"),
        (r"\b<=\b", "<", "cmp_le_to_lt"),
        (r"\b>=\b", ">", "cmp_ge_to_gt"),
        (r"\band\b", "or", "bool_and_to_or"),
        (r"\bor\b", "and", "bool_or_to_and"),
        (r"\bTrue\b", "False", "true_to_false"),
        (r"\bFalse\b", "True", "false_to_true"),
        (r"\bsorted\(([^()]*)\)\b", r"list(\1)", "sorted_to_list"),
        (r"\bmax\(([^()]*)\)\b", r"min(\1)", "max_to_min"),
        (r"\bmin\(([^()]*)\)\b", r"max(\1)", "min_to_max"),
    ]
    for pat, repl, tag in patterns:
        new_text, n = re.subn(pat, repl, text, count=1)
        if n > 0 and canonical(new_text) != canonical(text):
            return new_text, tag

    # Return-level mutation fallback.
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        m = re.match(r"^(\s*)return\s+(.+)$", ln)
        if not m:
            continue
        indent = m.group(1)
        expr = m.group(2).strip()
        if expr.startswith("-"):
            lines[i] = f"{indent}return {expr[1:].strip()}"
            return "\n".join(lines), "return_drop_minus"
        lines[i] = f"{indent}return -({expr})"
        return "\n".join(lines), "return_add_minus"

    return None, "no_mutation"


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
        "skip_dirty_chosen": 0,
        "skip_no_mutation": 0,
        "skip_parse_fail": 0,
        "skip_duplicate": 0,
        "final_pairs": 0,
        "target_pairs": args.target_pairs,
        "mutation_tags": {},
    }

    seen = set()
    records = []

    # 1) Keep existing high-quality seed DPO pairs.
    for row in iter_jsonl(seed_path):
        p = row.get("prompt", "")
        c = row.get("chosen", "")
        r = row.get("rejected", "")
        if not p.strip() or not c.strip() or not r.strip():
            continue
        if not is_clean_body(c) or not is_clean_body(r):
            continue
        if not parse_ok(p, c) or not parse_ok(p, r):
            continue
        key = sha(p + "\n@@\n" + c + "\n##\n" + r)
        if key in seen:
            continue
        seen.add(key)
        records.append({"prompt": p, "chosen": c, "rejected": r, "source": "seed_dpo"})
        stats["seed_pairs"] += 1
        if len(records) >= args.target_pairs:
            break

    # 2) Build hard negatives from SFT pairs.
    if len(records) < args.target_pairs:
        for row in iter_jsonl(sft_path):
            stats["sft_rows"] += 1
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            chosen = row.get("output", "")
            if not chosen.strip():
                stats["skip_empty"] += 1
                continue
            if not is_clean_body(chosen):
                stats["skip_dirty_chosen"] += 1
                continue

            prompt = normalize_prompt(instruction, input_text)
            rejected, tag = mutate_logic(chosen)
            if not rejected:
                stats["skip_no_mutation"] += 1
                continue
            if canonical(rejected) == canonical(chosen):
                stats["skip_no_mutation"] += 1
                continue
            if not is_clean_body(rejected):
                stats["skip_no_mutation"] += 1
                continue
            if not parse_ok(prompt, chosen) or not parse_ok(prompt, rejected):
                stats["skip_parse_fail"] += 1
                continue

            key = sha(prompt + "\n@@\n" + chosen + "\n##\n" + rejected)
            if key in seen:
                stats["skip_duplicate"] += 1
                continue

            seen.add(key)
            stats["mutation_tags"][tag] = stats["mutation_tags"].get(tag, 0) + 1
            records.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": f"sft_hardneg:{tag}",
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
