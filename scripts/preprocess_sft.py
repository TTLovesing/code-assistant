import argparse
import ast
import hashlib
import json
import re
from typing import Optional

from datasets import load_dataset


def has_garbled_text(s: str) -> bool:
    if not s:
        return True
    bad_markers = ["\x00", "�"]
    return any(m in s for m in bad_markers)


def maybe_extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def python_related(instruction: str, output: str, tag: str) -> bool:
    blob = f"{instruction}\n{output}\n{tag}".lower()
    keys = ["python", "def ", "```python", "function", "函数"]
    return any(k in blob for k in keys)


def extract_first_function(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    func = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node
            break
    if func is None:
        return None

    lines = code.splitlines()
    start = getattr(func, "lineno", None)
    end = getattr(func, "end_lineno", None)
    if not start or not end:
        return None
    snippet = "\n".join(lines[start - 1:end]).strip()
    if not snippet.startswith(("def ", "async def ")):
        return None

    try:
        ast.parse(snippet)
    except Exception:
        return None
    return snippet


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="OpenCoder-LLM/opc-sft-stage1")
    ap.add_argument("--config", default="filtered_infinity_instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-prompt-chars", type=int, default=1200)
    ap.add_argument("--max-response-chars", type=int, default=3000)
    ap.add_argument("--max-samples", type=int, default=50000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)

    stats = {
        "total": 0,
        "drop_empty": 0,
        "drop_too_long": 0,
        "drop_garbled": 0,
        "drop_not_python": 0,
        "drop_no_function": 0,
        "drop_dup_pair": 0,
        "drop_dup_instruction": 0,
        "drop_dup_output": 0,
        "kept": 0,
    }

    seen_pair = set()
    seen_instruction = set()
    seen_output = set()

    with open(args.output, "w", encoding="utf-8") as fout:
        for ex in ds:
            if stats["kept"] >= args.max_samples:
                break

            stats["total"] += 1
            instruction = str(ex.get("instruction", "") or "").strip()
            output = str(ex.get("output", "") or "").strip()
            tag = str(ex.get("tag", "") or "").strip()

            if not instruction or not output:
                stats["drop_empty"] += 1
                continue

            if len(instruction) > args.max_prompt_chars or len(output) > args.max_response_chars:
                stats["drop_too_long"] += 1
                continue

            if has_garbled_text(instruction) or has_garbled_text(output):
                stats["drop_garbled"] += 1
                continue

            if not python_related(instruction, output, tag):
                stats["drop_not_python"] += 1
                continue

            code = maybe_extract_code(output)
            func_code = extract_first_function(code)
            if not func_code:
                stats["drop_no_function"] += 1
                continue

            out_rec = {
                "instruction": instruction,
                "input": "",
                "output": func_code,
            }

            pair_key = hashlib.sha256((norm_text(instruction) + "\n" + norm_text(func_code)).encode("utf-8")).hexdigest()
            inst_key = hashlib.sha256(norm_text(instruction).encode("utf-8")).hexdigest()
            out_key = hashlib.sha256(norm_text(func_code).encode("utf-8")).hexdigest()

            if pair_key in seen_pair:
                stats["drop_dup_pair"] += 1
                continue
            if inst_key in seen_instruction:
                stats["drop_dup_instruction"] += 1
                continue
            if out_key in seen_output:
                stats["drop_dup_output"] += 1
                continue

            seen_pair.add(pair_key)
            seen_instruction.add(inst_key)
            seen_output.add(out_key)

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    with open(args.report, "w", encoding="utf-8") as frep:
        json.dump(stats, frep, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
