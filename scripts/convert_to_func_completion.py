import argparse
import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return text


@dataclass
class FunctionParts:
    name: str
    skeleton: str
    completion: str


def extract_first_function_parts(text: str) -> Optional[FunctionParts]:
    src = strip_code_fence(text).replace("\r\n", "\n")
    try:
        tree = ast.parse(src)
    except Exception:
        return None

    fn: Optional[ast.AST] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node
            break
    if fn is None:
        return None

    lines = src.splitlines()
    start = getattr(fn, "lineno", None)
    end = getattr(fn, "end_lineno", None)
    if not start or not end:
        return None

    func_text = "\n".join(lines[start - 1 : end]).rstrip()
    try:
        fn_tree = ast.parse(func_text)
    except Exception:
        return None

    fn_node = fn_tree.body[0]
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    header = func_text.splitlines()[0]
    body_nodes = list(fn_node.body)

    doc = ast.get_docstring(fn_node, clean=False)
    doc_block = ""
    if doc and body_nodes and isinstance(body_nodes[0], ast.Expr):
        expr = body_nodes[0]
        if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
            ds_start = getattr(expr, "lineno", None)
            ds_end = getattr(expr, "end_lineno", None)
            if ds_start and ds_end:
                doc_block = "\n".join(func_text.splitlines()[ds_start - 1 : ds_end]).rstrip()
                body_nodes = body_nodes[1:]

    completion = ""
    if body_nodes:
        first = getattr(body_nodes[0], "lineno", None)
        last = getattr(body_nodes[-1], "end_lineno", None)
        if first and last:
            completion = "\n".join(func_text.splitlines()[first - 1 : last]).rstrip()

    if not completion:
        return None

    skeleton_lines = [header]
    if doc_block:
        skeleton_lines.append(doc_block)
    skeleton = "\n".join(skeleton_lines) + "\n"
    if not skeleton.endswith("\n"):
        skeleton += "\n"
    if not completion.endswith("\n"):
        completion += "\n"

    return FunctionParts(name=getattr(fn_node, "name", ""), skeleton=skeleton, completion=completion)


def to_comment_block(task_text: str) -> str:
    task_text = (task_text or "").replace("\r\n", "\n").strip()
    if not task_text:
        return ""
    task_text = re.sub(r"```(?:python)?", "", task_text, flags=re.I)
    task_text = task_text.replace("```", "")
    lines = [ln.strip() for ln in task_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(f"# {ln}" for ln in lines) + "\n"


def build_prompt(task_text: str, skeleton: str) -> str:
    comment_block = to_comment_block(task_text)
    return f"{comment_block}{skeleton}"


def convert_sft(rows: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {
        "total": 0,
        "drop_no_function": 0,
        "drop_empty": 0,
        "drop_top_level_def_completion": 0,
        "kept": 0,
    }
    out: List[Dict] = []
    for row in rows:
        stats["total"] += 1
        instruction = str(row.get("instruction", "") or "").strip()
        inp = str(row.get("input", "") or "").strip()
        output = str(row.get("output", "") or "").strip()
        if not output:
            stats["drop_empty"] += 1
            continue
        task_text = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
        parts = extract_first_function_parts(output)
        if not parts:
            stats["drop_no_function"] += 1
            continue
        if parts.completion.lstrip().startswith(("def ", "class ")):
            stats["drop_top_level_def_completion"] += 1
            continue
        out.append(
            {
                "instruction": build_prompt(task_text, parts.skeleton),
                "input": "",
                "output": parts.completion,
            }
        )
        stats["kept"] += 1
    return out, stats


def convert_dpo(rows: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {
        "total": 0,
        "drop_empty": 0,
        "drop_no_function": 0,
        "drop_name_mismatch": 0,
        "drop_top_level_def_completion": 0,
        "kept": 0,
    }
    out: List[Dict] = []
    for row in rows:
        stats["total"] += 1
        prompt = str(row.get("prompt", "") or "").strip()
        chosen = str(row.get("chosen", "") or "").strip()
        rejected = str(row.get("rejected", "") or "").strip()
        if not prompt or not chosen or not rejected:
            stats["drop_empty"] += 1
            continue
        c = extract_first_function_parts(chosen)
        r = extract_first_function_parts(rejected)
        if not c or not r:
            stats["drop_no_function"] += 1
            continue
        if c.name != r.name:
            stats["drop_name_mismatch"] += 1
            continue
        if c.completion.lstrip().startswith(("def ", "class ")) or r.completion.lstrip().startswith(("def ", "class ")):
            stats["drop_top_level_def_completion"] += 1
            continue
        out.append(
            {
                "prompt": build_prompt(prompt, c.skeleton),
                "chosen": c.completion,
                "rejected": r.completion,
            }
        )
        stats["kept"] += 1
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-in", required=True)
    ap.add_argument("--dpo-in", required=True)
    ap.add_argument("--sft-out", required=True)
    ap.add_argument("--dpo-out", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    sft_rows = load_jsonl(args.sft_in)
    dpo_rows = load_jsonl(args.dpo_in)

    sft_new, sft_stats = convert_sft(sft_rows)
    dpo_new, dpo_stats = convert_dpo(dpo_rows)

    dump_jsonl(args.sft_out, sft_new)
    dump_jsonl(args.dpo_out, dpo_new)

    report = {
        "sft": sft_stats,
        "dpo": dpo_stats,
        "outputs": {"sft_out": args.sft_out, "dpo_out": args.dpo_out},
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
