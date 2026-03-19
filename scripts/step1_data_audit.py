import argparse
import ast
import json
import re
from pathlib import Path


FIM_MARKERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def text_stats(text: str) -> dict:
    text = text or ""
    has_fim = any(tok in text for tok in FIM_MARKERS)
    has_repl = "\uFFFD" in text
    has_unittest = ("import unittest" in text) or ("def test" in text)
    has_top_level_continuation = bool(re.search(r"(?m)^(def|class|from|import)\s+", text))
    non_ascii_ratio = 0.0
    if text:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii / len(text)
    return {
        "has_fim": has_fim,
        "has_repl_char": has_repl,
        "has_unittest_or_test_def": has_unittest,
        "has_top_level_continuation": has_top_level_continuation,
        "non_ascii_ratio": non_ascii_ratio,
    }


def is_body_like(text: str) -> bool:
    if not text or not text.strip():
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    # Function-completion body should normally be indented.
    indented = sum(1 for ln in lines if ln.startswith((" ", "\t")))
    return indented / len(lines) >= 0.6


def py_parse_ok(prompt: str, completion: str) -> bool:
    src = (prompt or "") + (completion or "")
    try:
        ast.parse(src)
        return True
    except Exception:
        return False


def audit_sft(path: Path) -> dict:
    total = 0
    empty_output = 0
    fim = 0
    repl = 0
    continuation = 0
    non_ascii_high = 0
    body_like = 0
    parse_ok = 0

    for row in iter_jsonl(path):
        total += 1
        ins = row.get("instruction", "")
        out = row.get("output", "")
        if not out.strip():
            empty_output += 1
        st = text_stats(out)
        fim += int(st["has_fim"])
        repl += int(st["has_repl_char"])
        continuation += int(st["has_top_level_continuation"])
        non_ascii_high += int(st["non_ascii_ratio"] > 0.05)
        body_like += int(is_body_like(out))
        parse_ok += int(py_parse_ok(ins, out))

    def r(x):
        return round(x / total, 4) if total else 0.0

    return {
        "file": str(path),
        "total": total,
        "empty_output": empty_output,
        "empty_output_rate": r(empty_output),
        "fim_rate": r(fim),
        "replacement_char_rate": r(repl),
        "top_level_continuation_rate": r(continuation),
        "high_non_ascii_rate": r(non_ascii_high),
        "body_like_rate": r(body_like),
        "parse_ok_rate": r(parse_ok),
    }


def audit_dpo(path: Path, sft_total: int, min_abs: int, min_rel: float) -> dict:
    total = 0
    chosen_empty = 0
    rejected_empty = 0
    chosen_polluted = 0
    rejected_polluted = 0
    chosen_parse_ok = 0
    rejected_parse_ok = 0
    usable_pairs = 0

    for row in iter_jsonl(path):
        total += 1
        prompt = row.get("prompt", "")
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")

        if not chosen.strip():
            chosen_empty += 1
        if not rejected.strip():
            rejected_empty += 1

        cst = text_stats(chosen)
        rst = text_stats(rejected)

        c_polluted = cst["has_fim"] or cst["has_repl_char"] or cst["has_top_level_continuation"]
        r_polluted = rst["has_fim"] or rst["has_repl_char"] or rst["has_top_level_continuation"]
        chosen_polluted += int(c_polluted)
        rejected_polluted += int(r_polluted)

        c_ok = py_parse_ok(prompt, chosen)
        r_ok = py_parse_ok(prompt, rejected)
        chosen_parse_ok += int(c_ok)
        rejected_parse_ok += int(r_ok)

        # "usable" gate: both non-empty, body-like, no severe pollution, both parse.
        usable = (
            bool(chosen.strip())
            and bool(rejected.strip())
            and is_body_like(chosen)
            and is_body_like(rejected)
            and (not c_polluted)
            and (not r_polluted)
            and c_ok
            and r_ok
        )
        usable_pairs += int(usable)

    def r(x):
        return round(x / total, 4) if total else 0.0

    threshold = max(min_abs, int(sft_total * min_rel))
    shortfall = max(0, threshold - usable_pairs)

    return {
        "file": str(path),
        "total": total,
        "chosen_empty_rate": r(chosen_empty),
        "rejected_empty_rate": r(rejected_empty),
        "chosen_polluted_rate": r(chosen_polluted),
        "rejected_polluted_rate": r(rejected_polluted),
        "chosen_parse_ok_rate": r(chosen_parse_ok),
        "rejected_parse_ok_rate": r(rejected_parse_ok),
        "usable_pairs": usable_pairs,
        "usable_rate": r(usable_pairs),
        "gate_threshold_pairs": threshold,
        "gate_pass": usable_pairs >= threshold,
        "gate_shortfall_pairs": shortfall,
        "gate_rule": f"max({min_abs}, int(SFT_total * {min_rel}))",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-jsonl", required=True)
    ap.add_argument("--dpo-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--dpo-min-abs", type=int, default=3000)
    ap.add_argument("--dpo-min-rel", type=float, default=0.2)
    args = ap.parse_args()

    sft_path = Path(args.sft_jsonl)
    dpo_path = Path(args.dpo_jsonl)

    sft = audit_sft(sft_path)
    dpo = audit_dpo(dpo_path, sft_total=sft["total"], min_abs=args.dpo_min_abs, min_rel=args.dpo_min_rel)

    result = {
        "step": "step1_data_audit",
        "sft": sft,
        "dpo": dpo,
        "decision": {
            "allow_enter_dpo_training": bool(dpo["gate_pass"]),
            "reason": "pass dpo quantity+quality gate" if dpo["gate_pass"] else "fail dpo quantity+quality gate; need data augmentation/cleanup",
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Step1 Data Audit Report",
        "",
        "## SFT",
        f"- total: `{sft['total']}`",
        f"- empty_output_rate: `{sft['empty_output_rate']}`",
        f"- fim_rate: `{sft['fim_rate']}`",
        f"- replacement_char_rate: `{sft['replacement_char_rate']}`",
        f"- top_level_continuation_rate: `{sft['top_level_continuation_rate']}`",
        f"- parse_ok_rate: `{sft['parse_ok_rate']}`",
        "",
        "## DPO",
        f"- total: `{dpo['total']}`",
        f"- usable_pairs: `{dpo['usable_pairs']}`",
        f"- usable_rate: `{dpo['usable_rate']}`",
        f"- gate_threshold_pairs: `{dpo['gate_threshold_pairs']}`",
        f"- gate_pass: `{dpo['gate_pass']}`",
        f"- gate_shortfall_pairs: `{dpo['gate_shortfall_pairs']}`",
        f"- chosen_polluted_rate: `{dpo['chosen_polluted_rate']}`",
        f"- rejected_polluted_rate: `{dpo['rejected_polluted_rate']}`",
        "",
        "## Decision",
        f"- allow_enter_dpo_training: `{result['decision']['allow_enter_dpo_training']}`",
        f"- reason: `{result['decision']['reason']}`",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

