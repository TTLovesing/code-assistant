import argparse
import ast
import json
import re
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_def(text: str) -> bool:
    return "def " in (text or "")


def has_fence(text: str) -> bool:
    return "```" in (text or "")


def starts_top_level_block(text: str) -> bool:
    t = (text or "").lstrip()
    return t.startswith("def ") or t.startswith("class ")


def parse_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def ratio(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def validate_sft(rows: List[Dict]) -> Dict:
    n = len(rows)
    prompt_has_def = 0
    prompt_no_fence = 0
    output_no_fence = 0
    output_no_top_def = 0
    concat_parse_ok = 0
    bad_samples: List[Dict] = []

    for i, r in enumerate(rows):
        prompt = str(r.get("instruction", "") or "")
        out = str(r.get("output", "") or "")
        p_has_def = has_def(prompt)
        p_no_fence = not has_fence(prompt)
        o_no_fence = not has_fence(out)
        o_no_top_def = not starts_top_level_block(out)
        parse = parse_ok(prompt + out)

        prompt_has_def += int(p_has_def)
        prompt_no_fence += int(p_no_fence)
        output_no_fence += int(o_no_fence)
        output_no_top_def += int(o_no_top_def)
        concat_parse_ok += int(parse)

        if len(bad_samples) < 20 and not (p_has_def and p_no_fence and o_no_fence and o_no_top_def and parse):
            bad_samples.append(
                {
                    "idx": i,
                    "prompt_head": prompt[:180],
                    "output_head": out[:180],
                    "flags": {
                        "prompt_has_def": p_has_def,
                        "prompt_no_fence": p_no_fence,
                        "output_no_fence": o_no_fence,
                        "output_no_top_def": o_no_top_def,
                        "concat_parse_ok": parse,
                    },
                }
            )

    return {
        "count": n,
        "metrics": {
            "prompt_has_def_ratio": ratio(prompt_has_def, n),
            "prompt_no_fence_ratio": ratio(prompt_no_fence, n),
            "output_no_fence_ratio": ratio(output_no_fence, n),
            "output_no_top_level_def_ratio": ratio(output_no_top_def, n),
            "concat_parse_ok_ratio": ratio(concat_parse_ok, n),
        },
        "bad_samples": bad_samples,
    }


def validate_dpo(rows: List[Dict]) -> Dict:
    n = len(rows)
    prompt_has_def = 0
    prompt_no_fence = 0
    chosen_no_top_def = 0
    rejected_no_top_def = 0
    chosen_parse_ok = 0
    rejected_parse_ok = 0
    bad_samples: List[Dict] = []

    for i, r in enumerate(rows):
        prompt = str(r.get("prompt", "") or "")
        c = str(r.get("chosen", "") or "")
        rej = str(r.get("rejected", "") or "")

        p_has_def = has_def(prompt)
        p_no_fence = not has_fence(prompt)
        c_no_top = not starts_top_level_block(c)
        r_no_top = not starts_top_level_block(rej)
        c_parse = parse_ok(prompt + c)
        r_parse = parse_ok(prompt + rej)

        prompt_has_def += int(p_has_def)
        prompt_no_fence += int(p_no_fence)
        chosen_no_top_def += int(c_no_top)
        rejected_no_top_def += int(r_no_top)
        chosen_parse_ok += int(c_parse)
        rejected_parse_ok += int(r_parse)

        if len(bad_samples) < 20 and not (p_has_def and p_no_fence and c_no_top and r_no_top and c_parse and r_parse):
            bad_samples.append(
                {
                    "idx": i,
                    "prompt_head": prompt[:180],
                    "chosen_head": c[:180],
                    "rejected_head": rej[:180],
                    "flags": {
                        "prompt_has_def": p_has_def,
                        "prompt_no_fence": p_no_fence,
                        "chosen_no_top_def": c_no_top,
                        "rejected_no_top_def": r_no_top,
                        "chosen_parse_ok": c_parse,
                        "rejected_parse_ok": r_parse,
                    },
                }
            )

    return {
        "count": n,
        "metrics": {
            "prompt_has_def_ratio": ratio(prompt_has_def, n),
            "prompt_no_fence_ratio": ratio(prompt_no_fence, n),
            "chosen_no_top_level_def_ratio": ratio(chosen_no_top_def, n),
            "rejected_no_top_level_def_ratio": ratio(rejected_no_top_def, n),
            "chosen_concat_parse_ok_ratio": ratio(chosen_parse_ok, n),
            "rejected_concat_parse_ok_ratio": ratio(rejected_parse_ok, n),
        },
        "bad_samples": bad_samples,
    }


def validate_eval_prompts(eval_json: str) -> Dict:
    with open(eval_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    n = len(rows)
    has_def_cnt = 0
    no_fence_cnt = 0
    for row in rows:
        p = str(row.get("prompt", "") or "")
        has_def_cnt += int(has_def(p))
        no_fence_cnt += int(not has_fence(p))
    return {
        "count": n,
        "metrics": {
            "prompt_has_def_ratio": ratio(has_def_cnt, n),
            "prompt_no_fence_ratio": ratio(no_fence_cnt, n),
        },
    }


def pass_gate(sft: Dict, dpo: Dict, evalp: Dict) -> Tuple[bool, List[str]]:
    checks = [
        ("sft.prompt_has_def_ratio", sft["metrics"]["prompt_has_def_ratio"] >= 0.98),
        ("sft.concat_parse_ok_ratio", sft["metrics"]["concat_parse_ok_ratio"] >= 0.95),
        ("dpo.prompt_has_def_ratio", dpo["metrics"]["prompt_has_def_ratio"] >= 0.98),
        ("dpo.chosen_concat_parse_ok_ratio", dpo["metrics"]["chosen_concat_parse_ok_ratio"] >= 0.95),
        ("dpo.rejected_concat_parse_ok_ratio", dpo["metrics"]["rejected_concat_parse_ok_ratio"] >= 0.95),
        ("eval.prompt_has_def_ratio", evalp["metrics"]["prompt_has_def_ratio"] >= 0.98),
    ]
    failed = [name for name, ok in checks if not ok]
    return len(failed) == 0, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    sft_rows = load_jsonl(args.sft)
    dpo_rows = load_jsonl(args.dpo)

    sft_r = validate_sft(sft_rows)
    dpo_r = validate_dpo(dpo_rows)
    eval_r = validate_eval_prompts(args.eval_json)
    ok, failed = pass_gate(sft_r, dpo_r, eval_r)

    report = {
        "summary": {"pass": ok, "failed_checks": failed},
        "sft": sft_r,
        "dpo": dpo_r,
        "eval": eval_r,
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
