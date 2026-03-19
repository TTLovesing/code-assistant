import argparse
import ast
import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Optional

from datasets import load_dataset


def has_garbled_text(s: str) -> bool:
    if not s:
        return True
    return "\x00" in s or "�" in s


def maybe_extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m2:
        return m2.group(1).strip()
    return text.strip()


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


def python_related(prompt: str, chosen: str, rejected: str) -> bool:
    blob = f"{prompt}\n{chosen}\n{rejected}".lower()
    keys = ["python", "def ", "```python", "函数"]
    return any(k in blob for k in keys)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def canonical_code(s: str) -> str:
    s = re.sub(r"#.*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def parse_rating(v) -> Optional[float]:
    try:
        return float(str(v))
    except Exception:
        return None


def pick_pair(example):
    responses = example.get("responses", []) or []
    annotations = example.get("annotations", []) or []

    model_to_resp = {}
    for r in responses:
        m = r.get("model")
        txt = r.get("response")
        if m and isinstance(txt, str):
            model_to_resp[m] = txt

    ratings = []
    for a in annotations:
        m = a.get("model")
        rt = parse_rating(a.get("rating"))
        if m in model_to_resp and rt is not None:
            ratings.append((m, rt))

    if len(ratings) < 2:
        return None

    # choose highest and lowest rating
    max_rating = max(x[1] for x in ratings)
    min_rating = min(x[1] for x in ratings)
    if max_rating <= min_rating:
        return None

    hi_models = sorted([m for m, r in ratings if r == max_rating])
    lo_models = sorted([m for m, r in ratings if r == min_rating])
    chosen_model = hi_models[0]
    rejected_model = lo_models[0]

    return model_to_resp[chosen_model], model_to_resp[rejected_model], max_rating, min_rating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coseal/CodeUltraFeedback")
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-prompt-chars", type=int, default=1200)
    ap.add_argument("--max-response-chars", type=int, default=3000)
    ap.add_argument("--min-rating-gap", type=float, default=1.0)
    ap.add_argument("--low-info-sim", type=float, default=0.95)
    ap.add_argument("--max-pairs", type=int, default=10000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)

    stats = {
        "total": 0,
        "drop_empty": 0,
        "drop_no_pair": 0,
        "drop_small_gap": 0,
        "drop_too_long": 0,
        "drop_garbled": 0,
        "drop_not_python": 0,
        "drop_no_function": 0,
        "drop_low_info": 0,
        "drop_dup_pair": 0,
        "drop_dup_prompt": 0,
        "kept": 0,
    }

    seen_pair = set()
    seen_prompt = set()

    with open(args.output, "w", encoding="utf-8") as fout:
        for ex in ds:
            if stats["kept"] >= args.max_pairs:
                break

            stats["total"] += 1
            prompt = str(ex.get("instruction", "") or "").strip()
            if not prompt:
                stats["drop_empty"] += 1
                continue

            pair = pick_pair(ex)
            if not pair:
                stats["drop_no_pair"] += 1
                continue

            chosen_raw, rejected_raw, hi, lo = pair
            if (hi - lo) < args.min_rating_gap:
                stats["drop_small_gap"] += 1
                continue

            if len(prompt) > args.max_prompt_chars or len(chosen_raw) > args.max_response_chars or len(rejected_raw) > args.max_response_chars:
                stats["drop_too_long"] += 1
                continue

            if has_garbled_text(prompt) or has_garbled_text(chosen_raw) or has_garbled_text(rejected_raw):
                stats["drop_garbled"] += 1
                continue

            if not python_related(prompt, chosen_raw, rejected_raw):
                stats["drop_not_python"] += 1
                continue

            chosen_code = extract_first_function(maybe_extract_code(chosen_raw))
            rejected_code = extract_first_function(maybe_extract_code(rejected_raw))
            if not chosen_code or not rejected_code:
                stats["drop_no_function"] += 1
                continue

            c1 = canonical_code(chosen_code)
            c2 = canonical_code(rejected_code)
            if not c1 or not c2:
                stats["drop_low_info"] += 1
                continue
            if c1 == c2:
                stats["drop_low_info"] += 1
                continue
            if SequenceMatcher(None, c1, c2).ratio() >= args.low_info_sim:
                stats["drop_low_info"] += 1
                continue

            rec = {
                "prompt": prompt,
                "chosen": chosen_code,
                "rejected": rejected_code,
            }

            pair_key = hashlib.sha256((norm_text(prompt) + "\n" + norm_text(chosen_code) + "\n" + norm_text(rejected_code)).encode("utf-8")).hexdigest()
            prompt_key = hashlib.sha256(norm_text(prompt).encode("utf-8")).hexdigest()

            if pair_key in seen_pair:
                stats["drop_dup_pair"] += 1
                continue
            if prompt_key in seen_prompt:
                stats["drop_dup_prompt"] += 1
                continue

            seen_pair.add(pair_key)
            seen_prompt.add(prompt_key)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
