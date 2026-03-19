import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_completion(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if "```" in text:
        # Keep first fenced code block if present.
        m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1)

    # Remove common non-solution tails and test snippets.
    stops = [
        "\nclass ",
        "\nif __name__",
        "\nprint(",
        "\ndef test",
        "\ndef check",
        "\n# Test",
        "\n# test",
        "\n# Check",
        "\n**Created Question**",
        "\nHere",
        "\nassert ",
        "\n```",
    ]
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut].rstrip()

    # Handle malformed continuation like "return xdef foo(...)".
    m = re.search(r"(?<!\n)def\s+[A-Za-z_]\w*\s*\(", text)
    if m:
        text = text[: m.start()].rstrip()
    return text


_FIM_MARKERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")


def sanitize_completion(prompt: str, completion: str) -> str:
    """
    Make model completion safer for HumanEval-style unit tests by:
    - dropping FIM markers and everything after them
    - removing common mojibake/replacement chars and non-printables
    - truncating on new top-level defs/imports that often indicate unwanted continuation
    - indenting unindented code as a fallback (body-only contract)
    """
    if not completion:
        return ""

    text = completion.replace("\r\n", "\n")

    for m in _FIM_MARKERS:
        i = text.find(m)
        if i != -1:
            text = text[:i]

    # Drop replacement char and other non-printables (keep newline/tab + ASCII).
    text = text.replace("\uFFFD", "")
    text = "".join(ch for ch in text if ch in ("\n", "\t") or 32 <= ord(ch) <= 126)

    # Truncate on common "top-level" continuations (extra defs/imports) that often get cut mid-way.
    out_lines = []
    for line in text.split("\n"):
        if re.match(r"^(def|class|from|import)\s+", line):
            break
        if line.startswith("<|") or line.startswith("#!"):
            break
        # e.g. string_xor.__doc__ += ... at column 0: likely unwanted execution-time code.
        if line and not line.startswith((" ", "\t")) and re.match(r"^[A-Za-z_]\w*(\.\w+)+\s*[\+\-*/]?=", line):
            break
        out_lines.append(line)

    text = "\n".join(out_lines).rstrip()

    # If the model forgot indentation, indent all non-empty lines as a last resort.
    lines = text.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        min_indent = min(len(ln) - len(ln.lstrip(" \t")) for ln in non_empty)
        if min_indent == 0:
            lines = [("    " + ln) if ln.strip() else ln for ln in lines]
            text = "\n".join(lines).rstrip()

    return text


def run_humaneval_check(code: str, test_code: str, entry_point: str, timeout_sec: int = 8) -> bool:
    # HumanEval's check(candidate) expects a callable, not a string name.
    # Passing check("foo") will always fail even when function foo is correct.
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


def generation_prompt(tokenizer, prompt: str, mode: str) -> str:
    if mode == "strict_chat" and getattr(tokenizer, "chat_template", None):
        user_msg = (
            "Complete the given Python function.\n"
            "Rules:\n"
            "1) Output only valid Python code.\n"
            "2) Do not add tests, explanations, markdown, or extra functions.\n"
            "3) Only continue the target function body.\n\n"
            "Function skeleton:\n"
            f"{prompt}"
        )
        msgs = [
            {"role": "system", "content": "You are a precise Python coding assistant."},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    k: int,
    max_new_tokens: int = 256,
    prompt_mode: str = "continuation",
) -> List[str]:
    device = model.device
    model_input_text = generation_prompt(tokenizer, prompt, prompt_mode)
    inputs = tokenizer(model_input_text, return_tensors="pt").to(device)
    outs = []
    for i in range(k):
        do_sample = i > 0
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.8 if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[1] :]
        txt = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outs.append(extract_completion(txt))
    return outs


def build_full_code(prompt: str, completion: str) -> str:
    # Keep prompt as-is so original indentation contract is preserved.
    return prompt + completion


def chat_text(tokenizer, user_prompt: str, assistant_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return user_prompt + "\n" + assistant_text


def avg_logprob(model, tokenizer, user_prompt: str, assistant_text: str) -> float:
    text = chat_text(tokenizer, user_prompt, assistant_text)
    toks = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = toks["input_ids"]
    with torch.no_grad():
        out = model(**toks)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.mean().item())



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_model(base_model: str, adapter: str = ""):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@dataclass
class ModelSpec:
    name: str
    adapter: str


def evaluate_model(
    model_name: str,
    spec: ModelSpec,
    humaneval_tasks,
    dpo_pairs,
    pass_k: int,
    out_dir: str,
    prompt_mode: str,
):
    model, tok = load_model(model_name, spec.adapter)

    model_case_dir = os.path.join(out_dir, "per_model_cases")
    os.makedirs(model_case_dir, exist_ok=True)
    humaneval_case_path = os.path.join(model_case_dir, f"{spec.name}_humaneval.jsonl")
    pref_case_path = os.path.join(model_case_dir, f"{spec.name}_preference.jsonl")
    open(humaneval_case_path, "w", encoding="utf-8").close()
    open(pref_case_path, "w", encoding="utf-8").close()

    p1_ok = 0
    pk_ok = 0
    total = len(humaneval_tasks)

    he_bar = tqdm(humaneval_tasks, desc=f"[{spec.name}] HumanEval", leave=True)
    for idx, ex in enumerate(he_bar):
        prompt = ex["prompt"]
        tests = ex["test"]
        ep = ex["entry_point"]
        task_id = ex.get("task_id", f"task_{idx}")

        cands = generate_candidates(model, tok, prompt, k=pass_k, prompt_mode=prompt_mode)
        cands = [sanitize_completion(prompt, c) for c in cands]
        results = []
        for c in cands:
            full_code = build_full_code(prompt, c)
            results.append(run_humaneval_check(full_code, tests, ep))

        p1_ok += 1 if results[0] else 0
        pk_ok += 1 if any(results) else 0
        he_bar.set_postfix(pass1=f"{p1_ok}/{idx+1}", passk=f"{pk_ok}/{idx+1}")

        append_jsonl(
            humaneval_case_path,
            {
                "task_id": task_id,
                "entry_point": ep,
                "pass_at_1": bool(results[0]),
                "pass_at_k": bool(any(results)),
                "candidate_results": results,
                "candidate_samples": cands[:2],
            },
        )

    pass1 = p1_ok / total if total else 0.0
    passk = pk_ok / total if total else 0.0
    unit_test_pass_rate = pass1

    wins = 0
    pref_total = len(dpo_pairs)
    pref_bar = tqdm(dpo_pairs, desc=f"[{spec.name}] Preference", leave=True)
    for idx, pair in enumerate(pref_bar):
        p = pair["prompt"]
        c = pair["chosen"]
        r = pair["rejected"]
        lc = avg_logprob(model, tok, p, c)
        lr = avg_logprob(model, tok, p, r)
        chosen_win = lc > lr
        if chosen_win:
            wins += 1
        pref_bar.set_postfix(win=f"{wins}/{idx+1}")

        append_jsonl(
            pref_case_path,
            {
                "idx": idx,
                "chosen_logprob": round(lc, 6),
                "rejected_logprob": round(lr, 6),
                "chosen_win": bool(chosen_win),
                "prompt_head": p[:140],
            },
        )

    pref = wins / pref_total if pref_total else 0.0

    del model
    torch.cuda.empty_cache()

    return {
        "model": spec.name,
        "pass@1": round(pass1, 4),
        f"pass@{pass_k}": round(passk, 4),
        "unit_test_pass_rate": round(unit_test_pass_rate, 4),
        "preference_win_rate": round(pref, 4),
        "humaneval_tasks": total,
        "preference_pairs": pref_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--dpo-pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-humaneval", type=int, default=20)
    ap.add_argument("--num-pref", type=int, default=100)
    ap.add_argument("--pass-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--humaneval-dataset", default="openai_humaneval")
    ap.add_argument("--humaneval-subset-json", default="")
    ap.add_argument("--prompt-mode", choices=["continuation", "strict_chat"], default="continuation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    if args.humaneval_subset_json and os.path.exists(args.humaneval_subset_json):
        print(f"Loading HumanEval subset from file: {args.humaneval_subset_json}", flush=True)
        humaneval_tasks = load_json(args.humaneval_subset_json)
        humaneval_tasks = humaneval_tasks[: args.num_humaneval]
    else:
        print("Loading HumanEval dataset...", flush=True)
        he = load_dataset(args.humaneval_dataset, split="test")
        idxs = list(range(len(he)))
        random.shuffle(idxs)
        idxs = idxs[: args.num_humaneval]
        humaneval_tasks = [he[i] for i in idxs]

    save_json(os.path.join(args.out_dir, "humaneval_subset.json"), humaneval_tasks)

    print("Loading DPO pairs...", flush=True)
    pairs = load_jsonl(args.dpo_pairs)
    random.shuffle(pairs)
    dpo_pairs = pairs[: args.num_pref]
    save_json(os.path.join(args.out_dir, "dpo_subset_preview.json"), dpo_pairs[:20])

    specs = [
        ModelSpec("base", ""),
        ModelSpec("sft", args.sft_adapter),
        ModelSpec("dpo", args.dpo_adapter),
    ]

    rows = []
    save_json(os.path.join(args.out_dir, "leaderboard.partial.json"), rows)
    for s in specs:
        print(f"=== Evaluating {s.name} ===", flush=True)
        row = evaluate_model(
            args.base_model,
            s,
            humaneval_tasks,
            dpo_pairs,
            args.pass_k,
            args.out_dir,
            args.prompt_mode,
        )
        rows.append(row)
        save_json(os.path.join(args.out_dir, "leaderboard.partial.json"), rows)
        print(f"Finished {s.name}: {row}", flush=True)

    out_json = os.path.join(args.out_dir, "leaderboard.json")
    out_csv = os.path.join(args.out_dir, "leaderboard.csv")
    save_json(out_json, rows)

    keys = [
        "model",
        "pass@1",
        f"pass@{args.pass_k}",
        "unit_test_pass_rate",
        "preference_win_rate",
        "humaneval_tasks",
        "preference_pairs",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("DONE", out_csv)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()



