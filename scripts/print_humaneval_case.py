import argparse
import json
from pathlib import Path


def _find_prompt(tasks_obj, task_id: str) -> str | None:
    if isinstance(tasks_obj, list):
        for t in tasks_obj:
            if str(t.get("task_id")) == task_id:
                return t.get("prompt") or t.get("instruction") or t.get("question")
        return None
    if isinstance(tasks_obj, dict):
        t = tasks_obj.get(task_id)
        if isinstance(t, dict):
            return t.get("prompt") or t.get("instruction") or t.get("question")
    return None


def _find_row(path: Path, task_id: str) -> dict | None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task_id") == task_id:
                return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--results-dir", default="/root/code-assistant/eval/results_full_0319")
    args = ap.parse_args()

    task_id = args.task_id
    res = Path(args.results_dir)
    subset = json.load((res / "humaneval_subset.json").open("r", encoding="utf-8"))
    prompt = _find_prompt(subset, task_id)

    print("TASK_ID:", task_id)
    print("\n=== PROMPT ===")
    print(prompt if prompt else "(prompt_not_found)")

    pm = res / "per_model_cases"
    files = {
        "BASE": pm / "base_humaneval.jsonl",
        "SFT": pm / "sft_humaneval.jsonl",
        "DPO": pm / "dpo_humaneval.jsonl",
    }

    for name, path in files.items():
        r = _find_row(path, task_id)
        if not r:
            print(f"\n=== {name} (missing) ===")
            continue
        samples = r.get("candidate_samples") or [""]
        first = samples[0] if samples else ""
        print(f"\n=== {name} (pass@1={r.get('pass_at_1')}) ===")
        print(first)


if __name__ == "__main__":
    main()

