import json
from pathlib import Path


def _first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    return json.loads(line)


def _find_prompt(tasks_obj, task_id: str) -> str | None:
    # eval_step10 may store the subset as list[dict] or dict keyed by task_id
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


def main() -> None:
    out = Path("/root/code-assistant/eval/results_full_0319")
    base = _first_jsonl(out / "per_model_cases" / "base_humaneval.jsonl")
    sft = _first_jsonl(out / "per_model_cases" / "sft_humaneval.jsonl")
    dpo = _first_jsonl(out / "per_model_cases" / "dpo_humaneval.jsonl")

    task_id = base["task_id"]
    if not (sft.get("task_id") == task_id and dpo.get("task_id") == task_id):
        raise SystemExit("First task_id mismatch across models.")

    tasks = json.load((out / "humaneval_subset.json").open("r", encoding="utf-8"))
    prompt = _find_prompt(tasks, task_id)

    print("TASK_ID:", task_id)
    print("ENTRY_POINT:", base.get("entry_point"))
    print("\n=== PROMPT ===")
    print(prompt if prompt else "(prompt_not_found)")

    def show(name: str, obj: dict) -> None:
        samples = obj.get("candidate_samples") or [""]
        first = samples[0] if samples else ""
        print(f"\n=== {name} (pass@1={obj.get('pass_at_1')}) ===")
        print(first)

    show("BASE", base)
    show("SFT", sft)
    show("DPO", dpo)


if __name__ == "__main__":
    main()

