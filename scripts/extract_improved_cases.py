import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shorten(text: str, limit: int = 280) -> str:
    text = (text or "").replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="eval output dir")
    ap.add_argument("--out-txt", required=True, help="output txt path")
    ap.add_argument("--max-cases", type=int, default=12)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    per = res_dir / "per_model_cases"
    base_p = per / "base_humaneval.jsonl"
    sft_p = per / "sft_humaneval.jsonl"
    dpo_p = per / "dpo_humaneval.jsonl"

    if not base_p.exists() or not sft_p.exists():
        raise SystemExit(f"missing required files: {base_p} or {sft_p}")

    base_rows = {x["task_id"]: x for x in load_jsonl(base_p)}
    sft_rows = {x["task_id"]: x for x in load_jsonl(sft_p)}
    dpo_rows = {x["task_id"]: x for x in load_jsonl(dpo_p)} if dpo_p.exists() else {}

    improved = []
    for task_id, b in base_rows.items():
        s = sft_rows.get(task_id)
        d = dpo_rows.get(task_id)
        if s is None:
            continue
        b1 = bool(b.get("pass_at_1", False))
        s1 = bool(s.get("pass_at_1", False))
        d1 = bool(d.get("pass_at_1", False)) if d else False

        # badcase improved by sft or dpo
        if (not b1) and (s1 or d1):
            winner = "SFT" if s1 else "DPO"
            improved.append((task_id, winner, b, s, d))

    improved.sort(key=lambda x: x[0])
    improved = improved[: args.max_cases]

    out = []
    out.append("Improved Badcases (base fail -> sft or dpo success)")
    out.append(f"results_dir: {res_dir}")
    out.append(f"total_improved_found: {len(improved)}")
    out.append("")

    if not improved:
        out.append("No improved badcases found under current outputs.")
    else:
        for idx, (task_id, winner, b, s, d) in enumerate(improved, 1):
            b_sample = ""
            s_sample = ""
            d_sample = ""
            if b.get("candidate_samples"):
                b_sample = b["candidate_samples"][0]
            if s.get("candidate_samples"):
                s_sample = s["candidate_samples"][0]
            if d and d.get("candidate_samples"):
                d_sample = d["candidate_samples"][0]

            out.append(f"[Case {idx}] {task_id}")
            out.append(f"- improved_by: {winner}")
            out.append(f"- base pass@1: {bool(b.get('pass_at_1', False))}")
            out.append(f"- sft  pass@1: {bool(s.get('pass_at_1', False))}")
            out.append(f"- dpo  pass@1: {bool(d.get('pass_at_1', False)) if d else 'N/A'}")
            out.append("- base sample:")
            out.append(shorten(b_sample))
            out.append("- sft sample:")
            out.append(shorten(s_sample))
            out.append("- dpo sample:")
            out.append(shorten(d_sample) if d else "N/A")
            out.append("")

    Path(args.out_txt).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written: {args.out_txt}")


if __name__ == "__main__":
    main()
