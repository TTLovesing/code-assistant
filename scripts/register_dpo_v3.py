import json
from pathlib import Path


def main():
    p = Path("/root/code-assistant/LLaMA-Factory/data/dataset_info.json")
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["dpo_pyfunc_v3"] = {
        "file_name": "dpo_train_func_completion_v3.jsonl",
        "ranking": True,
        "columns": {
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("registered:dpo_pyfunc_v3")


if __name__ == "__main__":
    main()

