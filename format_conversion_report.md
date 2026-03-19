# Function-Completion 格式转换与验收报告（2026-03-17）

## 1. 统一标准

任务统一为：`Python 函数补全（Function Completion）`

- Prompt：包含函数骨架（`def ...:`），不含 markdown 代码围栏。
- Completion：仅函数体，不允许顶层 `def/class`，不含解释文本。
- SFT：`instruction/input/output`
- DPO：`prompt/chosen/rejected`

标准文档：`/root/code-assistant/function_generation_standard.md`

## 2. 执行命令（远端）

```bash
/root/miniconda3/envs/lf/bin/python /root/code-assistant/scripts/convert_to_func_completion.py \
  --sft-in /root/code-assistant/data/processed/sft_train.jsonl \
  --dpo-in /root/code-assistant/data/processed/dpo_train.jsonl \
  --sft-out /root/code-assistant/data/processed_v2/sft_train_func_completion.jsonl \
  --dpo-out /root/code-assistant/data/processed_v2/dpo_train_func_completion.jsonl \
  --report /root/code-assistant/reports/convert_func_completion_report.json

/root/miniconda3/envs/lf/bin/python /root/code-assistant/scripts/validate_func_completion_format.py \
  --sft /root/code-assistant/data/processed_v2/sft_train_func_completion.jsonl \
  --dpo /root/code-assistant/data/processed_v2/dpo_train_func_completion.jsonl \
  --eval-json /root/code-assistant/eval/results_strict/humaneval_subset.json \
  --report-out /root/code-assistant/reports/validate_func_completion_report.json
```

## 3. 转换结果

- SFT：`50000 -> 49108`（丢弃 892）
  - `drop_no_function=29`
  - `drop_top_level_def_completion=863`
- DPO：`1490 -> 675`（丢弃 815）
  - `drop_no_function=7`
  - `drop_name_mismatch=788`
  - `drop_top_level_def_completion=20`

## 4. 达标验收（Gate）

当前阈值：

- `sft.prompt_has_def_ratio >= 0.98`
- `sft.concat_parse_ok_ratio >= 0.95`
- `dpo.prompt_has_def_ratio >= 0.98`
- `dpo.chosen_concat_parse_ok_ratio >= 0.95`
- `dpo.rejected_concat_parse_ok_ratio >= 0.95`
- `eval.prompt_has_def_ratio >= 0.98`

当前结果：

- `summary.pass = true`
- SFT
  - `prompt_has_def_ratio = 1.0`
  - `prompt_no_fence_ratio = 1.0`
  - `output_no_top_level_def_ratio = 1.0`
  - `concat_parse_ok_ratio = 0.9985`
- DPO
  - `prompt_has_def_ratio = 1.0`
  - `chosen_concat_parse_ok_ratio = 1.0`
  - `rejected_concat_parse_ok_ratio = 1.0`
- Eval prompt
  - `prompt_has_def_ratio = 1.0`
  - `prompt_no_fence_ratio = 1.0`

## 5. 报告文件

- `/root/code-assistant/reports/convert_func_completion_report.json`
- `/root/code-assistant/reports/validate_func_completion_report.json`

时间戳（远端）：

- convert：`2026-03-17 15:52:18 +0800`
- validate：`2026-03-17 15:53:19 +0800`
