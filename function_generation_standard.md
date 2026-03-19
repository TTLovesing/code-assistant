# Python Function Generation Unified Standard (v1)

## 1) Goal

Align `SFT`, `DPO`, and `Eval` to one task definition:

- Task: `Python function completion`
- Model input: a function skeleton prompt (`def ...:` + optional docstring)
- Model output: only the function body completion

This standard is designed to match HumanEval-style evaluation behavior.

## 2) Canonical sample schema

### SFT sample

```json
{
  "instruction": "<task description + function skeleton>",
  "input": "",
  "output": "<function body completion only>"
}
```

### DPO sample

```json
{
  "prompt": "<task description + function skeleton>",
  "chosen": "<better function body completion only>",
  "rejected": "<worse function body completion only>"
}
```

## 3) Prompt format

Prompt MUST contain:

1. Optional natural language task block.
2. One Python function skeleton block that starts with `def` (or `async def`).
3. No markdown code fence.

Recommended structure:

```text
Task:
<plain text task>

Complete the following Python function:
def foo(a, b):
    """docstring..."""

```

## 4) Completion format

Completion MUST:

1. Be function-body-only (no top-level `def`, `class`, or markdown fence).
2. Keep Python indentation.
3. Not include explanation/test text.

Completion SHOULD:

1. Parse when concatenated as `prompt + completion`.
2. Implement the intended behavior.

## 5) Conversion policy

### From existing SFT

1. Extract first top-level function from original `output`.
2. Build function skeleton from extracted function signature + optional docstring.
3. Use original `instruction`/`input` as task text prefix.
4. Use extracted function body (without `def` line) as `output`.

### From existing DPO

1. Extract first top-level function from `chosen` and `rejected`.
2. Keep only pairs where chosen/rejected function names are identical.
3. Build one skeleton from the extracted signature/docstring.
4. Keep body-only completions for both sides.

## 6) Acceptance checks

Required minimum checks:

1. Schema validity: required keys present and non-empty.
2. Prompt validity:
   - Contains `def `
   - No code fence
3. Completion validity:
   - Does not start with `def`/`class`
   - No code fence
4. Parse validity:
   - `ast.parse(prompt + completion)` succeeds
5. Ratio thresholds:
   - `prompt_has_def_ratio >= 0.98`
   - `completion_has_top_level_def_ratio <= 0.02`
   - `concat_parse_ok_ratio >= 0.95`

## 7) Notes

This standard optimizes format consistency first. Semantic quality is evaluated after conversion by model metrics and failure-case analysis.
