# Step1 Data Audit Report

## SFT
- total: `49108`
- empty_output_rate: `0.0`
- fim_rate: `0.0`
- replacement_char_rate: `0.0`
- top_level_continuation_rate: `0.0`
- parse_ok_rate: `0.9985`

## DPO
- total: `675`
- usable_pairs: `675`
- usable_rate: `1.0`
- gate_threshold_pairs: `9821`
- gate_pass: `False`
- gate_shortfall_pairs: `9146`
- chosen_polluted_rate: `0.0`
- rejected_polluted_rate: `0.0`

## Decision
- allow_enter_dpo_training: `False`
- reason: `fail dpo quantity+quality gate; need data augmentation/cleanup`
