# Code Assistant Function Generation Project

[中文说明](#中文说明) | [English](#english)

---

## 中文说明

### 项目简介

这是一个围绕 Python 函数生成任务构建的训练与评估工程，基于 `LLaMA-Factory` 和 `Qwen2.5-Coder-3B-Instruct`，实现了 `Base -> SFT -> DPO -> Eval` 的完整闭环。

核心目标：
- 跑通可复现的训练评估流程
- 建立数据质量门禁和统一评测口径
- 持续优化函数生成能力（pass@1 / pass@k / 偏好指标）

### 主要能力

- 训练流程自动化：SFT、DPO、Eval 脚本化运行
- 数据质量审计：SFT/DPO 样本数量与质量门禁
- DPO 数据构建：支持 hard negative 构造与策略迭代
- 统一评测：三模型同口径对比（Base/SFT/DPO）
- 问题闭环：badcase 诊断、后处理修复、参数回归

### 项目结构

```text
.
├── scripts/                      # 核心脚本（预处理、训练、评测、诊断）
├── data/
│   ├── processed/                # 已处理数据
│   └── raw/                      # 原始/缓存数据（默认不入库）
├── reports/                      # 数据审计与转换报告
├── eval/                         # 评测输出与日志
├── train_*.yaml                  # SFT/DPO 训练配置
├── 0317.md / 0319.md             # 过程记录
├── final.md                      # 项目总结（模板/详细版）
└── README.md
```

### 快速开始

1. 准备环境（建议 conda）
2. 检查/注册数据集映射
3. 执行 SFT
4. 执行 DPO
5. 运行统一评测
6. 分析 leaderboard 与 badcase

> 提示：大模型权重、训练产物、缓存文件通常较大，建议通过 `.gitignore` 排除。

### 常用结果指标

- `pass@1`：单次生成通过率
- `pass@k`：多次采样命中率
- `preference_win_rate`：偏好比较胜率

### 当前状态

- 仓库已支持端到端复现
- 已沉淀可复用脚本、配置与审计报告
- 后续可基于新轮次实验结果持续填充 `final.md`

---

## English

### Overview

This project builds a full training/evaluation pipeline for Python function generation, based on `LLaMA-Factory` and `Qwen2.5-Coder-3B-Instruct`.

Pipeline:
- `Base -> SFT -> DPO -> Eval`

Main goals:
- Reproducible end-to-end workflow
- Data quality gating and consistent evaluation protocol
- Continuous optimization on function-generation metrics

### Key Features

- Automated training stages (SFT, DPO, Eval)
- Dataset auditing with quantity/quality gates
- DPO pair construction with hard-negative strategy
- Unified three-model benchmarking (Base/SFT/DPO)
- Badcase-driven iteration loop

### Repository Layout

```text
.
├── scripts/                      # preprocessing, training, eval, diagnostics
├── data/
│   ├── processed/                # processed datasets
│   └── raw/                      # raw/cache data (usually ignored)
├── reports/                      # audit and conversion reports
├── eval/                         # evaluation outputs and logs
├── train_*.yaml                  # SFT/DPO configs
├── 0317.md / 0319.md             # execution notes
├── final.md                      # project summary
└── README.md
```

### Quick Start

1. Prepare environment (Conda recommended)
2. Verify/register dataset mappings
3. Run SFT training
4. Run DPO training
5. Run unified evaluation
6. Analyze leaderboard and badcases

> Note: model checkpoints, training artifacts, and cache files can be very large and should usually be excluded from git.

### Core Metrics

- `pass@1`: single-shot correctness
- `pass@k`: multi-sample hit rate
- `preference_win_rate`: preference comparison win rate

### Status

- End-to-end pipeline is reproducible
- Reusable scripts/configs/reports are in place
- New experiment rounds can be summarized in `final.md`

