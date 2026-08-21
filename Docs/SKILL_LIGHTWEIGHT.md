# Skill: 轻量血液面板模型（供 AI Agents 使用）/ Lightweight Blood-Panel Model (for AI Agents)

> 中文：用户想用蒸馏出的轻量模型（blood-token Transformer + DeepGompertz）直接从
> 少量血液生化指标得到 embedding 与代谢年龄输出时，按本 Skill 操作（Notebook 4 的流程）。
>
> English: Use this skill when the user wants the distilled lightweight model
> (blood-token Transformer + DeepGompertz) to map a small blood biochemistry panel
> directly to embeddings and metabolomic age outputs (Notebook 4 workflow).

## 1. 输入要求 / Input Requirements

- 权重：`Model_Weights/Lightweight/`（`model_weights.pth`、`model_conf.json`）
  + teacher 头 `Model_Weights/DeepGompertz/model_weights.pth`（只读，取头配置与 baseline 参数）
- 输入：`(n_samples, n_features)` 矩阵，特征顺序与训练血液面板一致
  （`model_conf.json["n_features"]` = 14），用训练统计量 z-score 化；NaN 安全

## 2. 核心流程 / Core Workflow

```python
import json, sys
from pathlib import Path
REPO_ROOT = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "Src"))

import numpy as np, torch
from metageformer_torch.checkpoint import load_teacher_gompertz_config
from metageformer_torch.models import MetAgeFormer_Lightweight_DeepGompertz

with open(REPO_ROOT / "Model_Weights/Lightweight/model_conf.json") as f:
    model_conf = json.load(f)
meta = load_teacher_gompertz_config(str(REPO_ROOT / "Model_Weights/DeepGompertz/model_weights.pth"))

model = MetAgeFormer_Lightweight_DeepGompertz(
    model_conf,
    gompertz_head_config=meta["gompertz_head_config"],
    baseline_params=meta["baseline_params"],
)
model.from_distilled(str(REPO_ROOT / "Model_Weights/Lightweight/model_weights.pth"))
model.eval()

x = torch.randn(8, model_conf["n_features"]); x[torch.rand(8, model_conf["n_features"]) < 0.1] = float("nan")
age = torch.linspace(45, 80, 8)
with torch.inference_mode():
    out = model(x, age)
```

## 3. 输出 / Outputs

`embs`（512 维）、`metabolomic_age`、`age_gap`、`mortality_risk_10y`（辅助）、
`linear_predictor` / `alpha_i` / `gamma_i`。

## 4. 关键点 / Key Points

- 无需 tokenizer / NMR backbone；缺失值走 mask embedding（NaN-safe）
- 权重键：`METAGEFORMER_DISTILLED`；旧 `student.*` 前缀会被 `normalize_distilled_state_dict`
  自动重映射；legacy MLP 权重会被拒绝
- teacher 头只读，用于恢复 `gompertz_head_config` + `baseline_params`
- 演示版 Notebook：`Notebooks/4_lightweight_usage.ipynb`（含真实血液面板的可选单元格）

## 5. 验证 / Verification

- `out["embs"].shape == (n, 512)`，其余输出 `(n, 1)`
- 随机面板 + 10% NaN 上跑通即视为验证通过
