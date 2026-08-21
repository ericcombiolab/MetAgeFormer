# Skill: Metabolomic Age (DeepGompertz, for AI Agents) / 代谢年龄预测（DeepGompertz，供 AI Agents 使用）

> English: Use this skill when the user wants to predict **metabolomic age** and
> **age gap (age acceleration)** with the backbone + DeepGompertz head
> (Notebook 2 workflow). `mortality_risk_10y` is only an auxiliary output — the
> focus is always the metabolomic age.
>
> 中文：用户想用 backbone + DeepGompertz 头预测 **metabolomic age（代谢年龄）** 与
> **age gap（年龄加速）** 时按本 Skill 操作（Notebook 2 的流程）。
> `mortality_risk_10y` 只是辅助输出，重点永远是代谢年龄。

## 1. Input Requirements / 输入要求

- Weights: `Model_Weights/MetAgeFormer/` + `Model_Weights/DeepGompertz/model_weights.pth`
- Data: AnnData NMR `.h5ad` with `layers['Z-score normalized']` and obs age column
  `"Age at assessment (estimated)"`
- No restricted data? Use `Data/fake/NMR_dataset_fake/` (fake data)

- 权重：`Model_Weights/MetAgeFormer/` + `Model_Weights/DeepGompertz/model_weights.pth`
- 数据：AnnData NMR `.h5ad`，需 `layers['Z-score normalized']` 和 obs 年龄列
  `"Age at assessment (estimated)"`
- 无真实数据时用 `Data/fake/NMR_dataset_fake/`（假数据）

## 2. Core Workflow / 核心流程

```python
import sys
from pathlib import Path
REPO_ROOT = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "Src"))

import anndata as ad
from common.constants import AGE_COL
from finetune.deep_gompertz.eval_common import load_model, predict

model, tokenizer, pretrained_config, head_config = load_model(
    str(REPO_ROOT / "Model_Weights/MetAgeFormer"),
    str(REPO_ROOT / "Model_Weights/DeepGompertz"), "cpu")

adata = ad.read_h5ad("Data/fake/NMR_dataset_fake/val.h5ad")
prediction, _ = predict(model, tokenizer, adata,
                        data_layer="Z-score normalized", age_col=AGE_COL,
                        device="cpu", batch_size=256, num_workers=0, prefetch_factor=2)
```

## 3. Output Columns / 输出列

| Column | Meaning / 含义 | Priority / 主次 |
|---|---|---|
| `metabolomic_age` | metabolomic aging clock (years) / 代谢年龄（年） | **primary / 主** |
| `age_gap` | age acceleration = metabolomic_age − chronological_age / 年龄加速 | **primary / 主** |
| `chronological_age` | observed age / 观测年龄 | reference / 参考 |
| `mortality_risk_10y` | 10-year all-cause mortality risk / 10 年全因死亡风险 | auxiliary / 辅助 |
| `linear_predictor` / `alpha_i` / `gamma_i` | Gompertz risk components / Gompertz 风险分量 | internal / 内部量 |

## 4. Key Points / 关键点

- `load_model` reads the three backbone files + the DeepGompertz head
  `model_weights.pth` (internal keys: `state_dict` / `config` / `baseline_params`)
- `predict` returns `(prediction DataFrame, None)`; batched inference, NaN-safe
- Demo notebook: `Notebooks/2_metabolomic_age.ipynb` (optional clock scatter plot, matplotlib)
- When describing this capability, lead with "Metabolomic Age / aging clock"; mortality
  risk is an auxiliary output

- `load_model` 读取 backbone 三个文件 + DeepGompertz 头的 `model_weights.pth`
  （内部键：`state_dict` / `config` / `baseline_params`）
- `predict` 返回 `(prediction DataFrame, None)`，批量推理，NaN-safe
- 演示版 Notebook：`Notebooks/2_metabolomic_age.ipynb`（含时钟散点图，可选 matplotlib）
- 对外描述时以"Metabolomic Age / aging clock"为主，死亡风险是辅助输出

## 5. Verification / 验证

- `prediction` contains all 6 columns; row count equals input samples
- Passing on fake data verifies the pipeline (values are meaningless, only the
  end-to-end path is checked)

- `prediction` 含全部 6 列，行数 = 输入样本数
- 假数据上跑通即视为验证通过（数值无实际意义，只验链路）
