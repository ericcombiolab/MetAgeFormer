# Skill: Metabolic Subtype Assignment (for AI Agents) / 代谢亚型分型（供 AI Agents 使用）

> English: Use this skill when the user wants to assign **metabolic subtypes**
> (Leiden clusters 0–12) and **meta-subtypes** (1–4) to samples from their
> metabolomic embeddings (Notebook 3 workflow).
>
> 中文：用户想从 metabolomic embeddings 给样本分配**代谢亚型**（Leiden 聚类 0–12）和
> **元亚型**（meta-subtype 1–4）时，按本 Skill 操作（Notebook 3 的流程）。

## 1. Model / 模型

- `Model_Weights/SubtypeClassifier/subtype_mlp_classifier_focal.joblib` (303KB) —
  focal-loss MLP classifier (512 → (128, 64) → 13), trained in the released backbone's
  embedding space
- **Note: the `.joblib` extension is misleading — the file is a PyTorch `torch.save`
  archive.** Load it with `FocalMLPClassifier.load()`, never `joblib.load()`.
- Class definition: `Src/metageformer_torch/subtype_mlp.py` (inference needs only
  torch + numpy; `fit()` additionally needs scikit-learn)

- `Model_Weights/SubtypeClassifier/subtype_mlp_classifier_focal.joblib`（303KB）
  — focal-loss MLP 分类器（512 → (128, 64) → 13，训练于发布的 backbone embedding 空间）
- **注意：文件扩展名是 `.joblib`，但内容是 PyTorch `torch.save` 归档**，
  必须用 `FocalMLPClassifier.load()` 加载，禁止 `joblib.load()`
- 类定义：`Src/metageformer_torch/subtype_mlp.py`（推理仅需 torch + numpy；
  `fit()` 需要 scikit-learn）

## 2. Input Requirements / 输入要求

- Input: **raw** 512-dim embeddings (float32, **no normalization/scaling**), produced by
  the released backbone (`Model_Weights/MetAgeFormer/`)
- No restricted data: fake data → backbone → embeddings (Notebook 1 flow)

- 输入：**原始** 512 维 embeddings（float32，**不做任何归一化/缩放**），
  必须来自发布的 backbone（`Model_Weights/MetAgeFormer/`）
- 无真实数据：假数据 → backbone → embeddings（同 Notebook 1 流程）

## 3. Core Workflow / 核心流程

```python
import sys
from pathlib import Path
REPO_ROOT = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "Src"))

import numpy as np
from metageformer_torch.subtype_mlp import FocalMLPClassifier

classifier = FocalMLPClassifier.load(
    str(REPO_ROOT / "Model_Weights/SubtypeClassifier/subtype_mlp_classifier_focal.joblib"),
    device="cpu")

embs = np.load("embeddings.npy")          # (n, 512) float32, raw backbone embeddings
subtype = classifier.predict(embs)        # int 0-12
proba = classifier.predict_proba(embs)    # (n, 13), columns = clusters 0..12
```

## 4. Meta-subtype Mapping / 元亚型映射（paper's fixed mapping, first match wins / 论文用固定映射，先匹配先得）

```python
META_SUBTYPES = {
    "Meta-subtype 1": [8],
    "Meta-subtype 2": [0, 5, 2, 3],
    "Meta-subtype 3": [6, 9, 12, 7, 10],
    "Meta-subtype 4": [4, 1, 11],
}
```

## 5. Key Points / 关键点

- The 13 cluster labels (0–12) come from Leiden (n_neighbors=15, resolution=1.0) on the
  training embeddings; `predict` returns cluster ids directly, **no renaming**
- Confidence = `proba.max(axis=1)`
- UMAP artifacts (`UMAP_reducer.pkl`, 3.5GB) are **not distributed** — visualization only
- Demo notebook: `Notebooks/3_metabolic_subtypes.ipynb`

- 13 个聚类标签（0–12）由 Leiden（n_neighbors=15, resolution=1.0）在训练嵌入上产生；
  `predict` 直接输出聚类编号，**不做重命名**
- 置信度取 `proba.max(axis=1)`
- UMAP 可视化文件（`UMAP_reducer.pkl` 3.5GB）**不随仓库分发**，仅为可视化
- 演示版 Notebook：`Notebooks/3_metabolic_subtypes.ipynb`

## 6. Verification / 验证

- `classifier.classes_ == [0..12]`; `proba.shape == (n, 13)` with rows summing to 1
- Passing on fake-data embeddings verifies the pipeline (values meaningless, path checked)

- `classifier.classes_ == [0..12]`；`proba.shape == (n, 13)` 且每行和为 1
- 假数据 embeddings 上跑通即视为验证通过（数值无实际意义，只验链路）
