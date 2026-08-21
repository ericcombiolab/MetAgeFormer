# Skill: 提取 Metabolomic Embeddings（供 AI Agents 使用）/ Embedding Extraction (for AI Agents)

> 中文：用户想用 MetAgeFormer 预训练 backbone 把 NMR 代谢组样本编码为 512 维
> metabolomic embedding 时，按本 Skill 操作（Notebook 1 的流程）。
>
> English: Use this skill when the user wants to encode NMR metabolomics samples into
> 512-dim metabolomic embeddings with the pretrained MetAgeFormer backbone
> (Notebook 1 workflow).

## 1. 输入要求 / Input Requirements

- 权重：`Model_Weights/MetAgeFormer/`（`config.json`、`tokenizer.pkl`、`model_weights.pth`）
- 数据：AnnData `.h5ad`，需有 `layers['Z-score normalized']`，且 `var_names` 都在
  tokenizer 词汇表内（107 个指标，见 `Model_Weights/MetAgeFormer/vocab.txt`）
- 没有真实数据：先用 `Data/generate_fake_data.py --synthetic` 生成假数据

## 2. 核心流程 / Core Workflow

```python
import json, sys
from pathlib import Path
REPO_ROOT = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "Src"))

import torch, anndata as ad
from utils import load_tokenizer
from metageformer_torch.models import MetAgeFormer_Pretrained

tokenizer = load_tokenizer(str(REPO_ROOT / "Model_Weights/MetAgeFormer/tokenizer.pkl"))
with open(REPO_ROOT / "Model_Weights/MetAgeFormer/config.json") as f:
    model_config = json.load(f)

embedding_module_conf = {"n_vocabs": {"identifier": tokenizer.vocab_size_identifiers}}
model = MetAgeFormer_Pretrained(embedding_module_conf, model_config,
                                str(REPO_ROOT / "Model_Weights/MetAgeFormer/model_weights.pth"))
model.eval()

adata = ad.read_h5ad("Data/fake/NMR_dataset_fake/val.h5ad")
inputs, _ = tokenizer.tokenize_from_anndata(
    adata[:64], padding="longest", masking="missing",
    data_layer="Z-score normalized", mode="inference", return_tensor=True, device="cpu")

with torch.inference_mode():
    outputs = model(inputs)
embs = outputs["embs"].cpu().numpy()   # (n_samples, 512)
```

## 3. 关键点 / Key Points

- `masking='missing'`：缺失测量值用学习到的 mask token，**不要零填充**（NaN-aware）
- `tokenizer.pkl` 是按引用 pickle 的 `metageformer_torch.tokenizer.MetAgeFormer_Tokenizer`，
  加载时必须把 `Src/` 放在 sys.path，包名/类名不可改
- 输出键：`embs`（512 维 embedding）、`logit_conc`（浓度重建）、`attn`、`metabolite embs`
- 演示版 Notebook：`Notebooks/1_embedding_extraction.ipynb`（含 PCA 可视化，可选 matplotlib）

## 4. 验证 / Verification

- `outputs["embs"].shape == (n_samples, 512)`
- 假数据上冒烟：见 `Docs/AGENT_SETUP_CN.md` §4.3（或 EN 版 §4.3）
