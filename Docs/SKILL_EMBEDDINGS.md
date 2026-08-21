# Skill: Embedding Extraction (for AI Agents) / 提取 Metabolomic Embeddings（供 AI Agents 使用）

> English: Use this skill when the user wants to encode NMR metabolomics samples into
> 512-dim metabolomic embeddings with the pretrained MetAgeFormer backbone
> (Notebook 1 workflow).
>
> 中文：用户想用 MetAgeFormer 预训练 backbone 把 NMR 代谢组样本编码为 512 维
> metabolomic embedding 时，按本 Skill 操作（Notebook 1 的流程）。

## 1. Input Requirements / 输入要求

- Weights: `Model_Weights/MetAgeFormer/` (`config.json`, `tokenizer.pkl`, `model_weights.pth`)
- Data: AnnData `.h5ad` with `layers['Z-score normalized']` and `var_names` all in the
  tokenizer vocabulary (107 measures, see `Model_Weights/MetAgeFormer/vocab.txt`)
- No restricted data? Run `Data/generate_fake_data.py --synthetic` first

- 权重：`Model_Weights/MetAgeFormer/`（`config.json`、`tokenizer.pkl`、`model_weights.pth`）
- 数据：AnnData `.h5ad`，需有 `layers['Z-score normalized']`，且 `var_names` 都在
  tokenizer 词汇表内（107 个指标，见 `Model_Weights/MetAgeFormer/vocab.txt`）
- 没有真实数据：先用 `Data/generate_fake_data.py --synthetic` 生成假数据

## 2. Core Workflow / 核心流程

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

## 3. Key Points / 关键点

- `masking='missing'`: missing measurements use the learned mask token — **do not
  zero-fill** (NaN-aware)
- `tokenizer.pkl` is a pickle-by-reference of
  `metageformer_torch.tokenizer.MetAgeFormer_Tokenizer`; `Src/` must be on sys.path and
  the package/class names must never change
- Output keys: `embs` (512-dim embedding), `logit_conc` (concentration reconstruction),
  `attn`, `metabolite embs`
- Demo notebook: `Notebooks/1_embedding_extraction.ipynb` (optional PCA plot, matplotlib)

- `masking='missing'`：缺失测量值用学习到的 mask token，**不要零填充**（NaN-aware）
- `tokenizer.pkl` 是按引用 pickle 的 `metageformer_torch.tokenizer.MetAgeFormer_Tokenizer`，
  加载时必须把 `Src/` 放在 sys.path，包名/类名不可改
- 输出键：`embs`（512 维 embedding）、`logit_conc`（浓度重建）、`attn`、`metabolite embs`
- 演示版 Notebook：`Notebooks/1_embedding_extraction.ipynb`（含 PCA 可视化，可选 matplotlib）

## 4. Verification / 验证

- `outputs["embs"].shape == (n_samples, 512)`
- Fake-data smoke test: `Docs/AGENT_SETUP_EN.md` §4.3 (or CN §4.3)

- `outputs["embs"].shape == (n_samples, 512)`
- 假数据上冒烟：见 `Docs/AGENT_SETUP_CN.md` §4.3（或 EN 版 §4.3）
