# MetAgeFormer Environment Auto-Setup Guide (for AI Agents)

> This document is written for AI agents (Claude Code, Codex, etc.). Goal: install and
> self-verify the MetAgeFormer runtime environment fully autonomously, and strictly
> follow the repository conventions below.

---

## 1. Prerequisites

Check in order; stop and report if any fails:

1. `conda --version` works (miniconda or anaconda)
2. Any mainstream OS (Linux / macOS / Windows, x86_64 or arm64)
3. The GPU build requires an NVIDIA GPU whose driver supports CUDA 12.1
   (`nvidia-smi` listing a GPU is usually sufficient); Apple Silicon and machines
   without an NVIDIA GPU use the CPU build
4. ≥ 10 GB free disk space (environment + model weights + caches)
5. Network access to the conda channels (pytorch, conda-forge) and PyPI

## 2. Clone the Repository (weights via Git LFS)

```bash
git lfs install          # if unavailable: conda install -c conda-forge git-lfs
git clone https://github.com/ericcombiolab/MetAgeFormer
cd MetAgeFormer
```

Verify the weights downloaded as real files (not 133-byte pointers):

```bash
ls -lh Model_Weights/MetAgeFormer/model_weights.pth   # ~74M
ls -lh Model_Weights/Lightweight/model_weights.pth    # ~17M
```

## 3. Install the Environment

```bash
conda env create -f environment_cpu.yml    # CPU build (universal)
# conda env create -f environment_gpu.yml  # GPU build (NVIDIA + CUDA 12.1 only)

conda activate metageformer                # env name is fixed: metageformer
```

- If conda solving is slow, use `mamba env create -f environment_cpu.yml`.
- Dependency set: torch 2.3.0, numpy<2, pandas, anndata + pip (einops, tqdm, torchsurv,
  permetrics). **Do not install extra packages for legacy scripts** — the repository
  only uses these; `wandb` is optional (training logging only, off by default).
- If a new dependency is truly needed, confirm it is imported by code under `Src/`
  or `Notebooks/` first, then update the yml files accordingly.

## 4. Automated Self-Check

### 4.1 Dependency imports

```bash
python -c "
import torch, numpy, pandas, anndata, einops, torchsurv, permetrics
print('deps OK, torch', torch.__version__)
"
```

### 4.2 Weight integrity

```bash
python -c "
import sys; sys.path.insert(0, 'Src')
from utils import load_tokenizer
tok = load_tokenizer('Model_Weights/MetAgeFormer/tokenizer.pkl')
assert tok.vocab_size_identifiers == 107
print('weights + tokenizer OK')
"
```

### 4.3 Fake-data smoke test (no restricted data required)

```bash
python Data/generate_fake_data.py --synthetic --outdir Data/fake
python -c "
import sys; sys.path.insert(0, 'Src')
import json, torch, anndata as ad
from utils import load_tokenizer
from metageformer_torch.models import MetAgeFormer_Pretrained
tok = load_tokenizer('Model_Weights/MetAgeFormer/tokenizer.pkl')
cfg = json.load(open('Model_Weights/MetAgeFormer/config.json'))
m = MetAgeFormer_Pretrained({'n_vocabs': {'identifier': tok.vocab_size_identifiers}}, cfg, 'Model_Weights/MetAgeFormer/model_weights.pth')
m.eval()
adata = ad.read_h5ad('Data/fake/NMR_dataset_fake/val.h5ad')[:8]
inputs, _ = tok.tokenize_from_anndata(adata, padding='longest', masking='missing',
    data_layer='Z-score normalized', mode='inference', return_tensor=True, device='cpu')
with torch.inference_mode():
    out = m(inputs)
assert out['embs'].shape[1] == cfg['d_model']
print('smoke test OK:', tuple(out['embs'].shape))
"
```

## 5. Repository Conventions (must follow)

| Rule | Detail |
|---|---|
| Run location | Scripts run from `Src/`: `cd Src && export PYTHONPATH=.` |
| Notebooks | Work from the repo root or inside `Notebooks/` (paths auto-resolve) |
| Data | Real cohort data is restricted and not distributed (see `Data/readme.md`); fake data via `Data/generate_fake_data.py` (rules in `Docs/SKILL_FAKE_DATA.md`) |
| Training outputs | Always under `artifacts/` (git-ignored); demo runs use `artifacts/demo/...` |
| Weights | `Model_Weights/*.pth` are Git LFS-tracked; do not plain `git add` modified .pth files (check `.gitattributes` rules exist) |
| wandb | Off by default (`--use_wandb false` / `"wandb_monitor": false`); enable only when the user asks |
| Scope | Only touch files in this repository; read the relevant SKILL doc before changing code |

## 6. Capability Entry Points

- Extract embeddings → `Docs/SKILL_EMBEDDINGS.md` (Notebook 1)
- Metabolomic age (DeepGompertz) → `Docs/SKILL_AGING_CLOCK.md` (Notebook 2)
- Metabolic subtype assignment → `Docs/SKILL_SUBTYPES.md` (Notebook 3)
- Lightweight blood-panel model → `Docs/SKILL_LIGHTWEIGHT.md` (Notebook 4)
- Training pipelines (pretrain / finetune / distillation / ablation) → `Docs/SKILL_TRAINING.md`
- Cohort data preprocessing (UKB / CHARLS / ADNI → model input) → `Docs/SKILL_DATA_PREPROCESSING.md`
- Private cohort → model input (NMR / blood, UKB normalization factors) → `Docs/SKILL_CUSTOM_COHORT.md`

## 7. Troubleshooting

- **Weights are 133-byte pointers after clone**: git-lfs missing — install it and run `git lfs pull`
- **tokenizer.pkl unpickle fails with ModuleNotFoundError**: `Src/` is not on sys.path;
  also the package name `metageformer_torch` must never change (pickle-by-reference)
- **conda stuck on "solving environment"**: use mamba, or drop the `defaults` channel and retry
- **Training demos slow on CPU**: expected (demos are ~1-minute scale); real training needs a GPU
