# MetAgeFormer 环境自动安装指南（供 AI Agents 阅读）

> 本文件面向 AI 代理（Claude Code、Codex 等）编写。目标：在没有人工干预的情况下，
> 自动完成 MetAgeFormer 运行环境的安装与自检，并严格遵守仓库约定。

---

## 1. 前置条件检查

按顺序检查，任一不满足则停下并报告：

1. `conda --version` 可用（miniconda 或 anaconda）
2. 主流操作系统均可（Linux / macOS / Windows，x86_64 或 arm64）
3. GPU 版仅限 NVIDIA GPU，且驱动支持 CUDA 12.1（`nvidia-smi` 能显示 GPU 即基本满足）；
   Apple Silicon / 无 NVIDIA GPU 的机器使用 CPU 版
4. 磁盘可用空间 ≥ 10 GB（环境 + 模型权重 + 缓存）
5. 网络可访问 conda 频道（pytorch、conda-forge）与 PyPI

## 2. 克隆仓库（权重走 Git LFS）

```bash
git lfs install          # 确认 git-lfs 可用；不可用则: conda install -c conda-forge git-lfs
git clone https://github.com/ericcombiolab/MetAgeFormer
cd MetAgeFormer
```

克隆完成后校验权重已下载（应为真实大小而非 133 字节指针）：

```bash
ls -lh Model_Weights/MetAgeFormer/model_weights.pth   # 约 74M
ls -lh Model_Weights/Lightweight/model_weights.pth    # 约 17M
```

## 3. 安装环境

```bash
conda env create -f environment_cpu.yml    # CPU 版（通用）
# conda env create -f environment_gpu.yml  # GPU 版（仅 NVIDIA + CUDA 12.1）

conda activate metageformer                # 环境名固定为 metageformer
```

- conda 解析慢时改用 `mamba env create -f environment_cpu.yml`
- 依赖集：torch 2.3.0、numpy<2、pandas、anndata + pip(einops、tqdm、torchsurv、permetrics)。
  **不要为了兼容旧脚本额外安装其他包**——仓库代码只用这些依赖；
  `wandb` 可选（仅训练日志，默认关闭）。
- 若确需新增依赖，必须先确认被 `Src/` 或 `Notebooks/` 代码实际 import，再同步修改 yml。

## 4. 自动自检

### 4.1 依赖导入检查

```bash
python -c "
import torch, numpy, pandas, anndata, einops, torchsurv, permetrics
print('deps OK, torch', torch.__version__)
"
```

### 4.2 权重完整性检查

```bash
python -c "
import sys; sys.path.insert(0, 'Src')
from utils import load_tokenizer
tok = load_tokenizer('Model_Weights/MetAgeFormer/tokenizer.pkl')
assert tok.vocab_size_identifiers == 107
print('weights + tokenizer OK')
"
```

### 4.3 假数据冒烟测试（无真实数据权限也可跑）

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

## 5. 仓库约定（必须遵守）

| 规则 | 说明 |
|---|---|
| 运行位置 | 脚本在 `Src/` 下运行：`cd Src && export PYTHONPATH=.` |
| Notebook | 在仓库根目录或 `Notebooks/` 内启动均可（路径自动解析） |
| 数据 | 真实队列数据受限，不随仓库分发（见 `Data/readme.md`）；假数据用 `Data/generate_fake_data.py` 生成（规则见 `Docs/SKILL_FAKE_DATA.md`） |
| 训练输出 | 一律写入 `artifacts/`（git-ignored），demo 用 `artifacts/demo/...` |
| 权重 | `Model_Weights/*.pth` 走 Git LFS，不要直接 `git add` 改动后的 pth（先确认 .gitattributes 规则存在） |
| wandb | 默认关闭（`--use_wandb false` / `"wandb_monitor": false`），用户明确要求才开 |
| 修改范围 | 只改本仓库内文件；改代码前先读对应 SKILL 文档 |

## 6. 各功能入口

- 提取 embeddings → `Docs/SKILL_EMBEDDINGS.md`（Notebook 1）
- 代谢年龄（DeepGompertz）→ `Docs/SKILL_AGING_CLOCK.md`（Notebook 2）
- 代谢亚型分型 → `Docs/SKILL_SUBTYPES.md`（Notebook 3）
- 轻量血液面板模型 → `Docs/SKILL_LIGHTWEIGHT.md`（Notebook 4）
- 训练管线（pretrain / finetune / distillation / ablation）→ `Docs/SKILL_TRAINING.md`

## 7. 故障排查

- **clone 后权重是 133 字节指针文件**：git-lfs 未安装或未 `git lfs install`，装好后 `git lfs pull`
- **tokenizer.pkl 反序列化报 ModuleNotFoundError**：`Src/` 不在 sys.path；且 `metageformer_torch`
  包名不可改（pickle 按引用序列化）
- **conda 卡在 solving environment**：用 mamba，或删掉 `defaults` 频道后重试
- **CPU 上训练 demo 很慢**：属正常（demo 设计为 ~1 分钟级），真实训练建议 GPU
