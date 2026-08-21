# Skill: Fake Data Generation (for AI Agents) / 假数据生成（供 AI Agents 使用）

> English: Use this skill whenever the user needs demo data, smoke-test data, or wants to
> run the pipelines without restricted cohort data. Fake data keeps only the **schema** of
> real data; all values are random and no real measurements are ever copied.
>
> 中文：当用户需要演示数据、冒烟测试数据、或想在不使用受限队列数据的情况下跑通管线时，
> 按本 Skill 生成**假数据**。假数据只保留真实数据的**结构**（schema），数值全部随机，
> 绝不复制任何真实测量值。

---

## 1. When to Use / 何时使用

- User asks for demo/example/fake/synthetic data
- Need to verify pipelines or notebooks run without data access
- Prepare data for new users running notebooks or training demos

- 用户要求生成演示/示例数据（demo/example/fake/synthetic data）
- 需要验证 pipeline / notebook 在无数据权限的情况下可运行
- 为新用户准备 Notebook 或训练 demo 所需的数据

## 2. How to Generate / 生成方式

Tool script: `Data/generate_fake_data.py` (deps: numpy/pandas/anndata only — do not add
scanpy/sklearn). Two modes:

工具脚本：`Data/generate_fake_data.py`（依赖仅 numpy/pandas/anndata，不要引入
scanpy/sklearn）。两种模式：

### Mode A / 模式 A: mirror a real h5ad's structure (randomized values)

```bash
python Data/generate_fake_data.py --from <real.h5ad> --n_samples 1000
# output: <same dir>/fake_<name>.h5ad (--out_dir redirects)
```

Rules / 规则:
- Each feature randomized separately: sample from the original mean±std, clipped to
  original min/max
- `var`/`obs` schema preserved; numeric obs columns resampled from the original
  distribution, categorical columns sampled from original values
- `layers['Z-score normalized']`: if `var` has `Z-score mean`/`Z-score std` columns,
  recompute z-scores from them; otherwise randomize per feature from the original layer
- `obsm` keys preserved (e.g. 512-dim `metabolomic embedding`), regenerated as
  zero-mean unit-variance random embeddings
- Files get a `fake_` prefix

### Mode B / 模式 B: synthetic from scratch (no data access needed)

```bash
python Data/generate_fake_data.py --synthetic --outdir Data/fake
```

Creates three dataset families (train/val/test splits) / 生成三套数据（train/val/test 划分）:

| Directory | Contents | Serves |
|---|---|---|
| `Data/fake/NMR_dataset_fake/` | 107 measures + Z-score layer + obs age | pretrain demo, Notebooks 1/2 |
| `Data/fake/deep_gompertz_fake/` | 512-dim embeddings + Gompertz survival obs | DeepGompertz finetune demo |
| `Data/fake/Blood_dataset_fake/` | 14 blood features + layer + embeddings + survival obs | distillation demo |

The 107 NMR feature names come from `Model_Weights/MetAgeFormer/vocab.txt` (so the fake
data is directly tokenizable; falls back to positional `NMR_%03d` names if the file is
missing). / 107 个 NMR 指标名取自 `Model_Weights/MetAgeFormer/vocab.txt`（保证可被
tokenizer 直接分词；文件缺失时回退为位置编号 `NMR_%03d`）。

## 3. Notes / 注意事项

- `Data/fake/` is git-ignored and never committed / `Data/fake/` 已被 `.gitignore` 忽略，不会入库
- Fake data must NEVER be used for real analysis or paper conclusions (the script prints
  this warning) / 假数据**严禁**用于任何真实分析/论文结论，脚本末尾也会打印此警告
- Real-data layout: `Data/readme.md` / 真实数据布局见 `Data/readme.md`
