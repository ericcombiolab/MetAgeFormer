# Skill: Custom Cohort → Model Input (for AI Agents) / 私有队列 → 模型输入（供 AI Agents 使用）

> English: Use this skill when the user has their own (private) cohort — NMR metabolomics
> or blood-test panel data — and wants to convert it into the model-input format using
> the published UKB-derived normalization factors. No restricted-data access is needed;
> everything this skill needs is already in this repository.
>
> 中文：当用户拥有自己的（私有）队列——NMR 代谢组或血液检验面板数据——并希望用已发布的
> UKB 派生归一化因子将其转换为模型输入格式时使用本 Skill。无需任何受限数据权限；
> 本 Skill 所需的一切均已包含在仓库中。

---

## 1. When to Use / 何时使用

- User has private cohort measurements and wants to run the released MetAgeFormer /
  DeepGompertz / Lightweight models on them
- User asks "how do I format my own NMR / blood data for your model"
- User's data is NOT UKB/ADNI/CHARLS raw files (for those, use
  `Docs/SKILL_DATA_PREPROCESSING.md`)

- 用户有私有队列测量值，希望在已发布的 MetAgeFormer / DeepGompertz / Lightweight 模型上运行
- 用户询问"如何把我自己的 NMR / 血液数据整理成你们的模型输入"
- 用户数据不是 UKB/ADNI/CHARLS 原始文件（那些场景用 `Docs/SKILL_DATA_PREPROCESSING.md`）

## 2. Privacy / 隐私

This workflow is one-directional: the user's own data stays on their machine; the repo
publishes only aggregate statistics (per-feature mean/std/unit). Never commit user data
files, subject IDs, or individual values to the repository.
该流程单向：用户自有数据留在本机；仓库只发布聚合统计量（逐特征 mean/std/unit）。绝不将用户数据文件、受试者 ID 或个体值提交进仓库。

## 3. The UKB Normalization Factors / UKB 归一化因子

Published in `Data/ukb_norm_factors/` (see its `readme.md` for provenance) / 发布在
`Data/ukb_norm_factors/`（口径与来源见该目录 `readme.md`）:

| File | Panel | Columns |
|---|---|---|
| `nmr107_factors.csv` | 107 NMR features, order = `Model_Weights/MetAgeFormer/vocab.txt` (token order) | `feature, display_name, unit, z_mean, z_std` |
| `blood14_factors.csv` | 14 blood features (Lightweight panel), row order is load-bearing | `feature, unit, z_mean, z_std` |

Why transferable / 为什么可以迁移: the models were trained on UKB z-scores
(`z = (x − mean)/std`); applying the same transformation anchors any new cohort to the
model's training distribution, provided the assays report the same units. NMR means/std
use pandas sample std (ddof=1); blood means/std come from a sklearn `StandardScaler`
(population std, ddof=0) — both fitted on the UKB **train split only**.
模型在 UKB z-score（`z = (x − mean)/std`）上训练；对任何新队列应用同一变换，可把新数据锚定到模型训练分布（前提是检测报告单位一致）。NMR 的 mean/std 用 pandas 样本标准差（ddof=1）；blood 的 mean/std 来自 sklearn `StandardScaler`（总体标准差 ddof=0）——两者都**仅拟合 UKB train 划分**。

## 4. NMR 107 Mode / NMR 107 模式

Requirements / 要求:

- A matrix with up to 107 features nameable to `Model_Weights/MetAgeFormer/vocab.txt`
  (the helper matches exact → case-insensitive → punctuation-insensitive; unmatched
  input columns are dropped with a warning) / 最多 107 个特征、且特征名可对齐到
  `Model_Weights/MetAgeFormer/vocab.txt`（工具按 精确 → 大小写不敏感 → 标点不敏感 匹配；未匹配的输入列会被丢弃并给出警告）
- **Units must be mmol/L** (particle-size diameters in nm; ApoB/ApoA1/Albumin in g/L;
  Unsaturation in degree) — the helper prints the expected unit per feature but does
  not convert / **单位必须是 mmol/L**（粒径为 nm；ApoB/ApoA1/Albumin 为 g/L；不饱和度为 degree）——工具会打印每个特征的期望单位，但不做换算
- Missing measurements = NaN/empty. **Never zero-fill** — NaN tokens are masked at
  inference / 缺失测量 = NaN/空。**严禁零填充**——NaN token 在推理时被 mask

CSV input convention: features as rows (a `feature` column or the index), samples as
columns / CSV 输入约定：特征为行（`feature` 列或索引），样本为列:

```bash
# real data: python Data/apply_ukb_norm.py --input /your/cohort.csv --mode nmr107 --out my_cohort_z.csv
```

h5ad mode (needs anndata; writes the full 107 panel + `'Z-score normalized'` layer to a
NEW file — the input is never modified) / h5ad 模式（需要 anndata；向**新文件**写入完整
107 面板 + `'Z-score normalized'` layer——绝不修改输入文件）:

```bash
# demo (smoke-testable): rebuild the fake NMR h5ad into canonical model-input format
python Data/apply_ukb_norm.py --mode nmr107 \
    --h5ad Data/fake/NMR_dataset_fake/val.h5ad --h5ad-out my_cohort_z.h5ad
# real data: same command with your own .h5ad
```

Missing features (not measured in your cohort) become NaN columns automatically in
h5ad mode (CSV mode: pass `--fill-missing`).
你的队列中未测的特征在 h5ad 模式下自动成为 NaN 列（CSV 模式加 `--fill-missing`）。

The output h5ad matches the canonical schema (see `Docs/SKILL_DATA_PREPROCESSING.md`
§3): `X` raw, `var` factor columns, layer `'Z-score normalized'`. From there run the
notebooks (`Notebooks/1_embedding_extraction.ipynb`) or the eval scripts.
输出 h5ad 即规范 schema（见 `Docs/SKILL_DATA_PREPROCESSING.md` §3）：`X` 原始值、`var` 因子列、
layer `'Z-score normalized'`。之后即可跑 notebooks（`Notebooks/1_embedding_extraction.ipynb`）或评估脚本。

## 5. Blood 14 Mode / Blood 14 模式

The 14-feature Lightweight panel and expected units / 14 特征 Lightweight 面板与期望单位:

| Feature | Unit |
|---|---|
| White blood cell (leukocyte) count | 10^9 cells/L |
| Haemoglobin concentration | g/dL |
| Haematocrit percentage | % |
| Mean corpuscular volume | fL |
| Platelet count | 10^9 cells/L |
| C-reactive protein | mg/L |
| Glycated haemoglobin (HbA1c) | mmol/mol |
| Cholesterol | mmol/L |
| HDL cholesterol | mmol/L |
| Triglycerides | mmol/L |
| Creatinine | µmol/L |
| Glucose | mmol/L |
| Urate | µmol/L |
| Cystatin C | mg/L |

Common unit conversions for non-UKB lab reports (same recipes as the CHARLS pipeline)
/ 非 UKB 化验报告的常见单位换算（与 CHARLS 管线相同的配方）:

| Measure | Conversion |
|---|---|
| HbA1c | NGSP % → IFCC mmol/mol: `(pct − 2.15) / 0.0915` (inverse: `mmol × 0.0915 + 2.15`) |
| Glucose | mg/dL ÷ 18 |
| Triglycerides | mg/dL ÷ 88.57 |
| Creatinine | mg/dL ÷ 11.312, then × 1000 (→ µmol/L) |
| Cholesterol, HDL | mg/dL ÷ 38.6654 |
| Urate | mg/dL ÷ 16.81, then × 1000 (→ µmol/L) |

Apply with the helper / 用工具应用:

```bash
# demo
python Data/apply_ukb_norm.py --input my_blood.csv --mode blood14 --out my_blood_z.csv --fill-missing
# real data: python Data/apply_ukb_norm.py --input /your/blood.csv --mode blood14 --out my_blood_z.csv --fill-missing
```

## 6. Verification Checklist / 验证清单

- Spot-check the arithmetic: `z == (x − z_mean) / z_std` on a few cells
  / 抽查算术：若干单元格上 `z == (x − z_mean) / z_std`
- NaN positions preserved from input; missing panel features are NaN columns, never 0
  / 输入 NaN 位置保留；面板中缺失特征为 NaN 列，绝不填 0
- Column order = factor order (`vocab.txt` for nmr107) / 列顺序 = 因子顺序（nmr107 即 `vocab.txt`）
- h5ad mode: layer named exactly `'Z-score normalized'`, float32 / h5ad 模式：layer 名精确为 `'Z-score normalized'`，float32
- Tokenizer smoke test (proves end-to-end compatibility) / tokenizer 冒烟测试（证明端到端兼容）:

```python
# run from Src/ (export PYTHONPATH=.)
import pickle, anndata as ad
with open('../Model_Weights/MetAgeFormer/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)
adata = ad.read_h5ad('my_cohort_z.h5ad')
out = tok.tokenize_from_anndata(adata, data_layer='Z-score normalized',
                                masking='missing', mode='inference', return_tensor=True)
# out[0]['input_ids'] -> {'identifier': (n, 107) long, 'concentration': (n, 107) float32}
```

## 7. Notes / 注意事项

- Never modify the factor files in `Data/ukb_norm_factors/` — they define the model's
  training distribution / 绝不修改 `Data/ukb_norm_factors/` 中的因子文件——它们定义了模型的训练分布
- If your cohort lacks features, they become masked NaN tokens (same mechanism as the
  ADNI Q300 overlap panel, `Docs/SKILL_DATA_PREPROCESSING.md` §6.3) / 队列缺失的特征会成为
  masked NaN token（与 ADNI Q300 重叠面板同一机制）
- Batch effects: the factors assume the same assay platform and preprocessing as UKB.
  Cross-platform data may need harmonization before z-scoring (out of scope here)
  / 批次效应：因子假设检测平台与预处理与 UKB 一致；跨平台数据可能需先做协调对齐（超出本 Skill 范围）
- Do not commit your data; keep it outside the repo or under a git-ignored directory
  / 不提交你的数据；置于仓库之外或 git 忽略目录
