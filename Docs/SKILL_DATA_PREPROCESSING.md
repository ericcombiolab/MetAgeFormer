# Skill: Cohort Data Preprocessing — UKB / CHARLS / ADNI (for AI Agents) / 队列数据预处理——UKB / CHARLS / ADNI（供 AI Agents 使用）

> English: Use this skill when the user has access to UK Biobank (UKB), CHARLS, or ADNI
> raw data and wants to convert it into the model-input `.h5ad` format used by the
> released MetAgeFormer models (this reproduces the paper's preprocessing exactly).
> This skill documents every parameter needed; it NEVER contains individual-level data.
>
> 中文：当用户拥有 UK Biobank（UKB）、CHARLS 或 ADNI 原始数据访问权限、需要将其转换为
> 已发布 MetAgeFormer 模型所读取的 `.h5ad` 输入格式时使用本 Skill（完全复现论文的预处理流程）。
> 本 Skill 记录了全部所需参数，**绝不包含任何个体数据**。

---

## 1. When to Use / 何时使用

- User has real UKB / CHARLS / ADNI raw files and wants the canonical model-input h5ad
- User wants to reproduce the paper's preprocessing (splits, technical-variation removal,
  z-score conventions) on their own copy of the data
- User asks how real cohort data becomes `Data/NMR_dataset_fullcohort_107nonderived/`,
  `Data/Blood_dataset_fullcohort_107nonderived_mlm/`, `Data/ADNI_datasets/`,
  `Data/Charls_datasets/` (see `Data/readme.md`)

- 用户有真实的 UKB / CHARLS / ADNI 原始文件，需要生成规范模型输入 h5ad
- 用户希望在自有数据副本上复现论文的预处理（数据划分、技术变异去除、z-score 口径）
- 用户询问真实队列数据如何变成 `Data/NMR_dataset_fullcohort_107nonderived/`、
  `Data/Blood_dataset_fullcohort_107nonderived_mlm/`、`Data/ADNI_datasets/`、
  `Data/Charls_datasets/`（见 `Data/readme.md`）

No data access? Use `Docs/SKILL_FAKE_DATA.md` instead.
无数据权限？改用 `Docs/SKILL_FAKE_DATA.md`。

## 2. Privacy & Publishability Rules / 隐私与可发布性规则

Hard rules (violations must be reported, never committed) / 硬性规则（违反必须报告，绝不提交）:

- **Never** copy individual values, rows, sample IDs, or participant IDs into the repo
  / **严禁**将任何个体测量值、数据行、样本 ID 或受试者 ID 写入仓库
- **Never** publish per-feature min/max or quantiles — extreme values can identify
  individuals / **严禁**发布逐特征的 min/max 或分位数——极端值可能识别个体
- Safe to publish: per-feature mean/std/units (already shipped in `Data/ukb_norm_factors/`),
  processing parameters (this document), aggregate counts / 可安全发布：逐特征
  mean/std/单位（已随仓库发布在 `Data/ukb_norm_factors/`）、处理参数（本文档）、聚合计数

Self-check before committing / 提交前自检:

```bash
grep -rn "/datahome\|eid_\|_[0-9]\{6,\}" Docs/ Data/ --include="*.md" --include="*.py" | grep -v ukb_norm_factors
# must return nothing relevant; any real data path or ID-like token is a leak
```

Real cohort data lives ONLY on the user's local/remote analysis machines, never in the
repo. Outputs go under `artifacts/` or user-chosen directories (git-ignored).
真实队列数据只存在于用户本地的分析环境中，绝不进入仓库。产出放 `artifacts/` 或用户自选目录（已被 git 忽略）。

## 3. Target Schema / 目标格式

All pipelines produce one AnnData h5ad per cohort split / 每个队列生成一个 AnnData h5ad:

| Component | NMR (107) / Blood (14) | Notes |
|---|---|---|
| `X` | raw concentrations | raw assay scale, **NaN = missing** (never zero-fill) |
| `var_names` | `Model_Weights/MetAgeFormer/vocab.txt` order (NMR); the 14 blood names in `Data/ukb_norm_factors/blood14_factors.csv` order | order is load-bearing |
| `var` | `Z-score mean`, `Z-score std` (NMR also `UKB NMR symbol`, `Unit`) | fit on UKB train only, see §4.6 |
| `layers['Z-score normalized']` | `(X − mean)/std`, float32 | the ONLY layer the models read (hardcoded everywhere) |
| `obs` | `Age at assessment (estimated)`, `Death event`, `Death event time`, `Sex`, ... | CHARLS uses `Sex`, `Chronological Age`, `BMI` |
| `obsm['metabolomic embedding']` | optional (n, 512) | needed only for finetune/downstream; see `Docs/SKILL_EMBEDDINGS.md` |

Column-name constants: `Src/common/constants.py`. Unit conventions: NMR concentrations
are mmol/L (particle sizes nm, ApoB/ApoA1/Albumin g/L, Unsaturation degree); blood units
in the table of §4.4.
列名常量见 `Src/common/constants.py`。单位口径：NMR 浓度为 mmol/L（粒径为 nm，ApoB/ApoA1/Albumin 为 g/L，不饱和度为 degree）；blood 单位见 §4.4 表格。

## 4. UKB Pipeline / UKB 处理流程

Runs on the UKB Research Analysis Platform (RAP); raw exports then processed locally.
在 UKB Research Analysis Platform（RAP）上执行，原始导出后本地处理。

### 4.1 RAP extraction / RAP 数据提取

Use `dx extract_dataset <project:record> --fields <column> -o <file>` per column group
(with retries; RAP rate-limits). Field IDs below are from the data dictionary snapshot
`app60434_20241212085826.dataset.data_dictionary.csv` — **always re-resolve titles
against your own applicant's dictionary version** (IDs shift between versions).
按列组使用 `dx extract_dataset <project:record> --fields <column> -o <file>`（带重试；RAP 有限流）。
下列字段 ID 来自数据字典快照 `app60434_20241212085826.dataset.data_dictionary.csv`——
**务必按你本人申请获得的数据字典版本复核字段标题**（ID 随版本可能变动）。

| RAP folder | Field IDs | Instances |
|---|---|---|
| NMR metabolomics | 23400–23648 (249 measures) + 20280 (glucose-lactate) + 20281 (spectrometer-corrected alanine) | 0/1 (stride 2) |
| NMR metabolomics processing | 20282 batch, 20283 resolved plate swaps, 23649 shipment plate, 23650 spectrometer, 23652 high lactate, 23653 high pyruvate, 23654 low glucose, 23655 low protein, 23658 sample measured datetime, 23659 sample prepared datetime, 23660 well position | 0/1 |
| Blood count | 30000–30300 | 0/1/2 (stride 3) |
| Blood biochemistry | 30600–30890 | 0/1 (stride 2) |
| Baseline characteristics | 31 sex, 34 year of birth, 52 month of birth | — |
| Reception | 53 date of attending, 54 assessment centre, 55 month of attending | 0–3 |
| Body size measures | 21001 BMI, 21002 weight | 0–3 |
| HES / Death | `hesin`, `hesin_diag` (ICD-9/10), `death`, `death_cause` entities (join on the `dnx_hesin_id`-style keys) | — |

Per-biomarker field IDs and QC-flag IDs live in `phrase2_biomarkers.csv`
(`UKB Field ID` / `QC Flag Field ID` columns) — keep this file with the RAP exports.
逐生物标志物字段 ID 与 QC 标志 ID 见 `phrase2_biomarkers.csv`（`UKB Field ID` / `QC Flag Field ID` 列）——此文件随 RAP 导出保留。

### 4.2 NMR technical variation removal / NMR 技术变异去除

Inputs: RAP NMR exports (`NMR_metabolomics_fullcohort/extracted_dataset.csv` +
`NMR_metabolomics_processing_fullcohort/...`). Apply **per biomarker** (all 107
non-derived; sklearn `HuberRegressor()` with default parameters — epsilon=1.35,
alpha=1e-4, max_iter=100, tol=1e-5 — these are sklearn defaults, the original code
passes no arguments):
输入：RAP NMR 导出文件。**逐生物标志物**执行（全部 107 个 non-derived；sklearn
`HuberRegressor()` 默认参数——epsilon=1.35、alpha=1e-4、max_iter=100、tol=1e-5——
原文代码未传参，即 sklearn 默认值）:

1. **Degradation**: degradation hours = (measured − prepared datetime); regress
   `y = log1p(concentration)` on `X = log1p(hours)`; keep residuals
   / **降解校正**：降解时长 =（测量时间 − 制备时间）；回归 `y = log1p(浓度)` 对 `X = log1p(小时数)`；保留残差
2. **Plate row/col**: parse well position into row letter + column int; one-hot with
   reference row `'D'` and reference column `6`; regress residuals on row dummies,
   then on column dummies; keep residuals each step / **板内行列校正**：孔位拆为行字母 +
   列序号；one-hot（参考行 `'D'`、参考列 `6`）；残差依次对行 dummy、列 dummy 回归，每步保留残差
3. **Spectrometer drift**: 8 spectrometers; per spectrometer, plate measurement day =
   most frequent measured date among that plate's samples; sort plates by day; partition
   the day sequence into exactly 10 bins with a min-max dynamic program (cost = max
   segment sum; plates measured on the same day never split across bins); reference bin
   = bin with most samples; regress residuals on bin dummies; keep residuals
   / **谱仪漂移校正**：8 台谱仪；每台内，板测量日 = 该板样本测量日期的众数；按日排序；
   用 min-max 动态规划把日序列恰好分为 10 个 bin（代价 = 最大段和；同一天测量的板不跨 bin）；
   参考 bin = 样本数最多的 bin；残差对 bin dummy 回归，保留残差
4. **Back-transform**: regress `log1p(concentration)` on the final residuals;
   `intercept_` is the estimated mean; corrected log-concentration = residual +
   intercept; corrected < 0 → NaN; `expm1` back to concentration scale
   / **还原浓度**：对 `log1p(浓度)` 与最终残差回归；`intercept_` 为估计均值；
   校正后 log 浓度 = 残差 + 截距；< 0 → NaN；`expm1` 还原为浓度
5. **Outlier plates**: per biomarker, compute per-plate medians; plate medians are
   assumed ~ N(μ, σ); a plate is an outlier if its median falls outside μ ± 3.3744σ
   (constant from the `ukbnmr` package paper); those samples → NaN
   / **离群板剔除**：逐生物标志物计算每板中位数；假设板中位数 ~ N(μ, σ)；中位数落在
   μ ± 3.3744σ（常数取自 `ukbnmr` 包论文）之外即离群，该板样本 → NaN

Output: `data_107nonderived_phrase3_technicalVarRemoved.csv`.
输出：`data_107nonderived_phrase3_technicalVarRemoved.csv`。

### 4.3 107 non-derived selection / 107 个 non-derived 指标选择

From `phrase2_biomarkers.csv`, keep rows with `Type == 'Non-derived'` (107 of 327;
the rest are Percentage/Composite/Ratio/derived types). These 107 are the first 107
columns of the 326-measure set and match `Model_Weights/MetAgeFormer/vocab.txt`
order exactly. Derived biomarkers (for the optional 326 panel) come from
`phrase2_derived_biomarkers.csv` formulas (`df.eval`, NaN-propagating, resolved
iteratively) — not needed for the released models.
取 `phrase2_biomarkers.csv` 中 `Type == 'Non-derived'` 的 107 行（其余为 Percentage/Composite/Ratio/derived 类型）。
这 107 个即 326 指标集的前 107 列，与 `Model_Weights/MetAgeFormer/vocab.txt` 顺序完全一致。
派生指标（可选 326 面板）按 `phrase2_derived_biomarkers.csv` 公式用 `df.eval` 计算（NaN 传播、迭代求解）——已发布模型不需要。

### 4.4 Blood 14 / Blood 14 项

UKB field ID → feature name (data dictionary titles) → unit:

| Field ID | Feature | Unit |
|---|---|---|
| 30000 | White blood cell (leukocyte) count | 10^9 cells/L |
| 30020 | Haemoglobin concentration | g/dL |
| 30030 | Haematocrit percentage | % |
| 30040 | Mean corpuscular volume | fL |
| 30080 | Platelet count | 10^9 cells/L |
| 30710 | C-reactive protein | mg/L |
| 30750 | Glycated haemoglobin (HbA1c) | mmol/mol |
| 30690 | Cholesterol | mmol/L |
| 30760 | HDL cholesterol | mmol/L |
| 30870 | Triglycerides | mmol/L |
| 30700 | Creatinine | µmol/L |
| 30740 | Glucose | mmol/L |
| 30880 | Urate | µmol/L |
| 30720 | Cystatin C | mg/L |

Row order above is load-bearing (it matches `Data/ukb_norm_factors/blood14_factors.csv`
and the Lightweight model). Blood count fields split by instance with stride 3,
biochemistry with stride 2.
上表行序是硬约束（与 `Data/ukb_norm_factors/blood14_factors.csv` 及 Lightweight 模型一致）。Blood count 按实例 stride 3 拆分，biochemistry stride 2。

### 4.5 Baseline, death, HES / 基线、死亡与 HES

- Age = (year diff) + (month diff)/12 from year/month of birth and date of attending,
  rounded to 2 decimals / 年龄 =（年份差）+（月份差）/12（由出生年/月与评估日期计算），保留 2 位小数
- Death: censor date **2022-11-30** (constant baked into the original pipeline — later
  UKB data releases change it); `Death event` = 0/1; `Death event time` = years from
  assessment to death or censor (days/365) / 死亡：截尾日期 **2022-11-30**（原管线写死的常量——后续 UKB 数据版本会变）；`Death event` = 0/1；`Death event time` = 评估日至死亡/截尾的年数（天/365）
- HES disease matrices: **prevalent** = events with −25 < t < 0 years relative to
  assessment; **incident** = 0 < t < 10 years (earliest event after attendance);
  endpoints need ≥ 50 events; prevalent cases are excluded from incident
  / HES 疾病矩阵：**prevalent** = 评估前 −25 < t < 0 年的事件；**incident** = 评估后 0 < t < 10 年
  （取评估后最早事件）；每个终点需 ≥ 50 事件；prevalent 病例从 incident 中排除

### 4.6 Splits and z-score / 数据划分与 z-score

- **train** = England + Wales assessment centres, participants with a single visit
  (exclude centre 11024, the revisit centre); **test** = Scotland + revisit
  participants; **val** = stratified holdout (1% of train: multilabel-stratified on
  prevalence case/control plus 1% random controls) / **train** = 英格兰 + 威尔士评估中心且仅一次访视的受试者（排除回访中心 11024）；**test** = 苏格兰 + 回访受试者；**val** = 分层抽取（train 的 1%：prevalence 病例/对照多标签分层 + 1% 随机对照）
- NMR z-score: compute `mean`/`std` on the **train split only** (pandas `.mean()`/
  `.std()`, i.e. sample std ddof=1); write `var['Z-score mean']/['Z-score std']` and
  `layers['Z-score normalized']` for train/val/test / NMR z-score：**仅 train 划分**计算
  `mean`/`std`（pandas `.mean()`/`.std()`，即样本标准差 ddof=1）；为 train/val/test 写入
  `var['Z-score mean']/['Z-score std']` 与 `layers['Z-score normalized']`
- Blood: fit a sklearn `StandardScaler` on train only, transform val/test
  / Blood：sklearn `StandardScaler` 仅拟合 train，transform val/test

The published copies of these statistics are `Data/ukb_norm_factors/nmr107_factors.csv`
and `blood14_factors.csv` (see that directory's readme for ddof conventions).
这些统计量的发布副本即 `Data/ukb_norm_factors/nmr107_factors.csv` 与
`blood14_factors.csv`（ddof 口径见该目录 readme）。

### 4.7 Verification checklist / 验证清单

- 107 columns in `vocab.txt` order; blood 14 in the §4.4 order / 107 列按 `vocab.txt` 顺序；blood 14 按 §4.4 顺序
- `var` has `Z-score mean`/`Z-score std`, all finite / `var` 含 `Z-score mean`/`Z-score std` 且全部有限
- `layers['Z-score normalized']` exists, float32, NaN positions match raw X / `layers['Z-score normalized']` 存在、float32、NaN 位置与原始 X 一致
- Spot check `z == (X − mean)/std` on a few cells / 抽查若干单元格 `z == (X − mean)/std`
- obs columns present (`Age at assessment (estimated)`, `Death event`, `Death event time`) / obs 列齐全
- No zero-filled missing values anywhere / 任何缺失值都不得被零填充

## 5. CHARLS Pipeline / CHARLS 处理流程

Raw official CHARLS `.dta` releases per wave / 官方 CHARLS 各期 `.dta` 原始发布文件:

| Wave | Raw files |
|---|---|
| 2011 | `Blood_20140429.dta`, `demographic_background.dta`, `health_status_and_functioning.dta`, `weight.dta` (interview time), `biomarkers.dta` (height/weight) |
| 2015 | `Blood.dta`, `Demographic_Background.dta`, `Health_Status_and_Functioning.dta`, `Weights.dta`, `Biomarker.dta`, `Sample_Infor.dta` (interview time + death) |

Column map to the UKB names / 列名映射为 UKB 名称:

| 2011 column | 2015 column | UKB name |
|---|---|---|
| qc1_vb002 | bl_wbc | White blood cell (leukocyte) count |
| qc1_vb004 | bl_hgb | Haemoglobin concentration |
| qc1_vb005 | bl_hct | Haematocrit percentage |
| qc1_vb006 | bl_mcv | Mean corpuscular volume |
| qc1_vb009 | bl_plt | Platelet count |
| newcrp | bl_crp | C-reactive protein |
| newhba1c | bl_hbalc | Glycated haemoglobin (HbA1c) |
| newcho | bl_cho | Cholesterol |
| newhdl | bl_hdl | HDL cholesterol |
| newtg | bl_tg | Triglycerides |
| newcrea | bl_crea | Creatinine |
| newglu | bl_glu | Glucose |
| newua | bl_ua | Urate |
| cystatinc | bl_cysc | Cystatin C |

Unit conversions to UKB units (2011 and 2015 both store HbA1c as NGSP %) /
单位换算到 UKB 单位（2011 与 2015 的 HbA1c 均存为 NGSP %）:

| Measure | Conversion |
|---|---|
| HbA1c | NGSP % → IFCC mmol/mol: `(pct − 2.15) / 0.0915` (inverse: `mmol × 0.0915 + 2.15`) |
| Glucose | mg/dL ÷ 18 |
| Triglycerides | mg/dL ÷ 88.57 |
| Creatinine | mg/dL ÷ 11.312, then × 1000 (→ µmol/L) |
| Cholesterol, HDL | mg/dL ÷ 38.6654 |
| Urate | mg/dL ÷ 16.81, then × 1000 (→ µmol/L) |

Other rules / 其他规则:

- **Wave-1 ID revision**: `id[:9] + '0' + id[-2:]`; household ID = `id + '0'` / **第一期 ID 修订**：`id[:9] + '0' + id[-2:]`；家户 ID = `id + '0'`
- Age = (interview year − birth year) + (interview month − birth month)/12, rounded
  to 2 decimals; interview time from `weight.dta` (2011) / `Sample_Infor.dta` (2015)
  / 年龄 =（访谈年 − 出生年）+（访谈月 − 出生月）/12，保留 2 位小数；访谈时间 2011 取自 `weight.dta`、2015 取自 `Sample_Infor.dta`
- Sex: 0 if the label equals `'2 Female'` (2011) / `'2 female'` (2015 — **case differs**),
  else 1 / 性别：标签为 `'2 Female'`（2011）/ `'2 female'`（2015——**大小写不同**）取 0，否则 1
- BMI = weight/(height/100)² rounded 2dp; height outside 120–220 cm → NaN; weight
  outside 30–200 kg → NaN / BMI = 体重/(身高/100)² 保留 2 位小数；身高超出 120–220 cm → NaN；体重超出 30–200 kg → NaN
- Z-score with the UKB scaler: use `Data/ukb_norm_factors/blood14_factors.csv`
  (`z_mean`/`z_std` columns — these ARE the original `scaler.pkl` mean_/scale_)
  / z-score 使用 UKB scaler：即 `Data/ukb_norm_factors/blood14_factors.csv` 的
  `z_mean`/`z_std` 列（它们就是原 `scaler.pkl` 的 mean_/scale_）
- Drop rows with missing Sex or Age / 删除 Sex 或 Age 缺失的行
- Output: `charls_{2011,2015}_adata.h5ad` with obs `Sex`, `Chronological Age`, `BMI`,
  layer `'Z-score normalized'` / 输出：`charls_{2011,2015}_adata.h5ad`，obs 为
  `Sex`、`Chronological Age`、`BMI`，layer 为 `'Z-score normalized'`

## 6. ADNI Pipeline / ADNI 处理流程

### 6.1 ADNI NMR (Nightingale platform) / ADNI NMR（Nightingale 平台）

Raw `data.csv` (Nightingale export): sample columns `RID`, `VISCODE`, `VISCODE2`,
`EXAMDATE`, ..., then the NMR block from `TOTAL_C` to `S_HDL_TG_PCT` (250 columns;
249 match UKB names, `GLYCEROL` does not).
原始 `data.csv`（Nightingale 导出）：样本列 `RID`、`VISCODE`、`VISCODE2`、`EXAMDATE` 等，
随后是从 `TOTAL_C` 到 `S_HDL_TG_PCT` 的 NMR 区段（250 列；249 个可匹配 UKB 名称，`GLYCEROL` 不匹配）。

- Sample id = `RID_VISCODE2` (`VISCODE2` lowercased, `RID` cleaned to int)
  / 样本 ID = `RID_VISCODE2`（`VISCODE2` 转小写，`RID` 清理为整数）
- Name matching against the UKB 326-measure var index: exact → case-insensitive →
  punctuation-insensitive (lowercase, strip non-alphanumerics) / 名称对齐到 UKB 326 指标集：
  精确 → 大小写不敏感 → 标点不敏感（小写 + 去除非字母数字字符）
- Creatinine: µmol/L → mmol/L (÷ 1000) / 肌酐：µmol/L → mmol/L（÷ 1000）
- Z-score: `z = (raw − UKB train mean)/UKB train std` — for the 107 model features these
  are `Data/ukb_norm_factors/nmr107_factors.csv` / z-score：`z = (原始值 − UKB train mean)/UKB train std`——
  对 107 个模型特征即 `Data/ukb_norm_factors/nmr107_factors.csv`
- Clinical metadata from `clinical_info.xlsx` (header at row 3; columns `RID`, `VISCODE`,
  `EXAMDATE`, `DX_bl`, `DX`, `AGE`, `BMI`, `PTGENDER`, `APOE4`, `CDRSB`, `ADAS11`,
  `ADAS13`, `MMSE`); chronological age = baseline `AGE` + (visit `EXAMDATE` − first
  `EXAMDATE`)/365.25 per subject / 临床元数据取 `clinical_info.xlsx`（表头在第 3 行）；年龄 = 基线 `AGE` +（本次访视 `EXAMDATE` − 首次 `EXAMDATE`）/365.25
- Visit-level diagnosis priority: Q300 `Diagnostic group` (1=HC, 2=EMCI, 3=LMCI, 4=AD,
  5=SMC) → clinical `DX` with `DX_bl` subtype recovery (e.g. MCI+EMCI→EMCI, MCI+LMCI→LMCI)
  → `DX_bl` fallback / 访视级诊断优先级：Q300 `Diagnostic group`（1=HC, 2=EMCI, 3=LMCI,
  4=AD, 5=SMC）→ 临床 `DX`（结合 `DX_bl` 恢复亚型，如 MCI+EMCI→EMCI、MCI+LMCI→LMCI）→ `DX_bl` 兜底
- Output `adni_nmr_processed.h5ad`: 249 matched features, layer
  `'Z-score normalized'`, obsm `Dementia Diagnose`/`Dementia Markers`
  / 输出 `adni_nmr_processed.h5ad`：249 个匹配特征、layer `'Z-score normalized'`、obsm `Dementia Diagnose`/`Dementia Markers`

### 6.2 ADNI Q300 (BGI LC-MS) / ADNI Q300（BGI LC-MS）

Raw Q300 Excel: **7 annotation header rows** — row 0 raw column name, row 1 category,
row 2 HMDB code, row 3 concentration unit (`nM`/`µM`/`mM`). Columns whose row 3 holds a
valid unit are metabolites (104); the others are metadata.
原始 Q300 Excel：**7 行注释表头**——第 0 行列名、第 1 行类别、第 2 行 HMDB 编号、
第 3 行浓度单位（`nM`/`µM`/`mM`）。第 3 行为有效单位的列即代谢物（104 个），其余为元数据列。

- Metadata columns (keep as obs): `Sample ID`, `Plate Number`, `Cohort`,
  `Injection Order`, `RID&VISCODE2`, `RID`, `VID`, `VISCODE`, `VISCODE2`, `Sheet`,
  `Diagnostic group (1 = HC; 2 = EMCI; 3 = LMCI; 4 = AD)` / 元数据列（保留为 obs）：同上列表
- Units → µM: nM × 0.001, µM × 1, mM × 1000 / 单位 → µM：nM × 0.001、µM × 1、mM × 1000
- Data starts at row 7; rows without `RID` are dropped; sample id = `RID_VISCODE2`
  (duplicates get `__dupN` suffixes) / 数据自第 7 行起；无 `RID` 的行删除；样本 ID = `RID_VISCODE2`（重复者加 `__dupN` 后缀）
- **No z-score layer** — Q300 stays in raw µM / **不做 z-score**——Q300 保持 µM 原始浓度

### 6.3 ADNI → model input / ADNI → 模型输入

- NMR: reindex the matched features to the 107 `vocab.txt` order (all 107 are present
  in the ADNI NMR match — verified against the mapping). If a future ADNI release loses
  a feature, NaN-fill it: missing tokens are masked at inference
  / NMR：将匹配特征重排为 107 个 `vocab.txt` 顺序（ADNI NMR 匹配集中 107 个全部存在——已对照映射验证）。
  若未来 ADNI 版本缺失某特征，用 NaN 填充：缺失 token 在推理时被 mask
- Q300 overlap panel: `Src/finetune/deep_gompertz/config/q300_nmr107_overlap_map.json`
  defines 9 chemical overlaps (L-Alanine→Ala, Glycine→Gly, L-Histidine→His,
  L-Leucine→Leu, L-Valine→Val, L-Tyrosine→Tyr, Citric acid→Citrate, C2:0→Acetate,
  3-Hydroxybutyric acid→bOHbutyrate) and 4 excluded matches (C18:2→LA, C22:6→DHA,
  C18:3/C20:5→Omega_3: Nightingale total-FA fractions vs Q300 free FA — semantic
  mismatch). Build the 107-token panel as
  `z = (µM × 0.001 − mean)/std` for the 9 overlaps, all-NaN elsewhere
  / Q300 重叠面板：`Src/finetune/deep_gompertz/config/q300_nmr107_overlap_map.json`
  定义 9 个化学重叠（见上）与 4 个排除项（C18:2→LA、C22:6→DHA、C18:3/C20:5→Omega_3：
  Nightingale 总脂肪酸组分 vs Q300 游离脂肪酸，语义不匹配）。构建 107 token 面板：
  9 个重叠特征取 `z = (µM × 0.001 − mean)/std`，其余全 NaN

Verification: 107 × n h5ad, layer `'Z-score normalized'`, obs age column
`Chronological age`, NaN positions preserved.
验证：107 × n 的 h5ad、layer `'Z-score normalized'`、obs 年龄列 `Chronological age`、NaN 位置保留。

## 7. Notes / 注意事项

- Run evaluation from `Src/` (`cd Src && export PYTHONPATH=.`); data paths follow
  `Data/readme.md` and `Src/common/paths.py` / 评估命令从 `Src/` 运行；数据路径遵循 `Data/readme.md` 与 `Src/common/paths.py`
- Real data never enters the repo; keep processing outputs under `artifacts/` or your
  own directories / 真实数据绝不入库；处理产物放 `artifacts/` 或自建目录
- The 512-dim embeddings (`obsm['metabolomic embedding']`) are only needed for
  finetune/distillation/downstream — not for raw-data conversion itself; see
  `Docs/SKILL_EMBEDDINGS.md` / 512 维 embedding（`obsm['metabolomic embedding']`）仅
  finetune/蒸馏/下游任务需要，数据转换本身不需要；见 `Docs/SKILL_EMBEDDINGS.md`
- Do not add new dependencies (numpy/pandas/anndata/sklearn are the ceiling; CHARLS
  `.dta` reading needs `pyreadstat` locally only) / 不引入新依赖（上限为
  numpy/pandas/anndata/sklearn；CHARLS `.dta` 读取仅本地需要 `pyreadstat`）
