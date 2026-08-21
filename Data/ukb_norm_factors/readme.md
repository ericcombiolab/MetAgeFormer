# UKB-derived Normalization Factors / UKB 派生的归一化因子

> English: Per-feature z-score parameters (`mean` and `std`) used by all released
> MetAgeFormer models. They are **aggregate statistics** derived from the UK Biobank
> (UKB) **training split only** — no individual-level data is contained here.
>
> 中文：所有已发布 MetAgeFormer 模型使用的逐特征 z-score 参数（`mean` 与 `std`）。
> 它们是仅由 UK Biobank（UKB）**训练划分**计算得到的**聚合统计量**，不含任何个体数据。

## Files / 文件

| File | Rows | Contents |
|---|---|---|
| `nmr107_factors.csv` | 107 | `feature,display_name,unit,z_mean,z_std` — order is exactly `Model_Weights/MetAgeFormer/vocab.txt` (token order) |
| `blood14_factors.csv` | 14 | `feature,unit,z_mean,z_std` — the Lightweight blood panel, order is load-bearing (see below) |

## Provenance / 来源与口径

- **NMR 107** — computed on the UKB NMR **train split only** (n_train = 430,046
  participants; splits defined in `Docs/SKILL_DATA_PREPROCESSING.md`), using
  pandas `.mean()` / `.std()` (**sample std, ddof=1**), then applied (never refit)
  to the val/test splits, CHARLS, and ADNI.
  `z = (x − z_mean) / z_std`, with `x` in the `unit` column's unit (mostly mmol/L).
- **Blood 14** — a sklearn `StandardScaler` fit on the UKB blood-panel **train split
  only**; `z_mean = scaler.mean_`, `z_std = scaler.scale_` (**population std,
  ddof=0**). The scaler pickle stores no feature names, so the row order below IS the
  model order — never reorder this file. The same scaler was applied to CHARLS.
- Extraction date: 2026-08-22, exported programmatically from the UKB train h5ad
  `var` annotations (NMR) and the committed `scaler.pkl` (blood).

## Privacy note / 隐私说明

Per-feature mean/std/unit are safe-to-publish aggregate statistics. The following are
**never** published and must never be added to this directory: per-sample values,
subject IDs, per-feature min/max or quantiles (extreme values could identify
individuals), or any raw data rows.
