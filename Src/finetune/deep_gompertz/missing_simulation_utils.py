"""Shared helpers for ADNI / UKB missingness simulation experiments."""

from __future__ import annotations

from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_RATIOS = [0.2, 0.4, 0.6, 0.8]


def ratio_dirname(ratio: float) -> str:
    return f"ratio_{ratio:g}"


def ratio_seed(base_seed: int, ratio: float) -> int:
    return int(base_seed) + int(round(ratio * 100))


def apply_random_missingness(
    adata: ad.AnnData,
    data_layer: str,
    missing_ratio: float,
    seed: int,
) -> Tuple[ad.AnnData, Dict[str, float]]:
    """Mask a fraction of currently observed metabolites per sample (NaN/0 excluded)."""
    if not (0.0 <= missing_ratio <= 1.0):
        raise ValueError(f"missing_ratio must be in [0, 1], got {missing_ratio}")
    if data_layer not in adata.layers:
        raise KeyError(f"Missing layer '{data_layer}' in AnnData")

    out = adata.copy()
    X = np.array(out.layers[data_layer], dtype=np.float32, copy=True)
    rng = np.random.default_rng(seed)

    n_obs = X.shape[0]
    n_vars = X.shape[1]
    n_observed = np.zeros(n_obs, dtype=np.int64)
    n_masked = np.zeros(n_obs, dtype=np.int64)

    for i in range(n_obs):
        observed = np.nonzero(~np.isnan(X[i]) & (X[i] != 0))[0]
        n_observed[i] = len(observed)
        if missing_ratio <= 0.0 or len(observed) == 0:
            continue
        n_mask = int(len(observed) * missing_ratio)
        if n_mask == 0:
            n_mask = 1
        n_mask = min(n_mask, len(observed))
        idx = rng.choice(observed, size=n_mask, replace=False)
        X[i, idx] = np.nan
        n_masked[i] = n_mask

    out.layers[data_layer] = X
    stats = {
        "mean_n_observed": float(n_observed.mean()) if n_obs else 0.0,
        "mean_n_masked": float(n_masked.mean()) if n_obs else 0.0,
        "mean_realized_mask_ratio": float(
            np.mean(n_masked / np.maximum(n_observed, 1))
        )
        if n_obs
        else 0.0,
        "n_metabolites": int(n_vars),
    }
    return out, stats


def prediction_summary(pred: pd.DataFrame) -> Dict[str, float]:
    chron = pred["chronological_age"].to_numpy(dtype=np.float64)
    met_age = pred["metabolomic_age"].to_numpy(dtype=np.float64)
    age_gap = pred["age_gap"].to_numpy(dtype=np.float64)
    corr = float(np.corrcoef(met_age, chron)[0, 1]) if len(pred) > 1 else float("nan")
    return {
        "n_samples": int(len(pred)),
        "mean_abs_age_gap": float(np.mean(np.abs(age_gap))),
        "corr_metabolomic_vs_chronological": corr,
    }
