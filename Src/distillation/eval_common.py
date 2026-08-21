"""Shared helpers for distilled Lightweight + DeepGompertz evaluation (no pathway aging)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from tqdm import tqdm

from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.training import in_slurm_job
from metageformer_torch.checkpoint import load_distilled_checkpoint
from metageformer_torch.models import MetAgeFormer_Lightweight_DeepGompertz
from distillation.train import load_teacher_gompertz_config
from utils import save_dict_2_json

AGE_COL_CANDIDATES = [
    AGE_COL,
    "Chronological Age",
    "Chronological age",
    "Age at assessment (estimated)",
]

OUTPUT_KEYS = [
    "linear_predictor",
    "log_age_effect",
    "alpha_i",
    "gamma_i",
    "mortality_risk_10y",
    "metabolomic_age",
    "age_gap",
]


def resolve_age_col(adata: ad.AnnData, age_col: Optional[str] = None) -> str:
    if age_col and age_col in adata.obs.columns:
        return age_col
    for candidate in AGE_COL_CANDIDATES:
        if candidate in adata.obs.columns:
            return candidate
    raise KeyError(
        f"No age column found. Tried: {AGE_COL_CANDIDATES + ([age_col] if age_col else [])}. "
        f"Available obs columns: {list(adata.obs.columns)}"
    )


def _load_gompertz_meta(model_dir: str, gompertz_head_path: Optional[str]) -> Dict:
    train_cfg_path = os.path.join(model_dir, "train_config.json")
    train_config: Dict = {}
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            train_config = json.load(f)
        head_config = train_config.get("gompertz_head_config")
        baseline_params = train_config.get("baseline_params")
        if head_config and baseline_params:
            return {
                "gompertz_head_config": head_config,
                "baseline_params": baseline_params,
            }

        cfg_head_path = train_config.get("gompertz_head_path")
        if cfg_head_path:
            resolved = cfg_head_path
            if not os.path.isabs(resolved):
                resolved = os.path.normpath(os.path.join(model_dir, resolved))
            teacher_ckpt = os.path.join(resolved, "model_weights.pth")
            if os.path.exists(teacher_ckpt):
                return load_teacher_gompertz_config(teacher_ckpt)

    if gompertz_head_path:
        teacher_ckpt = os.path.join(gompertz_head_path, "model_weights.pth")
        if os.path.exists(teacher_ckpt):
            return load_teacher_gompertz_config(teacher_ckpt)

    raise ValueError(
        f"Could not resolve DeepGompertz head metadata for {model_dir}. "
        f"Provide --gompertz_head_path or ensure train_config.json includes "
        "gompertz_head_config/baseline_params or gompertz_head_path."
    )


def load_distilled_model(
    model_dir: str,
    device: str,
    gompertz_head_path: Optional[str] = None,
) -> Tuple[MetAgeFormer_Lightweight_DeepGompertz, Dict]:
    weights_path = os.path.join(model_dir, "model_weights.pth")
    conf_path = os.path.join(model_dir, "model_conf.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Distilled checkpoint not found: {weights_path}")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(
            f"model_conf.json required for Lightweight reload (missing {conf_path}). "
            "Re-run distillation/train.py or distillation/train_ablation.py."
        )

    with open(conf_path, "r", encoding="utf-8") as f:
        model_conf = json.load(f)

    meta = _load_gompertz_meta(model_dir, gompertz_head_path)
    model = MetAgeFormer_Lightweight_DeepGompertz(
        model_conf,
        gompertz_head_config=meta["gompertz_head_config"],
        baseline_params=meta["baseline_params"],
    )
    payload = load_distilled_checkpoint(weights_path, map_location="cpu")
    model.load_state_dict(payload["METAGEFORMER_DISTILLED"], strict=True)
    model.to(device)
    model.eval()
    return model, {
        "model_conf": model_conf,
        "gompertz_head_config": meta["gompertz_head_config"],
        "baseline_params": meta["baseline_params"],
        "model_dir": model_dir,
        "gompertz_head_path": gompertz_head_path,
    }


def align_features(adata: ad.AnnData, reference_vars: Sequence[str]) -> ad.AnnData:
    missing = [v for v in reference_vars if v not in adata.var_names]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... ({len(missing)} total)" if len(missing) > 5 else ""
        raise ValueError(f"Missing required features: {preview}{suffix}")
    return adata[:, list(reference_vars)].copy()


def load_reference_features(reference_h5ad: str) -> List[str]:
    adata = ad.read_h5ad(reference_h5ad, backed="r")
    return list(adata.var_names.astype(str))


def _layer_matrix_keep_nan(adata: ad.AnnData, data_layer: str) -> np.ndarray:
    if data_layer == "X":
        x = adata.X
    else:
        if data_layer not in adata.layers:
            raise KeyError(f"Missing layer {data_layer!r}; available={list(adata.layers.keys())}")
        x = adata.layers[data_layer]
    arr = np.asarray(x, dtype=np.float32)
    if hasattr(arr, "toarray"):
        arr = arr.toarray().astype(np.float32)
    return arr


@torch.no_grad()
def evaluate_adata(
    model: torch.nn.Module,
    adata: ad.AnnData,
    age_col: str,
    device: str,
    data_layer: str = "Z-score normalized",
    batch_size: int = 512,
    include_survival: bool = True,
    event_col: str = EVENT_COL,
    time_col: str = TIME_COL,
) -> Tuple[
    pd.DataFrame,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    np.ndarray,
]:
    """Run Lightweight Transformer→DeepGompertz on AnnData.

    Missing biomarkers remain NaN (mask embedding + key_padding_mask; no zero-fill).
    Returns prediction table, optional survival tensors, and student embeddings.
    """
    x_all = _layer_matrix_keep_nan(adata, data_layer)
    ages = np.asarray(adata.obs[age_col].to_numpy(), dtype=np.float32)
    if np.isnan(ages).any():
        raise ValueError(f"Missing ages in column {age_col!r}")

    has_survival = (
        include_survival and event_col in adata.obs.columns and time_col in adata.obs.columns
    )
    events = (
        np.asarray(adata.obs[event_col].to_numpy(), dtype=np.float32) if has_survival else None
    )
    times = np.asarray(adata.obs[time_col].to_numpy(), dtype=np.float32) if has_survival else None

    n = adata.n_obs
    chunks = {key: [] for key in OUTPUT_KEYS}
    emb_chunks = []

    model.eval()
    for start in tqdm(range(0, n, batch_size), desc="eval", disable=in_slurm_job()):
        end = min(start + batch_size, n)
        x = torch.tensor(x_all[start:end], dtype=torch.float32, device=device)
        age = torch.tensor(ages[start:end], dtype=torch.float32, device=device)
        out = model(x, age)
        for key in OUTPUT_KEYS:
            chunks[key].append(out[key].reshape(-1).detach().cpu().numpy())
        emb_chunks.append(out["embs"].detach().cpu().numpy())

    risk = np.concatenate(chunks["mortality_risk_10y"], axis=0)
    prediction = {
        "eid": adata.obs_names.astype(str).to_numpy(),
        "chronological_age": ages,
        **{k: np.concatenate(v, axis=0) for k, v in chunks.items()},
    }
    risk_t = event_t = time_t = None
    if has_survival:
        prediction["event"] = events
        prediction["time"] = times
        risk_t = torch.tensor(risk, dtype=torch.float32)
        event_t = torch.tensor(events, dtype=torch.float32)
        time_t = torch.tensor(times, dtype=torch.float32)

    embeddings = np.concatenate(emb_chunks, axis=0)
    return pd.DataFrame(prediction), risk_t, event_t, time_t, embeddings


def save_metrics(save_dir: str, risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor, auc_time: float):
    cindex = ConcordanceIndex()
    try:
        cidx = cindex(risk, event.bool(), time)
        cidx_ci = cindex.confidence_interval()
        cidx_lines = [str(cidx), str(cidx_ci[0]), str(cidx_ci[1])]
    except Exception as exc:
        cidx_lines = ["nan", "nan", "nan", f"error: {exc}"]

    with open(os.path.join(save_dir, "test_cindex.txt"), "w") as file:
        file.write("\n".join(cidx_lines) + "\n")

    auc = Auc()
    try:
        new_time = torch.tensor([auc_time], dtype=time.dtype)
        auc_value = auc(risk, event.bool(), time, new_time=new_time)[0]
        auc_lines = [str(auc_value.data)]
    except Exception as exc:
        auc_lines = ["nan", f"error: {exc}"]

    with open(os.path.join(save_dir, "test_auc.txt"), "w") as file:
        file.write("\n".join(auc_lines) + "\n")


def write_run_config(save_dir: str, payload: Dict):
    save_dict_2_json(payload, "run_config.json", save_dir)
