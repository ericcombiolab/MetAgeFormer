"""ADNI Q300 overlap → DeepGompertz metabolomic age / age-gap.

Use only Q300 metabolites that chemically overlap NMR 107 non-derived:
map names, convert µM → mmol/L, UKB Z-score, fill those tokens, mask the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

from common.paths import (
    DEFAULT_ADNI_DATA,
    DEFAULT_ADNI_Q300_DATA,
    DEFAULT_DEEP_GOMPERTZ_DIR,
    DEFAULT_PRETRAINED_DIR,
    EVAL_ADNI_Q300_OVERLAP_ROOT,
)
from common.training import resolve_num_workers
from finetune.deep_gompertz.eval_adni import (
    load_reference_metabolites,
    resolve_age_col,
    subset_adata,
)
from finetune.deep_gompertz.eval_common import load_model, predict
from utils import create_directory, save_dict_2_json, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)

DATA_LAYER = "Z-score normalized"
UM_TO_MMOL_L = 0.001

DEFAULT_METABOLITE_MAP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "q300_nmr107_overlap_map.json",
)


def load_overlap_map(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mapping = payload.get("q300_to_nmr")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Empty or invalid q300_to_nmr in metabolite map: {path}")
    return payload


def load_ukb_z_stats(
    nmr_ref_path: str, nmr_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Optional[str]]]:
    """Load UKB train mean/std (and Unit) from ADNI NMR var annotations."""
    nmr = ad.read_h5ad(nmr_ref_path, backed="r")
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    units: Dict[str, Optional[str]] = {}
    missing = [n for n in nmr_names if n not in nmr.var_names]
    if missing:
        raise ValueError(
            f"NMR reference missing metabolites for UKB Z stats: {missing}"
        )
    for name in nmr_names:
        row = nmr.var.loc[name]
        if "Z-score mean" not in nmr.var.columns or "Z-score std" not in nmr.var.columns:
            raise KeyError(
                f"NMR reference var missing 'Z-score mean'/'Z-score std': {nmr_ref_path}"
            )
        mean = float(row["Z-score mean"])
        std = float(row["Z-score std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid UKB Z stats for {name}: mean={mean}, std={std}")
        means[name] = mean
        stds[name] = std
        units[name] = str(row["Unit"]) if "Unit" in nmr.var.columns else None
    nmr.file.close()
    return means, stds, units


def build_q300_overlap_panel(
    q300: ad.AnnData,
    reference_vars: List[str],
    q300_to_nmr: Dict[str, str],
    ukb_mean: Dict[str, float],
    ukb_std: Dict[str, float],
    age_col: str,
    data_layer: str = DATA_LAYER,
    conversion_factor: float = UM_TO_MMOL_L,
) -> Tuple[ad.AnnData, List[Dict], List[str]]:
    """Build 107-token AnnData with Q300-derived UKB Z for overlap tokens only."""
    missing_q = [q for q in q300_to_nmr if q not in q300.var_names]
    if missing_q:
        raise ValueError(f"Q300 data missing mapped metabolites: {missing_q}")

    missing_ref = [n for n in q300_to_nmr.values() if n not in reference_vars]
    if missing_ref:
        raise ValueError(
            f"Mapped NMR names not in tokenizer reference: {missing_ref}"
        )

    if age_col not in q300.obs.columns:
        raise KeyError(f"Age column '{age_col}' not in Q300 obs")

    age = q300.obs[age_col]
    keep = age.notna().to_numpy()
    if not bool(keep.any()):
        raise ValueError(f"No samples with finite age in column '{age_col}'")
    q300 = q300[keep].copy()

    n_obs = q300.n_obs
    n_vars = len(reference_vars)
    z_panel = np.full((n_obs, n_vars), np.nan, dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(reference_vars)}

    unit_rows: List[Dict] = []
    filled_vars: List[str] = []

    for q_name, n_name in q300_to_nmr.items():
        raw = np.asarray(q300[:, q_name].X, dtype=np.float64).reshape(-1)
        if hasattr(raw, "toarray"):
            raw = raw.toarray().reshape(-1)
        mmol = raw * float(conversion_factor)
        mean = ukb_mean[n_name]
        std = ukb_std[n_name]
        z = (mmol - mean) / std
        # Leave non-finite / zero-after-convert as NaN so inference masks them
        valid = np.isfinite(z) & np.isfinite(raw) & (raw != 0)
        col = np.full(n_obs, np.nan, dtype=np.float32)
        col[valid] = z[valid].astype(np.float32)
        z_panel[:, name_to_idx[n_name]] = col
        filled_vars.append(n_name)

        q_source = None
        q_stored = None
        if "source_unit" in q300.var.columns:
            q_source = str(q300.var.loc[q_name, "source_unit"])
        if "target_unit" in q300.var.columns:
            q_stored = str(q300.var.loc[q_name, "target_unit"])
        elif q300.uns.get("concentration_units"):
            q_stored = str(q300.uns["concentration_units"].get("target_unit", "µM"))

        unit_rows.append(
            {
                "q300_name": q_name,
                "nmr_name": n_name,
                "q300_source_unit": q_source,
                "q300_stored_unit": q_stored or "µM",
                "nmr_unit": "mmol/L",
                "conversion_factor": float(conversion_factor),
                "ukb_z_mean": mean,
                "ukb_z_std": std,
                "n_finite_input": int(valid.sum()),
            }
        )

    obs = q300.obs[[age_col]].copy()
    var = pd.DataFrame(index=pd.Index(reference_vars, name="metabolite"))
    out = ad.AnnData(X=np.zeros((n_obs, n_vars), dtype=np.float32), obs=obs, var=var)
    out.layers[data_layer] = z_panel
    out.obs_names = q300.obs_names.copy()
    return out, unit_rows, filled_vars


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ADNI Q300 overlap age-gap: map overlapping metabolites, convert "
            "µM→mmol/L, UKB Z-score, mask non-overlap tokens, predict with "
            "MetAgeFormer + DeepGompertz."
        )
    )
    parser.add_argument("--pretrained_dir", type=str, default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DEEP_GOMPERTZ_DIR)
    parser.add_argument("--q300_path", type=str, default=DEFAULT_ADNI_Q300_DATA)
    parser.add_argument(
        "--nmr_ref_path",
        type=str,
        default=DEFAULT_ADNI_DATA,
        help="ADNI NMR h5ad used only for UKB Z-score mean/std on mapped metabolites.",
    )
    parser.add_argument("--metabolite_map", type=str, default=DEFAULT_METABOLITE_MAP)
    parser.add_argument("--save_dir", type=str, default=EVAL_ADNI_Q300_OVERLAP_ROOT)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--age_col", type=str, default="Chronological age")
    parser.add_argument("--data_layer", type=str, default=DATA_LAYER)
    parser.add_argument("--debug_n", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save CLS embeddings; disabled by default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_directory(args.save_dir)
    set_seeds(args.seed)
    args.num_workers = resolve_num_workers(args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"DataLoader num_workers: {args.num_workers}")
    print(f"q300_path: {args.q300_path}")
    print(f"nmr_ref_path: {args.nmr_ref_path}")
    print(f"metabolite_map: {args.metabolite_map}")
    print(f"save_dir: {args.save_dir}")

    map_payload = load_overlap_map(args.metabolite_map)
    q300_to_nmr = {str(k): str(v) for k, v in map_payload["q300_to_nmr"].items()}
    unit_cfg = map_payload.get("unit_conversion") or {}
    conversion_factor = float(unit_cfg.get("conversion_factor", UM_TO_MMOL_L))

    reference_vars = load_reference_metabolites(args.pretrained_dir)
    ukb_mean, ukb_std, nmr_units = load_ukb_z_stats(
        args.nmr_ref_path, list(q300_to_nmr.values())
    )

    q300 = ad.read_h5ad(args.q300_path)
    age_col = resolve_age_col(q300, args.age_col)
    print(f"age column: {age_col}")
    print(f"Q300 raw samples: {q300.n_obs}, vars: {q300.n_vars}")

    adata, unit_rows, filled_vars = build_q300_overlap_panel(
        q300=q300,
        reference_vars=reference_vars,
        q300_to_nmr=q300_to_nmr,
        ukb_mean=ukb_mean,
        ukb_std=ukb_std,
        age_col=age_col,
        data_layer=args.data_layer,
        conversion_factor=conversion_factor,
    )
    for row in unit_rows:
        u = nmr_units.get(row["nmr_name"])
        if u:
            row["nmr_unit"] = u

    adata = subset_adata(adata, args.debug_n, random_seed=args.seed)
    print(f"samples after age filter/subset: {adata.n_obs}")
    print(f"overlap tokens filled: {len(filled_vars)}")
    print(f"masked tokens: {len(reference_vars) - len(filled_vars)}")

    if adata.obs[age_col].isna().any():
        n_missing = int(adata.obs[age_col].isna().sum())
        raise ValueError(f"Found {n_missing} samples with missing age after panel build.")

    model, tokenizer, model_config, deep_config = load_model(
        args.pretrained_dir, args.model_dir, device
    )
    prediction, embeddings = predict(
        model=model,
        tokenizer=tokenizer,
        adata=adata,
        data_layer=args.data_layer,
        age_col=age_col,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        id_col="sample_id",
        save_embeddings=args.save_embeddings,
    )

    map_used = {
        "q300_to_nmr": q300_to_nmr,
        "excluded_semantic_mismatch": map_payload.get("excluded_semantic_mismatch", {}),
        "unit_conversion": {
            "q300_stored_unit": unit_cfg.get("q300_stored_unit", "µM"),
            "nmr_unit": unit_cfg.get("nmr_unit", "mmol/L"),
            "conversion_factor": conversion_factor,
            "note": unit_cfg.get(
                "note", "µM → mmol/L: multiply by conversion_factor (0.001)"
            ),
        },
        "unit_table": unit_rows,
    }
    save_dict_2_json(map_used, "metabolite_map_used.json", args.save_dir)

    run_metadata = {
        "pretrained_dir": args.pretrained_dir,
        "model_dir": args.model_dir,
        "q300_path": args.q300_path,
        "nmr_ref_path": args.nmr_ref_path,
        "metabolite_map": args.metabolite_map,
        "data_layer": args.data_layer,
        "age_col": age_col,
        "n_samples": int(adata.n_obs),
        "n_overlap_tokens": len(filled_vars),
        "n_masked_tokens": len(reference_vars) - len(filled_vars),
        "reference_vars": reference_vars,
        "filled_vars": filled_vars,
        "conversion_factor": conversion_factor,
        "note": (
            "Input is Q300-derived UKB Z-score for chemical-overlap metabolites only; "
            "remaining NMR 107 tokens are NaN and masked at inference."
        ),
        "d_model": int(model_config["d_model"]),
        "deep_gompertz_hidden_dim": int(deep_config["hidden_dim"]),
        "unit_table": unit_rows,
        "q300_to_nmr": q300_to_nmr,
        "excluded_semantic_mismatch": map_payload.get("excluded_semantic_mismatch", {}),
    }
    save_dict_2_json(run_metadata, "run_metadata.json", args.save_dir)

    if args.save_embeddings:
        embedding_metadata = {
            "pretrained_dir": args.pretrained_dir,
            "model_dir": args.model_dir,
            "q300_path": args.q300_path,
            "data_layer": args.data_layer,
            "age_col": age_col,
            "d_model": int(model_config["d_model"]),
            "deep_gompertz_hidden_dim": int(deep_config["hidden_dim"]),
            "filled_vars": filled_vars,
            "embedding_layers": {
                "sample": "MetAgeFormer_Pretrained CLS sample embedding",
            },
        }
        save_dict_2_json(embedding_metadata, "embedding_metadata.json", args.save_dir)
        embedding_path = os.path.join(args.save_dir, "embedding.pkl")
        with open(embedding_path, "wb") as file:
            pickle.dump(
                {"sample": embeddings, "sample_ids": adata.obs_names.to_numpy()},
                file,
            )
        print(
            f"Saved embeddings to {embedding_path}: sample={embeddings.shape}",
            flush=True,
        )

    prediction_path = os.path.join(args.save_dir, "prediction.csv")
    prediction.to_csv(prediction_path, index=False)
    print(f"Saved predictions to {prediction_path} with {len(prediction)} rows")
    print(f"Saved run_metadata.json and metabolite_map_used.json to {args.save_dir}")


if __name__ == "__main__":
    main()
