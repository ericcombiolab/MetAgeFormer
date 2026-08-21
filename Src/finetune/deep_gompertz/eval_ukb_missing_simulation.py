"""UKB test missingness simulation for DeepGompertz metabolomic age.

Randomly masks observed metabolite concentrations on NMR test, then predicts
metabolomic age. Writes prediction.csv per ratio; does not save embeddings.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex

from ablation.train_from_scratch import load_split_adata
from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.paths import (
    DEFAULT_COHORT,
    DEFAULT_DEEP_GOMPERTZ_DIR,
    DEFAULT_NMR_DATA,
    DEFAULT_PRETRAINED_DIR,
    EVAL_DEEP_GOMPERTZ_ROOT,
)
from common.training import resolve_num_workers
from finetune.deep_gompertz.eval_adni import (
    align_adni_metabolites,
    load_reference_metabolites,
)
from finetune.deep_gompertz.eval_common import load_model, predict
from finetune.deep_gompertz.missing_simulation_utils import (
    DEFAULT_RATIOS,
    apply_random_missingness,
    prediction_summary,
    ratio_dirname,
    ratio_seed,
)
from utils import create_directory, save_dict_2_json, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_SAVE_DIR = os.path.join(EVAL_DEEP_GOMPERTZ_ROOT, DEFAULT_COHORT, "missing_simulation")


def compute_survival_metrics(
    risk: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    auc_time: float,
) -> Dict[str, Optional[float]]:
    risk_t = torch.as_tensor(risk, dtype=torch.float32)
    event_t = torch.as_tensor(event, dtype=torch.float32)
    time_t = torch.as_tensor(time, dtype=torch.float32)

    out: Dict[str, Optional[float]] = {
        "cindex": None,
        "cindex_ci_low": None,
        "cindex_ci_high": None,
        "auc": None,
    }
    cindex = ConcordanceIndex()
    try:
        cidx = cindex(risk_t, event_t.bool(), time_t)
        cidx_ci = cindex.confidence_interval()
        out["cindex"] = float(cidx)
        out["cindex_ci_low"] = float(cidx_ci[0])
        out["cindex_ci_high"] = float(cidx_ci[1])
    except Exception as exc:
        out["cindex_error"] = str(exc)

    auc = Auc()
    try:
        new_time = torch.tensor([auc_time], dtype=time_t.dtype)
        auc_value = auc(risk_t, event_t.bool(), time_t, new_time=new_time)[0]
        out["auc"] = float(auc_value.detach().cpu())
    except Exception as exc:
        out["auc_error"] = str(exc)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "UKB test missingness simulation: randomly mask NMR metabolite "
            "concentrations and predict DeepGompertz metabolomic age (no embeddings)."
        )
    )
    parser.add_argument("--pretrained_dir", type=str, default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DEEP_GOMPERTZ_DIR)
    parser.add_argument("--data_path", type=str, default=DEFAULT_NMR_DATA)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--missing_ratios",
        type=float,
        nargs="+",
        default=DEFAULT_RATIOS,
        help="Missing ratios applied to currently observed metabolites per sample.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--age_col", type=str, default=AGE_COL)
    parser.add_argument("--event_col", type=str, default=EVENT_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
    parser.add_argument("--auc_time", type=float, default=10.0)
    parser.add_argument("--debug_n", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def main():
    args = parse_args()
    create_directory(args.save_dir)
    set_seeds(args.seed)
    args.num_workers = resolve_num_workers(args.num_workers)

    ratios = [float(r) for r in args.missing_ratios]
    for r in ratios:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"Invalid missing ratio {r}; expected values in [0, 1].")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"DataLoader num_workers: {args.num_workers}")
    print(f"missing_ratios: {ratios}")
    print(f"save_dir: {args.save_dir}")
    print(f"split: {args.split}")

    reference_vars = load_reference_metabolites(args.pretrained_dir)
    adata = load_split_adata(
        args.data_path,
        args.split,
        age_col=args.age_col,
        event_col=args.event_col,
        time_col=args.time_col,
        data_layer=args.data_layer,
        debug_n=args.debug_n,
        random_seed=args.seed,
    )
    adata = align_adni_metabolites(adata, reference_vars)
    print(f"samples: {adata.n_obs}")
    print(f"metabolites: {adata.n_vars}")
    print(f"age column: {args.age_col}")

    if adata.obs[args.age_col].isna().any():
        n_missing = int(adata.obs[args.age_col].isna().sum())
        raise ValueError(f"Found {n_missing} samples with missing values in age column '{args.age_col}'.")

    events = np.ascontiguousarray(adata.obs[args.event_col].to_numpy(), dtype=np.float32)
    times = np.ascontiguousarray(adata.obs[args.time_col].to_numpy(), dtype=np.float32)

    model, tokenizer, model_config, deep_config = load_model(
        args.pretrained_dir, args.model_dir, device
    )

    summary_rows: List[Dict] = []

    for ratio in ratios:
        run_seed = ratio_seed(args.seed, ratio)
        ratio_dir = os.path.join(args.save_dir, ratio_dirname(ratio))
        create_directory(ratio_dir)
        print(f"\n=== missing_ratio={ratio:g} (seed={run_seed}) ===", flush=True)

        adata_masked, mask_stats = apply_random_missingness(
            adata,
            data_layer=args.data_layer,
            missing_ratio=ratio,
            seed=run_seed,
        )
        print(
            f"mean_n_observed={mask_stats['mean_n_observed']:.1f} "
            f"mean_n_masked={mask_stats['mean_n_masked']:.1f} "
            f"mean_realized_mask_ratio={mask_stats['mean_realized_mask_ratio']:.4f}",
            flush=True,
        )

        prediction, _ = predict(
            model=model,
            tokenizer=tokenizer,
            adata=adata_masked,
            data_layer=args.data_layer,
            age_col=args.age_col,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            id_col="eid",
        )
        prediction["event"] = events
        prediction["time"] = times

        # Match UKB prediction_test.csv column order.
        col_order = [
            "eid",
            "chronological_age",
            "linear_predictor",
            "log_age_effect",
            "alpha_i",
            "gamma_i",
            "mortality_risk_10y",
            "metabolomic_age",
            "age_gap",
            "event",
            "time",
        ]
        prediction = prediction[col_order]

        prediction_path = os.path.join(ratio_dir, "prediction.csv")
        prediction.to_csv(prediction_path, index=False)
        print(f"Saved predictions to {prediction_path} with {len(prediction)} rows", flush=True)

        metrics = prediction_summary(prediction)
        survival = compute_survival_metrics(
            risk=prediction["mortality_risk_10y"].to_numpy(dtype=np.float32),
            event=events,
            time=times,
            auc_time=args.auc_time,
        )
        metrics.update(survival)

        metadata = {
            "pretrained_dir": args.pretrained_dir,
            "model_dir": args.model_dir,
            "data_path": args.data_path,
            "split": args.split,
            "data_layer": args.data_layer,
            "age_col": args.age_col,
            "event_col": args.event_col,
            "time_col": args.time_col,
            "auc_time": args.auc_time,
            "missing_ratio": ratio,
            "seed": args.seed,
            "ratio_seed": run_seed,
            "d_model": int(model_config["d_model"]),
            "deep_gompertz_hidden_dim": int(deep_config["hidden_dim"]),
            "mask_stats": mask_stats,
            "metrics": metrics,
            "note": (
                "UKB NMR test: observed metabolites randomly set to NaN then inferred "
                "with masking='missing'. Embeddings not saved."
            ),
        }
        save_dict_2_json(metadata, "eval_metadata.json", ratio_dir)

        summary_rows.append(
            {
                "missing_ratio": ratio,
                "ratio_dir": ratio_dirname(ratio),
                "ratio_seed": run_seed,
                **mask_stats,
                **metrics,
            }
        )

    summary = {
        "pretrained_dir": args.pretrained_dir,
        "model_dir": args.model_dir,
        "data_path": args.data_path,
        "split": args.split,
        "data_layer": args.data_layer,
        "age_col": args.age_col,
        "event_col": args.event_col,
        "time_col": args.time_col,
        "auc_time": args.auc_time,
        "seed": args.seed,
        "missing_ratios": ratios,
        "results": summary_rows,
    }
    save_dict_2_json(summary, "summary.json", args.save_dir)
    print(f"\nWrote summary to {os.path.join(args.save_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
