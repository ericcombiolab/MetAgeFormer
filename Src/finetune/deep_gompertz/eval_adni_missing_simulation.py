"""ADNI missingness simulation for DeepGompertz metabolomic age.

Randomly masks observed metabolite concentrations at configured ratios, then
predicts metabolomic age. Writes prediction.csv per ratio; does not save embeddings.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List

import anndata as ad
import numpy as np
import torch

from common.paths import (
    DEFAULT_ADNI_DATA,
    DEFAULT_DEEP_GOMPERTZ_DIR,
    DEFAULT_PRETRAINED_DIR,
    EVAL_ADNI_ROOT,
)
from common.training import resolve_num_workers
from finetune.deep_gompertz.eval_adni import (
    align_adni_metabolites,
    load_reference_metabolites,
    resolve_age_col,
    subset_adata,
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

DEFAULT_SAVE_DIR = os.path.join(EVAL_ADNI_ROOT, "missing_simulation")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ADNI missingness simulation: randomly mask metabolite concentrations "
            "and predict DeepGompertz metabolomic age (no embeddings saved)."
        )
    )
    parser.add_argument("--pretrained_dir", type=str, default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DEEP_GOMPERTZ_DIR)
    parser.add_argument("--data_path", type=str, default=DEFAULT_ADNI_DATA)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--missing_ratios",
        type=float,
        nargs="+",
        default=DEFAULT_RATIOS,
        help="Missing ratios applied to currently observed metabolites per sample.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--age_col", type=str, default="Chronological age")
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
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

    reference_vars = load_reference_metabolites(args.pretrained_dir)
    adata = ad.read_h5ad(args.data_path)
    age_col = resolve_age_col(adata, args.age_col)
    adata = align_adni_metabolites(adata, reference_vars)
    adata = subset_adata(adata, args.debug_n, random_seed=args.seed)
    print(f"samples: {adata.n_obs}")
    print(f"metabolites: {adata.n_vars}")
    print(f"age column: {age_col}")

    if adata.obs[age_col].isna().any():
        n_missing = int(adata.obs[age_col].isna().sum())
        raise ValueError(f"Found {n_missing} samples with missing values in age column '{age_col}'.")

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
            age_col=age_col,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            id_col="sample_id",
        )
        prediction_path = os.path.join(ratio_dir, "prediction.csv")
        prediction.to_csv(prediction_path, index=False)
        print(f"Saved predictions to {prediction_path} with {len(prediction)} rows", flush=True)

        metrics = prediction_summary(prediction)
        metadata = {
            "pretrained_dir": args.pretrained_dir,
            "model_dir": args.model_dir,
            "data_path": args.data_path,
            "data_layer": args.data_layer,
            "age_col": age_col,
            "missing_ratio": ratio,
            "seed": args.seed,
            "ratio_seed": run_seed,
            "d_model": int(model_config["d_model"]),
            "deep_gompertz_hidden_dim": int(deep_config["hidden_dim"]),
            "mask_stats": mask_stats,
            "metrics": metrics,
            "note": (
                "Observed metabolites randomly set to NaN then inferred with "
                "masking='missing'. Embeddings not saved."
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
        "data_layer": args.data_layer,
        "age_col": age_col,
        "seed": args.seed,
        "missing_ratios": ratios,
        "results": summary_rows,
    }
    save_dict_2_json(summary, "summary.json", args.save_dir)
    print(f"\nWrote summary to {os.path.join(args.save_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
