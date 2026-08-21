"""Evaluate distilled Lightweight + DeepGompertz on CHARLS blood cohorts (2011 / 2015)."""

from __future__ import annotations

import argparse
import os
import warnings
from typing import List

import anndata as ad

from common.paths import (
    DATA_ROOT,
    DEFAULT_DEEP_GOMPERTZ_DIR,
    DEFAULT_DISTILLED_DIR,
    EVAL_DISTILLED_LIGHTWEIGHT_ROOT,
)
from common.training import resolve_num_workers
from distillation.eval_common import (
    align_features,
    evaluate_adata,
    load_distilled_model,
    load_reference_features,
    resolve_age_col,
    write_run_config,
)
from utils import create_directory, set_seeds

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CHARLS_DIR = os.path.join(DATA_ROOT, "Charls_datasets")
DEFAULT_CHARLS_FILES = [
    "charls_2011_adata.h5ad",
    "charls_2015_adata.h5ad",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate distilled Lightweight (blood-token Transformer) on CHARLS blood panels."
    )
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DISTILLED_DIR)
    parser.add_argument(
        "--gompertz_head_path",
        type=str,
        default=None,
        help="Teacher DeepGompertz dir (optional if train_config.json stores head meta).",
    )
    parser.add_argument("--charls_dir", type=str, default=DEFAULT_CHARLS_DIR)
    parser.add_argument(
        "--charls_files",
        nargs="+",
        default=DEFAULT_CHARLS_FILES,
        help="CHARLS AnnData filenames under charls_dir (two cohorts by default).",
    )
    parser.add_argument(
        "--reference_features_h5ad",
        type=str,
        default=None,
        help="Optional AnnData whose var_names define feature order. "
        "Default: use each CHARLS file's own var order (matches UKB blood panel).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Default: artifacts/eval/distilled/lightweight/<model_dir basename>/charls",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
    parser.add_argument("--age_col", type=str, default="Chronological Age")
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--debug_n", type=int, default=None)
    return parser.parse_args()


def cohort_tag(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    for part in stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return part
    return stem


def main():
    args = parse_args()
    set_seeds(args.seed)
    args.num_workers = resolve_num_workers(args.num_workers)

    model_note = os.path.basename(os.path.normpath(args.model_dir))
    if args.save_dir is None:
        args.save_dir = os.path.join(EVAL_DISTILLED_LIGHTWEIGHT_ROOT, model_note, "charls")
    create_directory(args.save_dir)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model_dir: {args.model_dir}")
    print(f"save_dir: {args.save_dir}")

    gompertz_head_path = args.gompertz_head_path or DEFAULT_DEEP_GOMPERTZ_DIR
    model, meta = load_distilled_model(
        args.model_dir, device, gompertz_head_path=gompertz_head_path
    )
    print(f"model_conf: {meta['model_conf']}", flush=True)

    reference_vars = None
    if args.reference_features_h5ad:
        reference_vars = load_reference_features(args.reference_features_h5ad)
        if len(reference_vars) != meta["model_conf"]["n_features"]:
            raise ValueError(
                f"Reference features n={len(reference_vars)} != model n_features="
                f"{meta['model_conf']['n_features']}"
            )

    evaluated: List[dict] = []
    for fname in args.charls_files:
        path = fname if os.path.isabs(fname) else os.path.join(args.charls_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CHARLS file not found: {path}")
        tag = cohort_tag(fname)
        adata = ad.read_h5ad(path)
        feat_order = reference_vars if reference_vars is not None else list(adata.var_names.astype(str))
        if len(feat_order) != meta["model_conf"]["n_features"]:
            raise ValueError(
                f"CHARLS {tag} n_features={len(feat_order)} != model "
                f"n_features={meta['model_conf']['n_features']}"
            )
        adata = align_features(adata, feat_order)
        if args.debug_n is not None and args.debug_n > 0 and args.debug_n < adata.n_obs:
            adata = adata[: int(args.debug_n)].copy()

        age_col = resolve_age_col(adata, args.age_col)
        print(f"CHARLS {tag}: n={adata.n_obs}; age_col={age_col}; file={path}", flush=True)

        prediction, _, _, _, _ = evaluate_adata(
            model,
            adata,
            age_col=age_col,
            device=device,
            data_layer=args.data_layer,
            batch_size=args.batch_size,
            include_survival=False,
        )
        assert "event" not in prediction.columns and "time" not in prediction.columns

        cohort_dir = os.path.join(args.save_dir, tag)
        create_directory(cohort_dir)
        pred_path = os.path.join(cohort_dir, "prediction.csv")
        prediction.to_csv(pred_path, index=False)
        print(f"wrote {pred_path}", flush=True)

        write_run_config(
            cohort_dir,
            {
                "task": "distilled_lightweight_eval_charls",
                "cohort": tag,
                "data_file": path,
                "model_dir": args.model_dir,
                "gompertz_head_path": meta["gompertz_head_path"],
                "n_samples": int(adata.n_obs),
                "n_features": int(meta["model_conf"]["n_features"]),
                "d_model": int(meta["model_conf"]["d_model"]),
                "feature_names": feat_order,
                "age_col": age_col,
                "data_layer": args.data_layer,
                "prediction_file": os.path.basename(pred_path),
                "note": "No event/time columns (CHARLS blood panel has no mortality follow-up).",
            },
        )
        evaluated.append({"cohort": tag, "n_samples": int(adata.n_obs), "dir": cohort_dir})

    write_run_config(
        args.save_dir,
        {
            "task": "distilled_lightweight_eval_charls_summary",
            "model_dir": args.model_dir,
            "gompertz_head_path": meta["gompertz_head_path"],
            "cohorts": evaluated,
        },
    )


if __name__ == "__main__":
    main()
