"""Evaluate distilled Lightweight + DeepGompertz on the UKB blood test set."""

from __future__ import annotations

import argparse
import os
import warnings

import anndata as ad
import numpy as np

from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.paths import (
    DEFAULT_DEEP_GOMPERTZ_DIR,
    DEFAULT_DISTILLED_DIR,
    DEFAULT_EMBEDDING_DATA,
    EVAL_DISTILLED_LIGHTWEIGHT_ROOT,
)
from common.training import resolve_num_workers
from distillation.eval_common import (
    align_features,
    evaluate_adata,
    load_distilled_model,
    load_reference_features,
    resolve_age_col,
    save_metrics,
    write_run_config,
)
from utils import create_directory, set_seeds

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate distilled Lightweight (blood-token Transformer) on UKB blood test set."
    )
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DISTILLED_DIR)
    parser.add_argument(
        "--gompertz_head_path",
        type=str,
        default=None,
        help="Teacher DeepGompertz dir (optional if train_config.json stores head meta).",
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_EMBEDDING_DATA)
    parser.add_argument(
        "--reference_features_h5ad",
        type=str,
        default=None,
        help="AnnData whose var_names define feature order (default: data_path/train.h5ad).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Default: artifacts/eval/distilled/lightweight/<model_dir basename>/ukb",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--auc_time", type=float, default=10.0)
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
    parser.add_argument("--age_col", type=str, default=AGE_COL)
    parser.add_argument("--event_col", type=str, default=EVENT_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--debug_n", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    args.num_workers = resolve_num_workers(args.num_workers)

    model_note = os.path.basename(os.path.normpath(args.model_dir))
    if args.save_dir is None:
        args.save_dir = os.path.join(EVAL_DISTILLED_LIGHTWEIGHT_ROOT, model_note, "ukb")
    create_directory(args.save_dir)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model_dir: {args.model_dir}")
    print(f"save_dir: {args.save_dir}")

    gompertz_head_path = args.gompertz_head_path or DEFAULT_DEEP_GOMPERTZ_DIR
    model, meta = load_distilled_model(
        args.model_dir, device, gompertz_head_path=gompertz_head_path
    )
    print(f"model_conf: {meta['model_conf']}")

    ref_path = args.reference_features_h5ad or os.path.join(args.data_path, "train.h5ad")
    reference_vars = load_reference_features(ref_path)
    if len(reference_vars) != meta["model_conf"]["n_features"]:
        raise ValueError(
            f"Reference features n={len(reference_vars)} != model n_features="
            f"{meta['model_conf']['n_features']}"
        )

    split_path = os.path.join(args.data_path, f"{args.split}.h5ad")
    adata = ad.read_h5ad(split_path)
    adata = align_features(adata, reference_vars)
    if args.debug_n is not None and args.debug_n > 0 and args.debug_n < adata.n_obs:
        adata = adata[: int(args.debug_n)].copy()

    age_col = resolve_age_col(adata, args.age_col)
    print(f"{args.split} samples: {adata.n_obs}; age_col={age_col}")

    prediction, risk, event, time, embeddings = evaluate_adata(
        model,
        adata,
        age_col=age_col,
        device=device,
        data_layer=args.data_layer,
        batch_size=args.batch_size,
        include_survival=True,
        event_col=args.event_col,
        time_col=args.time_col,
    )
    pred_path = os.path.join(args.save_dir, f"prediction_{args.split}.csv")
    prediction.to_csv(pred_path, index=False)
    emb_path = os.path.join(args.save_dir, f"embeddings_{args.split}.npz")
    np.savez_compressed(emb_path, embeddings=embeddings, eid=prediction["eid"].to_numpy())
    print(f"wrote {pred_path}")
    print(f"wrote {emb_path}")

    if args.split == "test" and risk is not None:
        save_metrics(args.save_dir, risk, event, time, args.auc_time)
        print(f"wrote test_cindex.txt / test_auc.txt under {args.save_dir}")

    write_run_config(
        args.save_dir,
        {
            "task": "distilled_lightweight_eval_ukb",
            "model_dir": args.model_dir,
            "gompertz_head_path": meta["gompertz_head_path"],
            "data_path": args.data_path,
            "split": args.split,
            "n_samples": int(adata.n_obs),
            "n_features": int(meta["model_conf"]["n_features"]),
            "d_model": int(meta["model_conf"]["d_model"]),
            "feature_names": reference_vars,
            "age_col": age_col,
            "data_layer": args.data_layer,
            "prediction_file": os.path.basename(pred_path),
            "embeddings_file": os.path.basename(emb_path),
            "missing_policy": "NaN -> mask_emb + key_padding_mask (no zero-fill)",
        },
    )


if __name__ == "__main__":
    main()
