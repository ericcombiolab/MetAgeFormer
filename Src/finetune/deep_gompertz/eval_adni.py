import argparse
import json
import os
import pickle
import warnings
from typing import List, Optional

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
from finetune.deep_gompertz.eval_common import load_model, predict
from metageformer_torch.models import MetAgeFormer_Pretrained
from utils import create_directory, load_tokenizer, save_dict_2_json, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)

AGE_COL_CANDIDATES = [
    "Chronological age",
    "Age at assessment (estimated)",
    "Chronological Age",
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


def load_reference_metabolites(pretrained_dir: str) -> List[str]:
    tokenizer_path = os.path.join(pretrained_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"The pre-trained tokenizer does not exist: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    return list(tokenizer.iden_tokens)


def align_adni_metabolites(adata: ad.AnnData, reference_vars: List[str]) -> ad.AnnData:
    missing = [var for var in reference_vars if var not in adata.var_names]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... ({len(missing)} total)" if len(missing) > 5 else ""
        raise ValueError(f"Missing required metabolites in ADNI data: {preview}{suffix}")
    return adata[:, reference_vars].copy()


def load_pretrained_model_adni(pretrained_dir: str, device: str):
    tokenizer_path = os.path.join(pretrained_dir, "tokenizer.pkl")
    config_path = os.path.join(pretrained_dir, "config.json")
    model_weights_path = os.path.join(pretrained_dir, "model_weights.pth")

    for path in (tokenizer_path, config_path, model_weights_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required pretrained file does not exist: {path}")

    tokenizer = load_tokenizer(tokenizer_path)
    with open(config_path, "r") as file:
        model_config = json.load(file)
    model_config["average_attn_weights"] = False

    embedding_module_conf = {"n_vocabs": {"identifier": tokenizer.vocab_size_identifiers}}
    model = MetAgeFormer_Pretrained(embedding_module_conf, model_config, model_weights_path)
    model.to(device)
    model.eval()
    return model, tokenizer, model_config


def subset_adata(adata: ad.AnnData, n_samples: Optional[int], random_seed: int) -> ad.AnnData:
    if not isinstance(n_samples, int) or n_samples <= 0 or n_samples >= adata.n_obs:
        return adata
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(adata.n_obs, size=n_samples, replace=False)
    return adata[indices].copy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepGompertz on ADNI NMR data: extract CLS embeddings and export predictions."
    )
    parser.add_argument("--pretrained_dir", type=str, default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DEEP_GOMPERTZ_DIR)
    parser.add_argument("--data_path", type=str, default=DEFAULT_ADNI_DATA)
    parser.add_argument("--save_dir", type=str, default=EVAL_ADNI_ROOT)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--age_col", type=str, default="Chronological age")
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
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


if __name__ == "__main__":
    args = parse_args()
    create_directory(args.save_dir)
    set_seeds(args.seed)
    args.num_workers = resolve_num_workers(args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"DataLoader num_workers: {args.num_workers}")

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

    embedding_metadata = {
        "pretrained_dir": args.pretrained_dir,
        "model_dir": args.model_dir,
        "data_path": args.data_path,
        "data_layer": args.data_layer,
        "age_col": age_col,
        "d_model": int(model_config["d_model"]),
        "deep_gompertz_hidden_dim": int(deep_config["hidden_dim"]),
        "embedding_layers": {
            "sample": "MetAgeFormer_Pretrained CLS sample embedding",
        },
    }
    if args.save_embeddings:
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
