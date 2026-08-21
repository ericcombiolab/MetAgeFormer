import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Tuple

import anndata as ad
import torch

from ablation.train_from_scratch import (
    build_index_loader,
    build_tensor_cache,
    fit_age_baseline_from_adata_from_scratch,
    load_split_adata,
    train,
)
from metageformer_torch.models import DeepGompertzFullyFinetuneModel
from metageformer_torch.checkpoint import load_pretrained_checkpoint
from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.training import init_wandb, str2bool
from common.paths import DEFAULT_PRETRAINED_DIR, FINETUNED_DEEP_GOMPERTZ_ROOT
from utils import create_directory, load_tokenizer, save_dict_2_json, save_tokenizer, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)


def align_adata_metabolites(adata: ad.AnnData, reference_vars: List[str]) -> ad.AnnData:
    missing = [var for var in reference_vars if var not in adata.var_names]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... ({len(missing)} total)" if len(missing) > 5 else ""
        raise ValueError(f"Missing required metabolites in data: {preview}{suffix}")
    return adata[:, reference_vars].copy()


def build_backbone_config(pretrained_config: Dict) -> Dict:
    activation = pretrained_config.get("activation", pretrained_config.get("f_act", "relu"))
    return {
        "n_heads": int(pretrained_config["n_heads"]),
        "n_blocks": int(pretrained_config["n_blocks"]),
        "d_ff": int(pretrained_config["d_ff"]),
        "d_model": int(pretrained_config["d_model"]),
        "dropout": float(pretrained_config.get("dropout", pretrained_config.get("drop_out", 0.1))),
        "activation": activation,
        "need_weights": False,
        "average_attn_weights": False,
        "attn_mode": str(pretrained_config.get("attn_mode", "mixdirect_mask")),
    }


def load_pretrained_assets(pretrained_dir: str) -> Tuple[object, Dict, Dict]:
    tokenizer_path = os.path.join(pretrained_dir, "tokenizer.pkl")
    config_path = os.path.join(pretrained_dir, "config.json")
    model_weights_path = os.path.join(pretrained_dir, "model_weights.pth")

    for path in (tokenizer_path, config_path, model_weights_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required pretrained file does not exist: {path}")

    tokenizer = load_tokenizer(tokenizer_path)
    with open(config_path, "r") as file:
        pretrained_config = json.load(file)

    checkpoint = load_pretrained_checkpoint(model_weights_path)
    backbone_config = build_backbone_config(pretrained_config)
    return tokenizer, backbone_config, checkpoint


def load_pretrained_backbone(model: DeepGompertzFullyFinetuneModel, checkpoint: Dict, device: str):
    model.metageformer_model.load_state_dict(checkpoint["METAGEFORMER"])
    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded pretrained MetAgeFormer weights from checkpoint; "
        f"trainable params: {trainable_params} / {total_params}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully fine-tune DeepGompertz end-to-end from a pretrained MetAgeFormer backbone."
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=DEFAULT_PRETRAINED_DIR,
    )
    parser.add_argument("--data_path", type=str, default="../Data/NMR_dataset_fullcohort_107nonderived")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(FINETUNED_DEEP_GOMPERTZ_ROOT, "fully_finetune_107nonderived"),
    )
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--n_epoch", type=int, default=1000)
    parser.add_argument("--baseline_epoch", type=int, default=1000)
    parser.add_argument("--baseline_n_toler", type=int, default=5)
    parser.add_argument("--n_toler", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--t_window", type=float, default=10.0)
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--init_alpha", type=float, default=-10.5)
    parser.add_argument("--init_gamma", type=float, default=0.09)
    parser.add_argument("--init_beta_age", type=float, default=0.1)
    parser.add_argument("--age_col", type=str, default=AGE_COL)
    parser.add_argument("--event_col", type=str, default=EVENT_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument("--debug_n_train", type=int, default=None)
    parser.add_argument("--debug_n_val", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="MetAgeFormer_DeepGompertz_FullyFinetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=20, help="Log step metrics every N batches; 0 = epoch only")
    parser.add_argument("--log_batch_every", type=int, default=50, help="Print batch progress every N steps")
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_directory(args.save_dir)
    set_seeds(args.seed)
    args.num_workers = 0
    print("Index DataLoader num_workers forced to 0 (lightweight index batches).", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"DataLoader num_workers: {args.num_workers}")
    print(f"pretrained_dir: {args.pretrained_dir}")

    tokenizer, backbone_config, pretrained_checkpoint = load_pretrained_assets(args.pretrained_dir)
    reference_vars = list(tokenizer.iden_tokens)
    print(f"pretrained metabolites: {len(reference_vars)}")

    load_t0 = time.perf_counter()
    train_adata = load_split_adata(
        args.data_path,
        "train",
        args.age_col,
        args.event_col,
        args.time_col,
        args.data_layer,
        debug_n=args.debug_n_train,
        random_seed=args.seed,
    )
    val_adata = load_split_adata(
        args.data_path,
        "val",
        args.age_col,
        args.event_col,
        args.time_col,
        args.data_layer,
        debug_n=args.debug_n_val,
        random_seed=args.seed + 1,
    )
    train_adata = align_adata_metabolites(train_adata, reference_vars)
    val_adata = align_adata_metabolites(val_adata, reference_vars)
    print(
        f"Loaded h5ad in {time.perf_counter() - load_t0:.1f}s | "
        f"train samples: {train_adata.n_obs}, val samples: {val_adata.n_obs}"
    )

    save_tokenizer(tokenizer, save_dir=args.save_dir)

    cache_t0 = time.perf_counter()
    train_cache = build_tensor_cache(
        train_adata,
        tokenizer,
        args.data_layer,
        args.age_col,
        args.event_col,
        args.time_col,
    )
    val_cache = build_tensor_cache(
        val_adata,
        tokenizer,
        args.data_layer,
        args.age_col,
        args.event_col,
        args.time_col,
    )
    del train_adata, val_adata
    print(f"Built tensor caches in {time.perf_counter() - cache_t0:.1f}s")

    train_loader = build_index_loader(
        train_cache.n_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_index_loader(
        val_cache.n_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    embedding_module_conf = {
        "n_vocabs": {"identifier": tokenizer.vocab_size_identifiers},
    }
    run_config = vars(args).copy()
    run_config["embedding_dim"] = int(backbone_config["d_model"])
    run_config["backbone_config"] = backbone_config
    save_dict_2_json(run_config, "run_config.json", args.save_dir)
    wandb_run = init_wandb(args, run_config)

    baseline_params, baseline_loss_epoch, baseline_val_loss_epoch = fit_age_baseline_from_adata_from_scratch(
        train_cache,
        val_cache,
        device,
        args,
        wandb_run=wandb_run,
    )

    model = DeepGompertzFullyFinetuneModel(
        pretrained_dir=args.pretrained_dir,
        embedding_module_conf=embedding_module_conf,
        model_conf=backbone_config,
        baseline_params=baseline_params,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        t_window=args.t_window,
        init_alpha=args.init_alpha,
        init_gamma=args.init_gamma,
        init_beta_age=args.init_beta_age,
        gamma_min=args.gamma_min,
    )
    load_pretrained_backbone(model, pretrained_checkpoint, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}", flush=True)
    backbone_config["n_params"] = total_params
    save_dict_2_json(backbone_config, "backbone_config.json", args.save_dir)

    print("Starting main training...", flush=True)
    train(
        model,
        train_loader,
        val_loader,
        train_cache,
        val_cache,
        device,
        args,
        baseline_params,
        baseline_loss_epoch,
        baseline_val_loss_epoch,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

