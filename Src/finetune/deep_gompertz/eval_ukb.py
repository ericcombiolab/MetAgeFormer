import argparse
import os
import warnings

import numpy as np
import torch
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex

from ablation.train_from_scratch import load_split_adata
from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.training import resolve_num_workers
from metageformer_torch.models import DeepGompertzMetabolomicAgeHead
from common.paths import (
    DEFAULT_COHORT,
    DEFAULT_DEEP_GOMPERTZ_DIR,
    DEFAULT_NMR_DATA,
    DEFAULT_PRETRAINED_DIR,
    EVAL_DEEP_GOMPERTZ_ROOT,
)
from finetune.deep_gompertz.eval_common import load_model, predict
from utils import create_directory, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)


def load_deep_gompertz_model(model_dir: str, device: str):
    checkpoint_path = os.path.join(model_dir, "model_weights.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"The DeepGompertz checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    baseline_params = checkpoint.get("baseline_params")
    if baseline_params is None:
        baseline_params = {
            "alpha_age_scale": config["alpha_age_scale"],
            "gamma_age_scale": config["gamma_age_scale"],
        }

    model = DeepGompertzMetabolomicAgeHead(
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        t_window=config["t_window"],
        age_scale=config.get("age_scale", 1.0),
        baseline_params=baseline_params,
        gamma_min=config.get("gamma_min", 1e-6),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, config


def align_metabolites(adata, reference_vars):
    missing = [name for name in reference_vars if name not in adata.var_names]
    if missing:
        raise ValueError(f"Missing required metabolites: {missing[:5]}")
    return adata[:, reference_vars].copy()


def save_metrics(save_dir: str, risk, event, time, auc_time: float):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end MetAgeFormer + DeepGompertz evaluation on UKB NMR."
    )
    parser.add_argument("--pretrained_dir", type=str, default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_DEEP_GOMPERTZ_DIR)
    parser.add_argument("--data_path", type=str, default=DEFAULT_NMR_DATA)
    parser.add_argument("--save_dir", type=str, default=os.path.join(EVAL_DEEP_GOMPERTZ_ROOT, DEFAULT_COHORT))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--auc_time", type=float, default=10.0)
    parser.add_argument("--data_layer", type=str, default="Z-score normalized")
    parser.add_argument("--age_col", type=str, default=AGE_COL)
    parser.add_argument("--event_col", type=str, default=EVENT_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--debug_n", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_directory(args.save_dir)
    set_seeds(args.seed)
    args.num_workers = resolve_num_workers(args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"DataLoader num_workers: {args.num_workers}")

    model, tokenizer, _, config = load_model(
        args.pretrained_dir, args.model_dir, device
    )
    reference_vars = list(tokenizer.iden_tokens)
    args.age_col = config.get("age_col", args.age_col)
    args.event_col = config.get("event_col", args.event_col)
    args.time_col = config.get("time_col", args.time_col)

    for split in args.splits:
        adata = load_split_adata(
            args.data_path,
            split,
            args.age_col,
            args.event_col,
            args.time_col,
            args.data_layer,
            debug_n=args.debug_n,
            random_seed=args.seed,
        )
        adata = align_metabolites(adata, reference_vars)
        print(f"{split} samples: {adata.n_obs}", flush=True)
        prediction, _ = predict(
            model=model,
            tokenizer=tokenizer,
            adata=adata,
            data_layer=args.data_layer,
            age_col=args.age_col,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            id_col="eid",
        )
        prediction["event"] = np.asarray(
            adata.obs[args.event_col], dtype=np.float32
        )
        prediction["time"] = np.asarray(
            adata.obs[args.time_col], dtype=np.float32
        )
        prediction.to_csv(os.path.join(args.save_dir, f"prediction_{split}.csv"), index=False)
        if split == "test":
            save_metrics(
                args.save_dir,
                torch.from_numpy(prediction["mortality_risk_10y"].to_numpy()),
                torch.from_numpy(prediction["event"].to_numpy()),
                torch.from_numpy(prediction["time"].to_numpy()),
                args.auc_time,
            )

