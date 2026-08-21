"""Train MetAgeFormer Lightweight (blood-token Transformer) + DeepGompertz from scratch."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
import torch
from tqdm import tqdm

from common.constants import AGE_COL
from common.gompertz_baseline import (
    AgeOnlyGompertzBaseline,
    evaluate_age_baseline,
    inverse_softplus,
)
from metageformer_torch.models import MetAgeFormer_Lightweight_DeepGompertz
from distillation.train import compute_gompertz_loss, get_age_years, get_survival_info, layer_to_tensor
from metageformer_torch.dataset import load_dataset_from_adata_NMR
from utils import create_directory, set_seeds

warnings.filterwarnings("ignore", category=UserWarning)


def collect_survival_arrays(adata: ad.AnnData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    age = np.ascontiguousarray(adata.obs[AGE_COL].to_numpy(), dtype=np.float32)
    event = np.ascontiguousarray(adata.obs["Death event"].to_numpy(), dtype=np.float32)
    time = np.ascontiguousarray(adata.obs["Death event time"].to_numpy(), dtype=np.float32)
    return age, event, time


def fit_age_baseline_from_adata(
    train_adata: ad.AnnData,
    val_adata: ad.AnnData,
    device: str,
    baseline_epoch: int = 50,
    baseline_lr: float = 1e-3,
    baseline_n_toler: int = 5,
    init_alpha: float = -10.5,
    init_gamma: float = 0.09,
    grad_clip: float = 0.5,
) -> Dict[str, float]:
    """Fit age-only Gompertz baseline before training the DeepGompertz head."""
    model = AgeOnlyGompertzBaseline(init_alpha=init_alpha, init_gamma=init_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=baseline_lr)

    train_age, train_event, train_time = collect_survival_arrays(train_adata)
    val_age, val_event, val_time = collect_survival_arrays(val_adata)
    train_age = torch.from_numpy(train_age).to(device)
    train_event = torch.from_numpy(train_event).to(device)
    train_time = torch.from_numpy(train_time).to(device)
    val_age = torch.from_numpy(val_age).to(device)
    val_event = torch.from_numpy(val_event).to(device)
    val_time = torch.from_numpy(val_time).to(device)

    best_val_loss = float("inf")
    best_params = model.params_dict()
    watchdog = 0

    for epoch in range(baseline_epoch):
        model.train()
        optimizer.zero_grad()
        train_loss = model(train_age, train_time, train_event)
        train_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

        train_loss_value = float(train_loss.detach().cpu())
        val_loss_value = evaluate_age_baseline(model, val_age, val_time, val_event)
        print(f"baseline epoch: {epoch}; train loss: {train_loss_value}; val loss: {val_loss_value}")

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_params = model.params_dict()
            watchdog = 0
        else:
            watchdog += 1
            if baseline_n_toler > 0 and watchdog >= baseline_n_toler:
                break

    with torch.no_grad():
        model.alpha.copy_(torch.tensor(best_params["alpha_age_scale"], device=device))
        model.raw_gamma.copy_(torch.tensor(inverse_softplus(best_params["gamma_age_scale"]), device=device))

    print(f"Fitted age baseline: {best_params}")
    return best_params


def train(
    model,
    dataloader,
    val_dataloader,
    lr: float = 0.0001,
    n_epoch: int = 20,
    n_toler: Optional[int] = None,
    save_dir: str = "./",
    device: Optional[str] = None,
):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-07)

    best_val_loss = 999999.0
    watchdog = 0

    for epoch in range(n_epoch):
        model.train()
        step_loss_collect = []

        for data in tqdm(dataloader):
            optim.zero_grad()
            age_years = get_age_years(data, device)
            survival_info = get_survival_info(data, return_tensor=True, device=device)
            X = layer_to_tensor(data, device)

            out = model(X, age_years)
            loss = compute_gompertz_loss(out, survival_info["event"], survival_info["time"])
            loss.backward()
            optim.step()
            step_loss_collect.append(loss.item())

        train_loss = float(np.mean(step_loss_collect))

        model.eval()
        val_loss_collect = []
        with torch.no_grad():
            for data_val in val_dataloader:
                X_val = layer_to_tensor(data_val, device)
                age_years = get_age_years(data_val, device)
                out_val = model(X_val, age_years)
                survival_info = get_survival_info(data_val, return_tensor=True, device=device)
                loss_val = compute_gompertz_loss(out_val, survival_info["event"], survival_info["time"])
                val_loss_collect.append(float(loss_val.detach().cpu()))

        val_loss = float(np.mean(val_loss_collect))
        print(f"Epoch {epoch + 1}/{n_epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            watchdog = 0
            model.save_distilled(os.path.join(save_dir, "model_weights.pth"))
        else:
            watchdog += 1
            if n_toler is not None and watchdog >= n_toler:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MetAgeFormer_Lightweight + DeepGompertz from scratch (ablation; Gompertz NLL only)."
    )
    parser.add_argument("--train_config", type=str, required=True)
    args = parser.parse_args()

    with open(args.train_config, "r") as file:
        train_settings = json.load(file)

    lr = train_settings["lr"]
    batch_size = train_settings["batch_size"]
    n_epoch = train_settings["n_epoch"]
    n_toler = train_settings["n_toler"]
    data_path = train_settings["data_path"]
    drop_out = train_settings["drop_out"]
    d_model = train_settings["d_model"]
    n_heads = int(train_settings.get("n_heads", 4))
    n_blocks = int(train_settings.get("n_blocks", 2))
    d_ff = int(train_settings.get("d_ff", max(256, d_model * 2)))

    baseline_epoch = train_settings.get("baseline_epoch", 50)
    baseline_lr = train_settings.get("baseline_lr", 1e-3)
    baseline_n_toler = train_settings.get("baseline_n_toler", 5)
    hidden_dim = train_settings.get("hidden_dim", 64)
    gompertz_dropout = train_settings.get("gompertz_dropout", 0.1)
    t_window = train_settings.get("t_window", 10.0)
    gamma_min = train_settings.get("gamma_min", 1e-6)

    save_dir = os.path.abspath(os.path.join(train_settings["save_dir"], train_settings["save_note"]))

    create_directory(save_dir)
    set_seeds(3047)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, train_data = load_dataset_from_adata_NMR(
        os.path.join(data_path, "train.h5ad"),
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    val_loader, val_data = load_dataset_from_adata_NMR(
        os.path.join(data_path, "val.h5ad"),
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )

    baseline_params = fit_age_baseline_from_adata(
        train_data,
        val_data,
        device=device,
        baseline_epoch=baseline_epoch,
        baseline_lr=baseline_lr,
        baseline_n_toler=baseline_n_toler,
    )

    gompertz_head_config = {
        "hidden_dim": hidden_dim,
        "dropout": gompertz_dropout,
        "t_window": t_window,
        "gamma_min": gamma_min,
    }
    model_conf = {
        "n_features": int(train_data.n_vars),
        "d_model": int(d_model),
        "dropout": float(drop_out),
        "n_heads": n_heads,
        "n_blocks": n_blocks,
        "d_ff": d_ff,
    }

    run_config = dict(train_settings)
    run_config.update(
        {
            "training_mode": "ablation_from_scratch",
            "gompertz_head_config": gompertz_head_config,
            "baseline_params": baseline_params,
        }
    )
    with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")
    with open(os.path.join(save_dir, "model_conf.json"), "w", encoding="utf-8") as f:
        json.dump(model_conf, f, indent=2)
        f.write("\n")

    model = MetAgeFormer_Lightweight_DeepGompertz(
        model_conf,
        gompertz_head_config=gompertz_head_config,
        baseline_params=baseline_params,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model_conf={model_conf}")
    print(f"Trainable params (lightweight + gompertz_head):\t{n_params}")
    print(f"Checkpoint dir: {save_dir}")

    train(
        model,
        train_loader,
        val_loader,
        lr=lr,
        n_epoch=n_epoch,
        n_toler=n_toler,
        save_dir=save_dir,
        device=device,
    )
    print(f"Ablation checkpoint saved under {save_dir} (model_weights.pth only).")
    print("Run distillation/eval_ukb.py or distillation/eval_charls.py for evaluation.")
