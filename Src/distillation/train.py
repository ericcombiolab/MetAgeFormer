"""Distill MetAgeFormer Lightweight (blood-token Transformer) + DeepGompertz."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.constants import AGE_COL
from metageformer_torch.checkpoint import load_teacher_gompertz_config
from metageformer_torch.dataset import load_dataset_from_adata_NMR
from metageformer_torch.models import MetAgeFormer_Lightweight_DeepGompertz, deep_gompertz_nll_loss
from utils import create_directory, set_seeds

warnings.filterwarnings("ignore", category=UserWarning)


def get_age_years(adata, device: str) -> torch.Tensor:
    return torch.tensor(adata.obs[AGE_COL].to_numpy(), dtype=torch.float32, device=device)


def get_survival_info(adata, return_tensor=False, device="cpu"):
    event = adata.obs["Death event"].values
    time = adata.obs["Death event time"].values
    if return_tensor:
        event = torch.tensor(event, dtype=torch.float32, device=device)
        time = torch.tensor(time, dtype=torch.float32, device=device)
    return {"event": event, "time": time}


def compute_gompertz_loss(gompertz_out, event, time):
    return deep_gompertz_nll_loss(
        gompertz_out["alpha_i"],
        gompertz_out["gamma_i"],
        gompertz_out["log_age_effect"],
        time,
        event,
    )


def compute_infonce_loss(pred_embs, true_embs, temperature=0.07):
    """InfoNCE: student↔teacher batch similarities, positives on the diagonal."""
    pred_embs = F.normalize(pred_embs, p=2, dim=1)
    true_embs = F.normalize(true_embs, p=2, dim=1)
    similarity_matrix = torch.matmul(pred_embs, true_embs.T) / temperature
    labels = torch.arange(pred_embs.shape[0], device=pred_embs.device)
    return F.cross_entropy(similarity_matrix, labels)


def compute_infonce_symmetric(pred_embs, true_embs, temperature: float = 0.07):
    """CLIP-style symmetric InfoNCE: mean of student→teacher and teacher→student CE."""
    return 0.5 * (
        compute_infonce_loss(pred_embs, true_embs, temperature)
        + compute_infonce_loss(true_embs, pred_embs, temperature)
    )


def layer_to_tensor(adata, device: str) -> torch.Tensor:
    """Keep NaNs (missing); do not fill with 0."""
    arr = np.asarray(adata.layers["Z-score normalized"].copy(), dtype=np.float32)
    if hasattr(arr, "toarray"):
        arr = arr.toarray().astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train(
    model,
    dataloader,
    val_dataloader,
    lr: float = 0.0001,
    n_epoch: int = 20,
    n_toler: Optional[int] = None,
    save_dir: str = "./",
    device: Optional[str] = None,
    temperature: float = 0.2,
    gompertz_weight: float = 1.0,
    emb_weight: float = 1.0,
    freeze_gompertz_head: bool = True,
):
    trainable_params = list(model.lightweight_model.parameters())
    if freeze_gompertz_head:
        model.freeze_gompertz_head()
    else:
        model.unfreeze_gompertz_head()
        trainable_params.extend(p for p in model.gompertz_head.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.98), eps=1e-07)

    best_val_loss = 999999.0
    watchdog = 0

    for epoch in range(n_epoch):
        model.train()
        if freeze_gompertz_head:
            model.freeze_gompertz_head()
        else:
            model.unfreeze_gompertz_head()

        step_loss, step_emb, step_gomp = [], [], []
        for data in tqdm(dataloader):
            optim.zero_grad()
            age_years = get_age_years(data, device)
            survival_info = get_survival_info(data, return_tensor=True, device=device)
            X = layer_to_tensor(data, device)
            teacher_embs = torch.tensor(
                data.obsm["metabolomic embedding"].copy(),
                dtype=torch.float32,
                device=device,
            )

            out = model(X, age_years)
            emb_loss = compute_infonce_symmetric(out["embs"], teacher_embs, temperature)
            gompertz_loss = compute_gompertz_loss(out, survival_info["event"], survival_info["time"])
            loss = emb_weight * emb_loss + gompertz_weight * gompertz_loss
            loss.backward()
            optim.step()

            step_loss.append(loss.item())
            step_emb.append(float(emb_loss.detach().cpu()))
            step_gomp.append(float(gompertz_loss.detach().cpu()))

        model.eval()
        val_loss_c, val_emb_c, val_gomp_c = [], [], []
        for data_val in val_dataloader:
            X_val = layer_to_tensor(data_val, device)
            teacher_embs_val = torch.tensor(
                data_val.obsm["metabolomic embedding"].copy(),
                dtype=torch.float32,
                device=device,
            )
            age_years = get_age_years(data_val, device)
            with torch.no_grad():
                out_val = model(X_val, age_years)
                survival_info = get_survival_info(data_val, return_tensor=True, device=device)
                emb_loss_val = compute_infonce_symmetric(
                    out_val["embs"], teacher_embs_val, temperature
                )
                gompertz_loss_val = compute_gompertz_loss(
                    out_val, survival_info["event"], survival_info["time"]
                )
                val_loss = emb_weight * emb_loss_val + gompertz_weight * gompertz_loss_val
            val_loss_c.append(float(val_loss.cpu()))
            val_emb_c.append(float(emb_loss_val.cpu()))
            val_gomp_c.append(float(gompertz_loss_val.cpu()))

        train_m = float(np.mean(step_loss))
        val_m = float(np.mean(val_loss_c))
        print(
            f"Epoch {epoch + 1}/{n_epoch} | Train Loss: {train_m:.6f} "
            f"(emb={np.mean(step_emb):.6f}, gompertz={np.mean(step_gomp):.6f})"
            f" | Val Loss: {val_m:.6f} "
            f"(emb={np.mean(val_emb_c):.6f}, gompertz={np.mean(val_gomp_c):.6f})"
        )

        if val_m < best_val_loss:
            best_val_loss = val_m
            watchdog = 0
            model.save_distilled(os.path.join(save_dir, "model_weights.pth"))
        else:
            watchdog += 1
            if n_toler is not None and watchdog >= n_toler:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distill MetAgeFormer_Lightweight (blood-token Transformer) + DeepGompertz."
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

    loss_mode = train_settings.get("loss_mode", "symmetric_infonce")
    if loss_mode not in {"symmetric_infonce", "contrastive", "infonce"}:
        raise ValueError(
            f"Unsupported loss_mode={loss_mode!r}. Use 'symmetric_infonce' "
            f"(legacy 'contrastive'/'infonce' also accepted)."
        )
    temperature = float(train_settings.get("temperature", 0.2))
    gompertz_weight = float(train_settings.get("gompertz_weight", train_settings.get("cox_weight", 1.0)))
    emb_weight = float(train_settings.get("emb_weight", 1.0))
    freeze_gompertz_head = bool(train_settings.get("freeze_gompertz_head", True))

    if train_settings.get("pathway_head_path") or float(train_settings.get("pathway_weight", 0.0)) > 0:
        raise ValueError(
            "Pathway activity distillation is removed. Use Lightweight+DeepGompertz only "
            "(no pathway_head_path / pathway_weight)."
        )

    save_dir = os.path.abspath(os.path.join(train_settings["save_dir"], train_settings["save_note"]))
    gompertz_head_path = train_settings["gompertz_head_path"]
    checkpoint_path = os.path.join(gompertz_head_path, "model_weights.pth")

    create_directory(save_dir)
    set_seeds(3047)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_meta = load_teacher_gompertz_config(checkpoint_path)
    run_config = dict(train_settings)
    run_config.update(
        {
            "training_mode": "distill_from_teacher",
            "gompertz_head_config": teacher_meta["gompertz_head_config"],
            "baseline_params": teacher_meta["baseline_params"],
        }
    )
    with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    train_loader, train_data = load_dataset_from_adata_NMR(
        os.path.join(data_path, "train.h5ad"),
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )
    val_loader, _ = load_dataset_from_adata_NMR(
        os.path.join(data_path, "val.h5ad"),
        shuffle=True,
        batch_size=batch_size,
        device=device,
    )

    model_conf = {
        "n_features": int(train_data.n_vars),
        "d_model": int(d_model),
        "dropout": float(drop_out),
        "n_heads": n_heads,
        "n_blocks": n_blocks,
        "d_ff": d_ff,
    }
    with open(os.path.join(save_dir, "model_conf.json"), "w", encoding="utf-8") as f:
        json.dump(model_conf, f, indent=2)
        f.write("\n")

    model = MetAgeFormer_Lightweight_DeepGompertz(
        model_conf,
        gompertz_head_config=teacher_meta["gompertz_head_config"],
        baseline_params=teacher_meta["baseline_params"],
    )
    model._load_gompertz_head_weights(checkpoint_path, device=device)
    if freeze_gompertz_head:
        model.freeze_gompertz_head()
    else:
        model.unfreeze_gompertz_head()
    model.to(device)

    n_student = sum(p.numel() for p in model.lightweight_model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.gompertz_head.parameters() if p.requires_grad)
    print(
        f"symmetric InfoNCE | temp={temperature} emb_w={emb_weight} "
        f"gompertz_w={gompertz_weight} freeze_head={freeze_gompertz_head}"
    )
    print(f"model_conf={model_conf}")
    print(f"Trainable lightweight params:\t{n_student}")
    print(f"Trainable gompertz_head params:\t{n_head}")
    print(f"Checkpoint dir: {save_dir}")
    print(f"Teacher DeepGompertz path (read-only): {checkpoint_path}")

    train(
        model,
        train_loader,
        val_loader,
        lr=lr,
        n_epoch=n_epoch,
        n_toler=n_toler,
        save_dir=save_dir,
        device=device,
        temperature=temperature,
        gompertz_weight=gompertz_weight,
        emb_weight=emb_weight,
        freeze_gompertz_head=freeze_gompertz_head,
    )
    print(f"Distilled checkpoint saved under {save_dir} (model_weights.pth only).")
    print("Run distillation/eval_ukb.py or distillation/eval_charls.py for evaluation.")
