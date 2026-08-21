import argparse
import os
import warnings
from typing import Dict

import numpy as np
import torch

from common.constants import AGE_COL, EVENT_COL, TIME_COL, EMBEDDING_KEY
from common.training import (
    init_wandb,
    progress_iter,
    resolve_num_workers,
    safe_cindex,
    str2bool,
)
from common.gompertz_baseline import fit_age_baseline, inverse_softplus
from metageformer_torch.models import DeepGompertzMetabolomicAgeHead, deep_gompertz_nll_loss
from finetune.deep_gompertz.data import (
    GompertzSplitData,
    load_gompertz_loader,
    move_batch_to_device,
)
from common.paths import DEFAULT_DEEP_GOMPERTZ_DIR, DEFAULT_EMBEDDING_DATA
from utils import create_directory, save_dict_2_json, save_txt_single_column, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)


def run_epoch_deep_gompertz(
    model,
    loader,
    optimizer,
    device: str,
    args,
    training: bool,
    epoch: int,
    batch_step_offset: int = 0,
    wandb_run=None,
):
    model.train(training)
    losses = []
    risk_collect = []
    event_collect = []
    time_collect = []
    desc = "Training" if training else "Validation"
    phase = "train" if training else "val"

    for batch_idx, batch in enumerate(progress_iter(loader, desc=desc, total=len(loader))):
        embedding, age, event, time = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(training):
            outputs = model(embedding, age)
            loss = deep_gompertz_nll_loss(
                outputs["alpha_i"],
                outputs["gamma_i"],
                outputs["log_age_effect"],
                time,
                event,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        risk_collect.append(outputs["mortality_risk_10y"].detach().reshape(-1).cpu())
        event_collect.append(event.detach().cpu())
        time_collect.append(time.detach().cpu())

        if wandb_run is not None:
            global_step = args.baseline_epoch + epoch * args.steps_per_epoch + batch_step_offset + batch_idx
            wandb_run.log(
                {
                    "step/epoch": epoch,
                    "step/batch_idx": batch_idx,
                    f"step/{phase}_loss": loss_value,
                    f"step/{phase}_batch_size": int(event.numel()),
                    f"step/{phase}_event_rate": float(event.float().mean().detach().cpu()),
                    f"step/{phase}_time_mean": float(time.float().mean().detach().cpu()),
                    f"step/{phase}_alpha_mean": float(outputs["alpha_i"].mean().detach().cpu()),
                    f"step/{phase}_gamma_mean": float(outputs["gamma_i"].mean().detach().cpu()),
                    "step/lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

    risk_collect = torch.cat(risk_collect, dim=0)
    event_collect = torch.cat(event_collect, dim=0)
    time_collect = torch.cat(time_collect, dim=0)
    mean_loss = float(np.mean(losses))
    cindex = safe_cindex(risk_collect, event_collect, time_collect)
    return mean_loss, cindex


def save_checkpoint_deep_gompertz(
    model,
    args,
    save_dir: str,
    baseline_params: Dict[str, float],
    best_epoch: int,
    baseline_loss_epoch: list,
    baseline_val_loss_epoch: list,
    train_loss_epoch: list,
    val_loss_epoch: list,
    train_cidx_epoch: list,
    val_cidx_epoch: list,
):
    config = model.config_dict()
    config.update(
        {
            "embedding_key": args.embedding_key,
            "age_col": args.age_col,
            "event_col": args.event_col,
            "time_col": args.time_col,
            "best_epoch": best_epoch,
        }
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "baseline_params": baseline_params,
        },
        os.path.join(save_dir, "model_weights.pth"),
    )
    save_dict_2_json(config, "config.json", save_dir)
    save_dict_2_json(baseline_params, "baseline_gompertz.json", save_dir)
    save_txt_single_column(baseline_loss_epoch, save_dir=save_dir, filename="baseline_train_loss.txt")
    if baseline_val_loss_epoch:
        save_txt_single_column(baseline_val_loss_epoch, save_dir=save_dir, filename="baseline_val_loss.txt")
    save_txt_single_column(train_loss_epoch, save_dir=save_dir, filename="train_loss.txt")
    save_txt_single_column(val_loss_epoch, save_dir=save_dir, filename="val_loss.txt")
    save_txt_single_column(train_cidx_epoch, save_dir=save_dir, filename="train_cidx.txt")
    save_txt_single_column(val_cidx_epoch, save_dir=save_dir, filename="val_cidx.txt")


def load_best_checkpoint(model, save_dir: str, device: str):
    checkpoint_path = os.path.join(save_dir, "model_weights.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return checkpoint


def train(
    model,
    train_loader,
    val_loader,
    device: str,
    args,
    baseline_params: Dict[str, float],
    baseline_loss_epoch: list,
    baseline_val_loss_epoch: list,
    wandb_run=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loss_epoch = []
    train_cidx_epoch = []
    val_loss_epoch = []
    val_cidx_epoch = []

    best_val_loss = float("inf")
    best_epoch = -1
    watchdog = 0
    args.steps_per_epoch = len(train_loader) + len(val_loader)

    for epoch in range(args.n_epoch):
        train_loss, train_cidx = run_epoch_deep_gompertz(
            model,
            train_loader,
            optimizer,
            device,
            args,
            training=True,
            epoch=epoch,
            batch_step_offset=0,
            wandb_run=wandb_run,
        )
        val_loss, val_cidx = run_epoch_deep_gompertz(
            model,
            val_loader,
            optimizer,
            device,
            args,
            training=False,
            epoch=epoch,
            batch_step_offset=len(train_loader),
            wandb_run=wandb_run,
        )

        train_loss_epoch.append(train_loss)
        train_cidx_epoch.append(train_cidx)
        val_loss_epoch.append(val_loss)
        val_cidx_epoch.append(val_cidx)

        print(
            f"epoch: {epoch}; train loss: {train_loss}, C-index: {train_cidx}; "
            f"val loss: {val_loss}, C-index: {val_cidx}"
        )

        watchdog += 1
        checkpoint_saved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            watchdog = 0
            checkpoint_saved = True
            save_checkpoint_deep_gompertz(
                model,
                args,
                args.save_dir,
                baseline_params,
                best_epoch,
                baseline_loss_epoch,
                baseline_val_loss_epoch,
                train_loss_epoch,
                val_loss_epoch,
                train_cidx_epoch,
                val_cidx_epoch,
            )

        if wandb_run is not None:
            with torch.no_grad():
                embedding, age, _, _ = next(iter(val_loader))
                sample_outputs = model(embedding.to(device), age.to(device))

            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/cindex": train_cidx,
                    "val/loss": val_loss,
                    "val/cindex": val_cidx,
                    "best/val_loss": best_val_loss,
                    "best/epoch": best_epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "checkpoint/saved": int(checkpoint_saved),
                    "early_stop/watchdog": watchdog,
                    "model/beta_age": float(model.beta_age.detach().cpu()),
                    "model/val_alpha_mean": float(sample_outputs["alpha_i"].mean().detach().cpu()),
                    "model/val_gamma_mean": float(sample_outputs["gamma_i"].mean().detach().cpu()),
                },
                step=args.baseline_epoch + (epoch + 1) * args.steps_per_epoch,
            )

        if isinstance(args.n_toler, int) and args.n_toler > 0 and watchdog >= args.n_toler:
            print(
                f"early stop at epoch {epoch}; best epoch: {best_epoch}; best val loss: {best_val_loss}",
                flush=True,
            )
            break
        torch.cuda.empty_cache()

    # Persist full epoch curves (including post-best patience epochs). Best weights stay as last improve save.
    save_txt_single_column(train_loss_epoch, save_dir=args.save_dir, filename="train_loss.txt")
    save_txt_single_column(val_loss_epoch, save_dir=args.save_dir, filename="val_loss.txt")
    save_txt_single_column(train_cidx_epoch, save_dir=args.save_dir, filename="train_cidx.txt")
    save_txt_single_column(val_cidx_epoch, save_dir=args.save_dir, filename="val_cidx.txt")

    if not os.path.exists(os.path.join(args.save_dir, "model_weights.pth")):
        raise FileNotFoundError("Training finished without saving a checkpoint.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a DeepGompertz metabolomic age head.")
    parser.add_argument("--data_path", type=str, default="../Data/Blood_dataset_fullcohort_107nonderived_mlm")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_DEEP_GOMPERTZ_DIR)
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
    parser.add_argument("--age_scale", type=float, default=1.0, help="Deprecated; ignored. Ages are always in years.")
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--init_alpha", type=float, default=-10.5)
    parser.add_argument("--init_gamma", type=float, default=0.09)
    parser.add_argument("--init_beta_age", type=float, default=0.1)
    parser.add_argument("--embedding_key", type=str, default=EMBEDDING_KEY)
    parser.add_argument("--age_col", type=str, default=AGE_COL)
    parser.add_argument("--event_col", type=str, default=EVENT_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument("--debug_n_train", type=int, default=None)
    parser.add_argument("--debug_n_val", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="MetAgeFormer_DeepGompertz")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
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

    train_loader, train_data = load_gompertz_loader(
        args.data_path,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        age_scale=args.age_scale,
        embedding_key=args.embedding_key,
        age_col=args.age_col,
        event_col=args.event_col,
        time_col=args.time_col,
        debug_n=args.debug_n_train,
        random_seed=args.seed,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader, val_data = load_gompertz_loader(
        args.data_path,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        age_scale=args.age_scale,
        embedding_key=args.embedding_key,
        age_col=args.age_col,
        event_col=args.event_col,
        time_col=args.time_col,
        debug_n=args.debug_n_val,
        random_seed=args.seed + 1,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"train samples: {train_data.n_samples}, val samples: {val_data.n_samples}")

    embedding_dim = train_data.embedding_dim
    run_config = vars(args).copy()
    run_config["embedding_dim"] = int(embedding_dim)
    save_dict_2_json(run_config, "run_config.json", args.save_dir)
    wandb_run = init_wandb(args, run_config)

    baseline_params, baseline_loss_epoch, baseline_val_loss_epoch = fit_age_baseline(
        train_data, val_data, device, args, wandb_run=wandb_run
    )

    model = DeepGompertzMetabolomicAgeHead(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        t_window=args.t_window,
        age_scale=args.age_scale,
        baseline_params=baseline_params,
        init_alpha=args.init_alpha,
        init_gamma=args.init_gamma,
        init_beta_age=args.init_beta_age,
        gamma_min=args.gamma_min,
    ).to(device)

    train(
        model,
        train_loader,
        val_loader,
        device,
        args,
        baseline_params,
        baseline_loss_epoch,
        baseline_val_loss_epoch,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

