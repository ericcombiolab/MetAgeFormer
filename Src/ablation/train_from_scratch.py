import argparse
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from metageformer_torch.models import (
    DeepGompertzEndToEndModel,
    deep_gompertz_nll_loss,
)
from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.gompertz_baseline import AgeOnlyGompertzBaseline, evaluate_age_baseline, inverse_softplus
from common.paths import FINETUNED_DEEP_GOMPERTZ_ROOT
from common.training import init_wandb, progress_iter, resolve_num_workers, safe_cindex, str2bool
from metageformer_torch.tokenizer import MetAgeFormer_Tokenizer
from utils import create_directory, save_dict_2_json, save_txt_single_column, save_tokenizer, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)


class NMRSurvivalTensorCache:
    """Pre-extract NMR concentrations and survival labels to avoid per-batch AnnData concat."""

    def __init__(
        self,
        adata: ad.AnnData,
        data_layer: str,
        iden_ids: np.ndarray,
        age_col: str,
        event_col: str,
        time_col: str,
    ):
        validate_survival_columns(adata, age_col, event_col, time_col, "adata")
        if data_layer not in adata.layers.keys():
            raise KeyError(f"Missing required layer: {data_layer}")

        self.concentration = np.ascontiguousarray(adata.layers[data_layer], dtype=np.float32)
        self.iden_ids = np.ascontiguousarray(iden_ids, dtype=np.int64)
        self.masking_mask = (
            np.isnan(self.concentration) | (self.concentration == 0)
        ).astype(np.int64)
        self.age, self.event, self.time = collect_survival_arrays_from_scratch(adata, age_col, event_col, time_col)
        self.n_samples = int(self.concentration.shape[0])
        self.n_vars = int(self.concentration.shape[1])

    def get_batch(self, indices: np.ndarray, device: str):
        conc = torch.from_numpy(self.concentration[indices]).to(device, non_blocking=True)
        iden = (
            torch.from_numpy(self.iden_ids)
            .unsqueeze(0)
            .expand(len(indices), -1)
            .to(device, non_blocking=True)
        )
        mask = torch.from_numpy(self.masking_mask[indices]).to(device, non_blocking=True)
        padding = torch.zeros_like(mask)
        inputs = {
            "input_ids": {"identifier": iden.long(), "concentration": conc.float()},
            "masking_mask": mask.long(),
            "padding_mask": padding.long(),
        }
        age = torch.from_numpy(self.age[indices]).to(device, non_blocking=True)
        event = torch.from_numpy(self.event[indices]).to(device, non_blocking=True)
        surv_time = torch.from_numpy(self.time[indices]).to(device, non_blocking=True)
        return inputs, age.float(), event.float(), surv_time.float()


class IndexDataset(Dataset):
    def __init__(self, n_samples: int):
        self.indices = np.arange(n_samples, dtype=np.int64)

    def __len__(self) -> int:
        return self.n_samples

    @property
    def n_samples(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


def collate_indices(batch: List[int]) -> np.ndarray:
    return np.asarray(batch, dtype=np.int64)


def build_index_loader(
    n_samples: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = IndexDataset(n_samples)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_indices,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_tensor_cache(
    adata: ad.AnnData,
    tokenizer: MetAgeFormer_Tokenizer,
    data_layer: str,
    age_col: str,
    event_col: str,
    time_col: str,
) -> NMRSurvivalTensorCache:
    iden_ids = np.array(
        [tokenizer.token_to_id_iden(token) for token in adata.var_names],
        dtype=np.int64,
    )
    return NMRSurvivalTensorCache(
        adata=adata,
        data_layer=data_layer,
        iden_ids=iden_ids,
        age_col=age_col,
        event_col=event_col,
        time_col=time_col,
    )


def subset_adata(adata: ad.AnnData, n_samples: Optional[int], random_seed: int) -> ad.AnnData:
    if not isinstance(n_samples, int) or n_samples <= 0 or n_samples >= adata.n_obs:
        return adata
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(adata.n_obs, size=n_samples, replace=False)
    return adata[indices].copy()


def validate_survival_columns(adata: ad.AnnData, age_col: str, event_col: str, time_col: str, split_name: str):
    missing = [col for col in [age_col, event_col, time_col] if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns in {split_name}: {missing}")


def load_split_adata(
    data_path: str,
    split: str,
    age_col: str,
    event_col: str,
    time_col: str,
    data_layer: str,
    debug_n: Optional[int] = None,
    random_seed: int = 3047,
) -> ad.AnnData:
    h5ad_path = os.path.join(data_path, f"{split}.h5ad")
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"Split file does not exist: {h5ad_path}")

    use_subset = isinstance(debug_n, int) and debug_n > 0
    if use_subset:
        adata_backed = ad.read_h5ad(h5ad_path, backed="r")
        validate_survival_columns(adata_backed, age_col, event_col, time_col, split)
        if data_layer not in adata_backed.layers.keys():
            raise KeyError(f"Missing required layer in {split}: {data_layer}")
        if debug_n < adata_backed.n_obs:
            rng = np.random.default_rng(random_seed)
            indices = np.sort(rng.choice(adata_backed.n_obs, size=debug_n, replace=False))
            adata = adata_backed[indices].to_memory()
        else:
            adata = adata_backed.to_memory()
        adata_backed.file.close()
        return adata

    adata = ad.read_h5ad(h5ad_path)
    validate_survival_columns(adata, age_col, event_col, time_col, split)
    if data_layer not in adata.layers.keys():
        raise KeyError(f"Missing required layer in {split}: {data_layer}")
    return adata


def collect_survival_arrays_from_scratch(
    adata: ad.AnnData,
    age_col: str,
    event_col: str,
    time_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_survival_columns(adata, age_col, event_col, time_col, "adata")
    age = np.ascontiguousarray(adata.obs[age_col].to_numpy(), dtype=np.float32)
    event = np.ascontiguousarray(adata.obs[event_col].to_numpy(), dtype=np.float32)
    time = np.ascontiguousarray(adata.obs[time_col].to_numpy(), dtype=np.float32)
    return age, event, time


def tokenize_survival_batch(
    cache: NMRSurvivalTensorCache,
    indices: np.ndarray,
    device: str,
):
    return cache.get_batch(indices, device)


def fit_age_baseline_from_adata_from_scratch(
    train_cache: NMRSurvivalTensorCache,
    val_cache: NMRSurvivalTensorCache,
    device: str,
    args,
    wandb_run=None,
) -> Tuple[Dict[str, float], list, list]:
    model = AgeOnlyGompertzBaseline(init_alpha=args.init_alpha, init_gamma=args.init_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.baseline_lr, weight_decay=args.weight_decay)

    train_age = torch.from_numpy(train_cache.age).to(device)
    train_event = torch.from_numpy(train_cache.event).to(device)
    train_time = torch.from_numpy(train_cache.time).to(device)
    val_age = torch.from_numpy(val_cache.age).to(device)
    val_event = torch.from_numpy(val_cache.event).to(device)
    val_time = torch.from_numpy(val_cache.time).to(device)

    train_loss_epoch = []
    val_loss_epoch = []
    best_val_loss = float("inf")
    best_params = model.params_dict()
    watchdog = 0

    for epoch in range(args.baseline_epoch):
        model.train()
        optimizer.zero_grad()
        train_loss = model(train_age, train_time, train_event)
        train_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss_value = float(train_loss.detach().cpu())
        val_loss_value = evaluate_age_baseline(model, val_age, val_time, val_event)
        train_loss_epoch.append(train_loss_value)
        val_loss_epoch.append(val_loss_value)
        params = model.params_dict()

        print(
            f"baseline epoch: {epoch}; train loss: {train_loss_value}; "
            f"val loss: {val_loss_value}; params: {params}"
        )

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_params = params
            watchdog = 0
        else:
            watchdog += 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "baseline/train_loss": train_loss_value,
                    "baseline/val_loss": val_loss_value,
                    "baseline/alpha_age_scale": params["alpha_age_scale"],
                    "baseline/gamma_age_scale": params["gamma_age_scale"],
                    "baseline/best_val_loss": best_val_loss,
                    "baseline/early_stop_watchdog": watchdog,
                },
                step=epoch,
            )

        if isinstance(args.baseline_n_toler, int) and args.baseline_n_toler > 0 and watchdog >= args.baseline_n_toler:
            print(f"baseline early stop at epoch {epoch}; best val loss: {best_val_loss}")
            break

    with torch.no_grad():
        model.alpha.copy_(torch.tensor(best_params["alpha_age_scale"], device=device))
        inv_gamma = inverse_softplus(best_params["gamma_age_scale"])
        model.raw_gamma.copy_(torch.tensor(inv_gamma, device=device))

    return best_params, train_loss_epoch, val_loss_epoch


def run_epoch_from_scratch(
    model: DeepGompertzEndToEndModel,
    loader,
    cache: NMRSurvivalTensorCache,
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

    for batch_idx, indices in enumerate(progress_iter(loader, desc=desc, total=len(loader))):
        if batch_idx == 0:
            print(f"{phase} epoch {epoch}: starting batch 0/{len(loader)}", flush=True)
        inputs, age, event, time = tokenize_survival_batch(cache, indices, device)

        with torch.set_grad_enabled(training):
            outputs = model(inputs, age)
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

        if args.log_batch_every > 0 and batch_idx % args.log_batch_every == 0:
            print(
                f"{phase} epoch {epoch} batch {batch_idx}/{len(loader)} loss={loss_value:.6f}",
                flush=True,
            )

        if wandb_run is not None and args.wandb_log_every > 0 and batch_idx % args.wandb_log_every == 0:
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


def save_checkpoint_from_scratch(
    model: DeepGompertzEndToEndModel,
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
            "data_path": args.data_path,
            "data_layer": args.data_layer,
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
            "backbone_config": model.backbone_config,
            "baseline_params": baseline_params,
        },
        os.path.join(save_dir, "model_weights.pth"),
    )
    save_dict_2_json(config, "config.json", save_dir)
    save_dict_2_json(model.backbone_config, "backbone_config.json", save_dir)
    save_dict_2_json(baseline_params, "baseline_gompertz.json", save_dir)
    save_txt_single_column(baseline_loss_epoch, save_dir=save_dir, filename="baseline_train_loss.txt")
    if baseline_val_loss_epoch:
        save_txt_single_column(baseline_val_loss_epoch, save_dir=save_dir, filename="baseline_val_loss.txt")
    save_txt_single_column(train_loss_epoch, save_dir=save_dir, filename="train_loss.txt")
    save_txt_single_column(val_loss_epoch, save_dir=save_dir, filename="val_loss.txt")
    save_txt_single_column(train_cidx_epoch, save_dir=save_dir, filename="train_cidx.txt")
    save_txt_single_column(val_cidx_epoch, save_dir=save_dir, filename="val_cidx.txt")


def train(
    model: DeepGompertzEndToEndModel,
    train_loader,
    val_loader,
    train_cache: NMRSurvivalTensorCache,
    val_cache: NMRSurvivalTensorCache,
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
        epoch_t0 = time.perf_counter()
        print(f"main training epoch {epoch}/{args.n_epoch} started", flush=True)
        train_loss, train_cidx = run_epoch_from_scratch(
            model,
            train_loader,
            train_cache,
            optimizer,
            device,
            args,
            training=True,
            epoch=epoch,
            batch_step_offset=0,
            wandb_run=wandb_run,
        )
        val_loss, val_cidx = run_epoch_from_scratch(
            model,
            val_loader,
            val_cache,
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
            f"val loss: {val_loss}, C-index: {val_cidx}; "
            f"epoch time: {time.perf_counter() - epoch_t0:.1f}s",
            flush=True,
        )

        watchdog += 1
        checkpoint_saved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            watchdog = 0
            checkpoint_saved = True
            save_checkpoint_from_scratch(
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
                sample_indices = next(iter(val_loader))
                sample_inputs, sample_age, _, _ = tokenize_survival_batch(val_cache, sample_indices, device)
                sample_outputs = model(sample_inputs, sample_age)

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
                    "model/beta_age": float(model.head.beta_age.detach().cpu()),
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
    parser = argparse.ArgumentParser(
        description="Train DeepGompertz end-to-end from scratch on raw NMR data (no pre-trained embeddings)."
    )
    parser.add_argument("--data_path", type=str, default="../Data/NMR_dataset_fullcohort_107nonderived")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(FINETUNED_DEEP_GOMPERTZ_ROOT, "from_scratch_107nonderived"),
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
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--f_act", type=str, default="relu")
    parser.add_argument("--attn_mode", type=str, default="mixdirect_mask")
    parser.add_argument("--debug_n_train", type=int, default=None)
    parser.add_argument("--debug_n_val", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="MetAgeFormer_DeepGompertz_Ablation")
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
    print(
        f"Loaded h5ad in {time.perf_counter() - load_t0:.1f}s | "
        f"train samples: {train_adata.n_obs}, val samples: {val_adata.n_obs}"
    )

    metabo_id = train_adata.var_names.values.tolist()
    tokenizer = MetAgeFormer_Tokenizer(VOCAB_Identifiers=metabo_id)
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
    backbone_config = {
        "n_heads": args.n_heads,
        "n_blocks": args.n_blocks,
        "d_ff": args.d_ff,
        "d_model": args.d_model,
        "dropout": args.drop_out,
        "activation": args.f_act,
        "need_weights": False,
        "average_attn_weights": False,
        "attn_mode": args.attn_mode,
    }
    run_config = vars(args).copy()
    run_config["embedding_dim"] = int(args.d_model)
    save_dict_2_json(run_config, "run_config.json", args.save_dir)
    wandb_run = init_wandb(args, run_config)

    baseline_params, baseline_loss_epoch, baseline_val_loss_epoch = fit_age_baseline_from_adata_from_scratch(
        train_cache,
        val_cache,
        device,
        args,
        wandb_run=wandb_run,
    )

    model = DeepGompertzEndToEndModel(
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
    ).to(device)

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

