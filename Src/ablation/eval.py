import argparse
import os
import pickle
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex

from ablation.train_fully_finetune import align_adata_metabolites
from ablation.train_from_scratch import (
    NMRSurvivalTensorCache,
    build_index_loader,
    build_tensor_cache,
    load_split_adata,
    tokenize_survival_batch,
)
from common.constants import AGE_COL, EVENT_COL, TIME_COL
from common.paths import EVAL_DEEP_GOMPERTZ_ROOT, FINETUNED_DEEP_GOMPERTZ_ROOT
from common.training import progress_iter
from metageformer_torch.models import DeepGompertzEndToEndModel, DeepGompertzFullyFinetuneModel
from utils import create_directory, load_tokenizer, set_seeds


warnings.filterwarnings("ignore", category=UserWarning)


def load_ablation_e2e_model(model_dir: str, device: str) -> Tuple[torch.nn.Module, dict]:
    checkpoint_path = os.path.join(model_dir, "model_weights.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"The ablation checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    backbone_config = checkpoint["backbone_config"]
    baseline_params = checkpoint.get("baseline_params")
    if baseline_params is None:
        baseline_params = {
            "alpha_age_scale": config["alpha_age_scale"],
            "gamma_age_scale": config["gamma_age_scale"],
        }

    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing tokenizer in model_dir: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    embedding_module_conf = {
        "n_vocabs": {"identifier": tokenizer.vocab_size_identifiers},
    }
    model_kwargs = {
        "embedding_module_conf": embedding_module_conf,
        "model_conf": backbone_config,
        "baseline_params": baseline_params,
        "hidden_dim": config.get("hidden_dim", 64),
        "dropout": config.get("dropout", 0.1),
        "t_window": config.get("t_window", 10.0),
        "gamma_min": config.get("gamma_min", 1e-6),
    }

    model_type = config.get("model_type", "DeepGompertzEndToEnd")
    if model_type == "DeepGompertzFullyFinetune":
        model = DeepGompertzFullyFinetuneModel(
            pretrained_dir=config.get("pretrained_dir", ""),
            **model_kwargs,
        )
    else:
        model = DeepGompertzEndToEndModel(**model_kwargs)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model_type={model_type}, training_mode={config.get('training_mode')}", flush=True)
    return model, config


def prepare_split_cache(
    data_path: str,
    split: str,
    tokenizer,
    data_layer: str,
    age_col: str,
    event_col: str,
    time_col: str,
    align_to_tokenizer: bool,
    debug_n: Optional[int],
    random_seed: int,
) -> Tuple[NMRSurvivalTensorCache, np.ndarray]:
    adata = load_split_adata(
        data_path,
        split,
        age_col,
        event_col,
        time_col,
        data_layer,
        debug_n=debug_n,
        random_seed=random_seed,
    )
    if align_to_tokenizer:
        adata = align_adata_metabolites(adata, list(tokenizer.iden_tokens))
    sample_ids = adata.obs_names.to_numpy()
    cache = build_tensor_cache(
        adata,
        tokenizer,
        data_layer,
        age_col,
        event_col,
        time_col,
    )
    return cache, sample_ids


def forward_e2e_with_embedding(model: torch.nn.Module, inputs, age):
    inputs = model.generate_mixdirect_mask(inputs)
    hidden, _ = model.metageformer_model(inputs)
    embedding = hidden[:, 0, :]
    outputs = model.head(embedding, age)
    return outputs, embedding


def evaluate_split_ablation(
    model: torch.nn.Module,
    cache: NMRSurvivalTensorCache,
    sample_ids: np.ndarray,
    loader,
    split: str,
    device: str,
):
    linear_predictor_collect = []
    log_age_effect_collect = []
    alpha_collect = []
    gamma_collect = []
    mortality_risk_collect = []
    metabolomic_age_collect = []
    age_gap_collect = []
    event_collect = []
    time_collect = []
    age_collect = []
    embedding_collect = []

    for batch_idx, indices in enumerate(
        progress_iter(loader, desc=f"Evaluating {split}", total=len(loader))
    ):
        if batch_idx == 0:
            print(f"{split}: starting batch 0/{len(loader)}", flush=True)

        inputs, age, event, surv_time = tokenize_survival_batch(cache, indices, device)
        with torch.no_grad():
            outputs, embedding = forward_e2e_with_embedding(model, inputs, age)

        embedding_collect.append(embedding.cpu())

        linear_predictor_collect.append(outputs["linear_predictor"].reshape(-1).cpu())
        log_age_effect_collect.append(outputs["log_age_effect"].reshape(-1).cpu())
        alpha_collect.append(outputs["alpha_i"].reshape(-1).cpu())
        gamma_collect.append(outputs["gamma_i"].reshape(-1).cpu())
        mortality_risk_collect.append(outputs["mortality_risk_10y"].reshape(-1).cpu())
        metabolomic_age_collect.append(outputs["metabolomic_age"].reshape(-1).cpu())
        age_gap_collect.append(outputs["age_gap"].reshape(-1).cpu())
        event_collect.append(event.reshape(-1).cpu())
        time_collect.append(surv_time.reshape(-1).cpu())
        age_collect.append(age.reshape(-1).cpu())

    linear_predictor = torch.cat(linear_predictor_collect, dim=0)
    log_age_effect = torch.cat(log_age_effect_collect, dim=0)
    alpha_i = torch.cat(alpha_collect, dim=0)
    gamma_i = torch.cat(gamma_collect, dim=0)
    mortality_risk = torch.cat(mortality_risk_collect, dim=0)
    metabolomic_age = torch.cat(metabolomic_age_collect, dim=0)
    age_gap = torch.cat(age_gap_collect, dim=0)
    event = torch.cat(event_collect, dim=0)
    surv_time = torch.cat(time_collect, dim=0)
    chronological_age = torch.cat(age_collect, dim=0)
    embeddings = torch.cat(embedding_collect, dim=0).numpy()

    if len(sample_ids) != int(mortality_risk.shape[0]):
        raise RuntimeError(
            f"Sample count mismatch in {split}: ids={len(sample_ids)}, preds={mortality_risk.shape[0]}"
        )

    prediction = pd.DataFrame(
        {
            "eid": sample_ids,
            "chronological_age": chronological_age.numpy(),
            "linear_predictor": linear_predictor.numpy(),
            "log_age_effect": log_age_effect.numpy(),
            "alpha_i": alpha_i.numpy(),
            "gamma_i": gamma_i.numpy(),
            "mortality_risk_10y": mortality_risk.numpy(),
            "metabolomic_age": metabolomic_age.numpy(),
            "age_gap": age_gap.numpy(),
            "event": event.numpy(),
            "time": surv_time.numpy(),
        }
    )
    return prediction, mortality_risk, event, surv_time, embeddings


def save_embeddings(save_dir: str, split: str, embeddings: np.ndarray, sample_ids: np.ndarray):
    embedding_path = os.path.join(save_dir, f"embeddings_{split}.pkl")
    emb_dict = {
        "sample": embeddings,
        "eid": sample_ids,
    }
    with open(embedding_path, "wb") as file:
        pickle.dump(emb_dict, file)
    print(f"Saved embeddings to {embedding_path} with shape {embeddings.shape}", flush=True)


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
        description="Evaluate ablation DeepGompertz end-to-end models on raw NMR data."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(FINETUNED_DEEP_GOMPERTZ_ROOT, "fully_finetune_107nonderived"),
    )
    parser.add_argument("--data_path", type=str, default="../Data/NMR_dataset_fullcohort_107nonderived")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(EVAL_DEEP_GOMPERTZ_ROOT, "fully_finetune_107nonderived"),
    )
    parser.add_argument("--data_layer", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--auc_time", type=float, default=10.0)
    parser.add_argument("--age_col", type=str, default=AGE_COL)
    parser.add_argument("--event_col", type=str, default=EVENT_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument("--debug_n_test", type=int, default=None)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to evaluate, e.g. test",
    )
    parser.add_argument(
        "--save_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save CLS sample embeddings to embeddings_{split}.pkl (off by default; mainline eval does not export embeddings).",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_directory(args.save_dir)
    set_seeds(args.seed)
    num_workers = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print(f"DataLoader num_workers: {num_workers}", flush=True)

    model, config = load_ablation_e2e_model(args.model_dir, device)

    tokenizer = load_tokenizer(os.path.join(args.model_dir, "tokenizer.pkl"))
    args.age_col = config.get("age_col", args.age_col)
    args.event_col = config.get("event_col", args.event_col)
    args.time_col = config.get("time_col", args.time_col)
    data_layer = args.data_layer or config.get("data_layer", "Z-score normalized")
    align_to_tokenizer = config.get("model_type") == "DeepGompertzFullyFinetune"
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    for split in splits:
        split_t0 = time.perf_counter()
        cache, sample_ids = prepare_split_cache(
            args.data_path,
            split,
            tokenizer,
            data_layer,
            args.age_col,
            args.event_col,
            args.time_col,
            align_to_tokenizer=align_to_tokenizer,
            debug_n=args.debug_n_test if split == "test" else None,
            random_seed=args.seed,
        )
        loader = build_index_loader(
            cache.n_samples,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        print(f"{split} samples: {cache.n_samples}, batches: {len(loader)}", flush=True)

        prediction, risk, event, surv_time, embeddings = evaluate_split_ablation(
            model,
            cache,
            sample_ids,
            loader,
            split,
            device,
        )
        prediction.to_csv(os.path.join(args.save_dir, f"prediction_{split}.csv"), index=False)
        if args.save_embeddings:
            save_embeddings(args.save_dir, split, embeddings, sample_ids)
        print(f"{split} eval done in {time.perf_counter() - split_t0:.1f}s", flush=True)
        if split == "test":
            save_metrics(args.save_dir, risk, event, surv_time, args.auc_time)
