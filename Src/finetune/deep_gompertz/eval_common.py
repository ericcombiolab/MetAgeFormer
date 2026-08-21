"""Shared MetAgeFormer + DeepGompertz inference utilities."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from metageformer_torch.checkpoint import load_pretrained_checkpoint
from metageformer_torch.models import DeepGompertzEndToEndModel
from utils import load_tokenizer


OUTPUT_KEYS = [
    "linear_predictor",
    "log_age_effect",
    "alpha_i",
    "gamma_i",
    "mortality_risk_10y",
    "metabolomic_age",
    "age_gap",
]


class _IndexDataset(Dataset):
    def __init__(self, n_samples: int):
        self.n_samples = int(n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return int(index)


def _collate_indices(indices):
    return np.asarray(indices, dtype=np.int64)


class NMRInferenceCache:
    """Tensor-ready concentrations and labels without per-batch AnnData concat."""

    def __init__(self, adata: ad.AnnData, tokenizer, data_layer: str, age_col: str):
        if data_layer not in adata.layers:
            raise KeyError(f"Missing layer '{data_layer}' in AnnData")
        if age_col not in adata.obs:
            raise KeyError(f"Missing age column '{age_col}' in AnnData")

        self.concentration = np.ascontiguousarray(
            adata.layers[data_layer], dtype=np.float32
        )
        self.masking_mask = np.ascontiguousarray(
            np.isnan(self.concentration) | (self.concentration == 0),
            dtype=np.int64,
        )
        self.identifier_ids = np.asarray(
            [tokenizer.token_to_id_iden(name) for name in adata.var_names],
            dtype=np.int64,
        )
        self.age = np.ascontiguousarray(
            adata.obs[age_col].to_numpy(), dtype=np.float32
        )
        self.sample_ids = adata.obs_names.to_numpy()
        self.n_samples = int(adata.n_obs)

    def get_batch(self, indices: np.ndarray, device: str):
        concentration = torch.from_numpy(self.concentration[indices]).to(
            device, non_blocking=True
        )
        masking_mask = torch.from_numpy(self.masking_mask[indices]).to(
            device, non_blocking=True
        )
        identifiers = (
            torch.from_numpy(self.identifier_ids)
            .unsqueeze(0)
            .expand(len(indices), -1)
            .to(device, non_blocking=True)
        )
        inputs = {
            "input_ids": {
                "identifier": identifiers.long(),
                "concentration": concentration.float(),
            },
            "masking_mask": masking_mask.long(),
            "padding_mask": torch.zeros_like(masking_mask).long(),
        }
        age = torch.from_numpy(self.age[indices]).to(device, non_blocking=True)
        return inputs, age.float()


def _backbone_config(config: Dict) -> Dict:
    return {
        "n_heads": int(config["n_heads"]),
        "n_blocks": int(config["n_blocks"]),
        "d_ff": int(config["d_ff"]),
        "d_model": int(config["d_model"]),
        "dropout": float(config.get("dropout", config.get("drop_out", 0.1))),
        "activation": config.get("activation", config.get("f_act", "relu")),
        "need_weights": False,
        "average_attn_weights": False,
        "attn_mode": str(config.get("attn_mode", "mixdirect_mask")),
    }


def load_model(
    pretrained_dir: str, model_dir: str, device: str
) -> Tuple[DeepGompertzEndToEndModel, object, Dict, Dict]:
    """Load pretrained backbone + DeepGompertz head as one inference model."""
    tokenizer_path = os.path.join(pretrained_dir, "tokenizer.pkl")
    pretrained_config_path = os.path.join(pretrained_dir, "config.json")
    pretrained_weights_path = os.path.join(pretrained_dir, "model_weights.pth")
    head_weights_path = os.path.join(model_dir, "model_weights.pth")
    for path in (
        tokenizer_path,
        pretrained_config_path,
        pretrained_weights_path,
        head_weights_path,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model asset does not exist: {path}")

    tokenizer = load_tokenizer(tokenizer_path)
    with open(pretrained_config_path, "r") as file:
        pretrained_config = json.load(file)
    backbone_config = _backbone_config(pretrained_config)
    backbone_checkpoint = load_pretrained_checkpoint(pretrained_weights_path)

    head_checkpoint = torch.load(head_weights_path, map_location="cpu")
    head_config = head_checkpoint["config"]
    baseline_params = head_checkpoint.get("baseline_params") or {
        "alpha_age_scale": head_config["alpha_age_scale"],
        "gamma_age_scale": head_config["gamma_age_scale"],
    }

    model = DeepGompertzEndToEndModel(
        embedding_module_conf={
            "n_vocabs": {"identifier": tokenizer.vocab_size_identifiers}
        },
        model_conf=backbone_config,
        baseline_params=baseline_params,
        hidden_dim=int(head_config["hidden_dim"]),
        dropout=float(head_config["dropout"]),
        t_window=float(head_config["t_window"]),
        gamma_min=float(head_config.get("gamma_min", 1e-6)),
    )
    model.metageformer_model.load_state_dict(backbone_checkpoint["METAGEFORMER"])
    model.head.load_state_dict(head_checkpoint["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, tokenizer, pretrained_config, head_config


def build_index_loader(
    n_samples: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": _collate_indices,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(_IndexDataset(n_samples), **kwargs)


def predict(
    model,
    tokenizer,
    adata: ad.AnnData,
    data_layer: str,
    age_col: str,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    id_col: str = "sample_id",
    save_embeddings: bool = False,
):
    """Run backbone + DeepGompertz head and return predictions."""
    cache = NMRInferenceCache(adata, tokenizer, data_layer, age_col)
    loader = build_index_loader(
        cache.n_samples, batch_size, num_workers, prefetch_factor
    )
    collected = {key: [] for key in OUTPUT_KEYS}
    ages = []
    embeddings = [] if save_embeddings else None

    for batch_index, indices in enumerate(loader):
        inputs, age = cache.get_batch(indices, device)
        with torch.inference_mode():
            inputs = model.generate_mixdirect_mask(inputs)
            hidden, _ = model.metageformer_model(inputs)
            cls = hidden[:, 0, :]
            outputs = model.head(cls, age)

        for key in OUTPUT_KEYS:
            collected[key].append(outputs[key].reshape(-1).cpu())
        ages.append(age.reshape(-1).cpu())
        if embeddings is not None:
            embeddings.append(cls.cpu())
        if batch_index == 0 or (batch_index + 1) % 50 == 0:
            done = min((batch_index + 1) * batch_size, cache.n_samples)
            print(f"Inference: {done}/{cache.n_samples}", flush=True)

    prediction = pd.DataFrame(
        {
            id_col: cache.sample_ids,
            "chronological_age": torch.cat(ages).numpy(),
            **{
                key: torch.cat(values).numpy()
                for key, values in collected.items()
            },
        }
    )
    embedding_array: Optional[np.ndarray] = None
    if embeddings is not None:
        embedding_array = torch.cat(embeddings).numpy()
    return prediction, embedding_array
