"""Tensor-based data loading for Gompertz finetune/eval (replaces AnnData per-batch concat)."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class GompertzSplitData:
    embedding: np.ndarray
    age: np.ndarray
    event: np.ndarray
    time: np.ndarray
    sample_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.shape[1])


def subset_split_data(data: GompertzSplitData, n_samples: Optional[int], random_seed: int) -> GompertzSplitData:
    if not isinstance(n_samples, int) or n_samples <= 0 or n_samples >= data.n_samples:
        return data

    rng = np.random.default_rng(random_seed)
    indices = rng.choice(data.n_samples, size=n_samples, replace=False)
    return GompertzSplitData(
        embedding=data.embedding[indices],
        age=data.age[indices],
        event=data.event[indices],
        time=data.time[indices],
        sample_ids=data.sample_ids[indices],
    )


def load_gompertz_split(
    data_path: str,
    split: str,
    embedding_key: str,
    age_col: str,
    event_col: str,
    time_col: str,
    debug_n: Optional[int] = None,
    random_seed: int = 3047,
) -> GompertzSplitData:
    h5ad_path = os.path.join(data_path, f"{split}.h5ad")
    adata = ad.read_h5ad(h5ad_path)

    missing_obs = [col for col in [age_col, event_col, time_col] if col not in adata.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing required obs columns: {missing_obs}")
    if embedding_key not in adata.obsm.keys():
        raise KeyError(f"Missing required obsm key: {embedding_key}")

    data = GompertzSplitData(
        embedding=np.ascontiguousarray(adata.obsm[embedding_key], dtype=np.float32),
        age=np.ascontiguousarray(adata.obs[age_col].to_numpy(), dtype=np.float32),
        event=np.ascontiguousarray(adata.obs[event_col].to_numpy(), dtype=np.float32),
        time=np.ascontiguousarray(adata.obs[time_col].to_numpy(), dtype=np.float32),
        sample_ids=adata.obs_names.to_numpy(),
    )
    return subset_split_data(data, debug_n, random_seed=random_seed)


def build_gompertz_loader(
    data: GompertzSplitData,
    batch_size: int,
    shuffle: bool,
    age_scale: float,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    # Age is always passed downstream in calendar years. Keep age_scale in the
    # signature only for older call sites; it must not change the mathematical unit.
    age_years = np.ascontiguousarray(data.age, dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(data.embedding),
        torch.from_numpy(age_years),
        torch.from_numpy(data.event),
        torch.from_numpy(data.time),
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def load_gompertz_loader(
    data_path: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    age_scale: float,
    embedding_key: str,
    age_col: str,
    event_col: str,
    time_col: str,
    debug_n: Optional[int] = None,
    random_seed: int = 3047,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, GompertzSplitData]:
    data = load_gompertz_split(
        data_path=data_path,
        split=split,
        embedding_key=embedding_key,
        age_col=age_col,
        event_col=event_col,
        time_col=time_col,
        debug_n=debug_n,
        random_seed=random_seed,
    )
    loader = build_gompertz_loader(
        data=data,
        batch_size=batch_size,
        shuffle=shuffle,
        age_scale=age_scale,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return loader, data


def move_batch_to_device(batch, device: str):
    embedding, age, event, time = batch
    return (
        embedding.to(device, non_blocking=True),
        age.to(device, non_blocking=True),
        event.to(device, non_blocking=True),
        time.to(device, non_blocking=True),
    )

