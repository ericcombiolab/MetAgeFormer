"""Generate fake demonstration data for MetAgeFormer pipelines.

Two modes:

1. Reference-based (mirror the schema of an existing .h5ad):
       python Data/generate_fake_data.py --from <real.h5ad> --n_samples 1000
   Reads the *structure* of a real .h5ad (var/obs schema, layer names, obsm keys),
   replaces values with random numbers drawn per feature from the original
   mean/std (clipped to the original min/max), and writes `fake_<name>.h5ad`
   next to the input. No real values are copied.

2. Synthetic from scratch (no data access needed):
       python Data/generate_fake_data.py --synthetic --outdir Data/fake
   Builds demo datasets for the three pipelines:
     - Data/fake/NMR_dataset_fake/{train,val,test}.h5ad
       (107 measures, layer 'Z-score normalized', var z-score params, obs age)
     - Data/fake/deep_gompertz_fake/{train,val,test}.h5ad
       (512-dim embeddings + Gompertz-like survival obs)
     - Data/fake/Blood_dataset_fake/{train,val,test}.h5ad
       (14 blood features, layer 'Z-score normalized', embeddings + survival obs)
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

N_METABOLITES = 107
BLOOD_N_FEATURES = 14  # matches the released Lightweight model_conf n_features
EMBEDDING_DIM = 512
SEED = 3047

# Column names required by the pipelines (Src/common/constants.py)
AGE_COL = "Age at assessment (estimated)"
EVENT_COL = "Death event"
TIME_COL = "Death event time"
EMBEDDING_KEY = "metabolomic embedding"


def _randomize_column(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n random values from the distribution of `values` (per-feature)."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros(n)
    mean, std = float(np.mean(finite)), float(np.std(finite))
    lo, hi = float(np.min(finite)), float(np.max(finite))
    fake = rng.normal(mean, std if std > 0 else 1.0, n)
    return np.clip(fake, lo, hi)


def make_fake_from_reference(
    ref_path: str, n_samples: int, seed: int, out_dir: str | None = None
) -> str:
    """Mirror the structure of a real h5ad with randomized values (remote-repo style)."""
    ref_path = str(ref_path)
    original = ad.read_h5ad(ref_path)
    rng = np.random.default_rng(seed)
    n_features = original.n_vars

    # Randomize X per feature, preserving var/obs schema
    fake_X = np.column_stack(
        [
            _randomize_column(np.asarray(original.X[:, i]).flatten(), n_samples, rng)
            for i in range(n_features)
        ]
    )
    fake = ad.AnnData(X=fake_X, var=original.var.copy())
    fake.var_names = original.var_names

    # obs: keep schema, randomize numeric columns, sample categorical columns
    obs = pd.DataFrame(index=[f"fake_{i:05d}" for i in range(n_samples)])
    for col in original.obs.columns:
        col_values = original.obs[col]
        if pd.api.types.is_numeric_dtype(col_values):
            obs[col] = _randomize_column(col_values.to_numpy(dtype=float), n_samples, rng)
        else:
            uniq = col_values.dropna().unique()
            obs[col] = rng.choice(uniq, size=n_samples) if len(uniq) else np.nan
    fake.obs = obs

    # Rebuild layers with the same names; z-score layers recomputed from var params
    for name in original.layers.keys():
        layer = np.asarray(original.layers[name])
        if name == "Z-score normalized" and {"Z-score mean", "Z-score std"} <= set(original.var.columns):
            mean = original.var["Z-score mean"].to_numpy()
            std = original.var["Z-score std"].to_numpy()
            fake.layers[name] = (fake_X - mean) / (std + 1e-8)
        else:
            fake.layers[name] = np.column_stack(
                [_randomize_column(layer[:, i].flatten(), n_samples, rng) for i in range(layer.shape[1])]
            )

    # Rebuild obsm with the same keys/shapes (e.g. 512-dim 'metabolomic embedding')
    for key, arr in original.obsm.items():
        dim = arr.shape[1]
        embs = rng.normal(0.0, 1.0, size=(n_samples, dim))
        embs = (embs - embs.mean(axis=0)) / (embs.std(axis=0) + 1e-8)
        fake.obsm[key] = embs.astype(np.float32)

    out_path = Path(out_dir or Path(ref_path).parent) / f"fake_{Path(ref_path).name}"
    os.makedirs(out_path.parent, exist_ok=True)
    fake.write_h5ad(out_path)
    print(f"wrote {out_path}  (shape {fake.shape})")
    return str(out_path)


def load_vocab_names() -> list:
    """Feature names for the synthetic NMR panel.

    Uses the released tokenizer vocabulary (Model_Weights/MetAgeFormer/vocab.txt)
    so the fake data is tokenizable by the released model; falls back to
    positional NMR_%03d names if the file is unavailable.
    """
    vocab_path = (
        Path(__file__).resolve().parent.parent
        / "Model_Weights" / "MetAgeFormer" / "vocab.txt"
    )
    if vocab_path.exists():
        names = [line.strip() for line in vocab_path.read_text().splitlines() if line.strip()]
        if len(names) >= N_METABOLITES:
            return names[:N_METABOLITES]
    return [f"NMR_{i:03d}" for i in range(N_METABOLITES)]


def make_nmr_split(n_samples: int, rng: np.random.Generator) -> ad.AnnData:
    """Synthetic NMR split: z-scored layer + var z-score params + NaN gaps + obs age."""
    var_names = load_vocab_names()
    mean = rng.normal(0.0, 1.0, size=N_METABOLITES)
    std = np.abs(rng.normal(1.0, 0.2, size=N_METABOLITES))
    X = rng.normal(mean, std, size=(n_samples, N_METABOLITES))

    nan_mask = rng.random(X.shape) < 0.05  # ~5% missing, like real panels
    X[nan_mask] = np.nan
    z_scored = (X - mean) / std
    z_scored[nan_mask] = np.nan

    adata = ad.AnnData(X=X, var={"Z-score mean": mean, "Z-score std": std})
    adata.var_names = var_names
    adata.layers["Z-score normalized"] = z_scored
    adata.obs_names = [f"SAMPLE_{i:04d}" for i in range(n_samples)]
    adata.obs[AGE_COL] = rng.uniform(40.0, 80.0, size=n_samples)
    return adata


def _simulate_gompertz_survival(age: np.ndarray, rng: np.random.Generator):
    """Gompertz-like (time, event) pairs with censoring, for demo survival obs."""
    alpha, gamma, beta_age = -10.5, 0.09, 0.1
    u = rng.random(len(age))
    survival_time = np.log(1 - gamma * np.log(u) / np.exp(alpha + beta_age * age)) / gamma
    censor_time = rng.uniform(0.0, 20.0, size=len(age))
    time = np.minimum(survival_time, censor_time).astype(np.float32)
    event = (survival_time <= censor_time).astype(np.float32)
    return time, event


def make_gompertz_split(n_samples: int, rng: np.random.Generator) -> ad.AnnData:
    """Synthetic DeepGompertz split: random embeddings + Gompertz-like survival obs."""
    embedding = rng.normal(0.0, 1.0, size=(n_samples, EMBEDDING_DIM)).astype(np.float32)
    age = rng.uniform(40.0, 80.0, size=n_samples).astype(np.float32)
    time, event = _simulate_gompertz_survival(age, rng)

    adata = ad.AnnData(X=np.zeros((n_samples, 1), dtype=np.float32))
    adata.obs[AGE_COL] = age
    adata.obs[EVENT_COL] = event
    adata.obs[TIME_COL] = time
    adata.obsm[EMBEDDING_KEY] = embedding
    adata.obs_names = [f"EMB_SAMPLE_{i:04d}" for i in range(n_samples)]
    return adata


def make_blood_split(n_samples: int, rng: np.random.Generator) -> ad.AnnData:
    """Synthetic blood biochemistry split for the distillation demo.

    Layer 'Z-score normalized' (NaN-safe panel), teacher embeddings, and
    Gompertz-like survival obs — the format expected by distillation/train.py.
    """
    X = rng.normal(0.0, 1.0, size=(n_samples, BLOOD_N_FEATURES)).astype(np.float32)
    nan_mask = rng.random(X.shape) < 0.05  # ~5% missing, like real panels
    X[nan_mask] = np.nan
    age = rng.uniform(40.0, 80.0, size=n_samples).astype(np.float32)
    time, event = _simulate_gompertz_survival(age, rng)
    embedding = rng.normal(0.0, 1.0, size=(n_samples, EMBEDDING_DIM)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = [f"Blood_{i:02d}" for i in range(BLOOD_N_FEATURES)]
    adata.layers["Z-score normalized"] = X
    adata.obs[AGE_COL] = age
    adata.obs[EVENT_COL] = event
    adata.obs[TIME_COL] = time
    adata.obsm[EMBEDDING_KEY] = embedding
    adata.obs_names = [f"BLOOD_SAMPLE_{i:04d}" for i in range(n_samples)]
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from", dest="ref_path", metavar="REAL.h5ad",
        help="mirror the structure of a real .h5ad with randomized values (remote-repo style)",
    )
    group.add_argument(
        "--synthetic", action="store_true",
        help="build demo NMR + DeepGompertz + blood datasets from scratch (no data access needed)",
    )
    parser.add_argument("--n_samples", type=int, default=1000, help="samples per reference-based file")
    parser.add_argument("--outdir", type=str, default="Data/fake", help="output root for --synthetic")
    parser.add_argument("--out_dir", type=str, default=None, help="output dir for --from (default: next to input)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.ref_path:
        make_fake_from_reference(args.ref_path, args.n_samples, args.seed, args.out_dir)
    else:
        rng = np.random.default_rng(args.seed)
        nmr_dir = Path(args.outdir) / "NMR_dataset_fake"
        os.makedirs(nmr_dir, exist_ok=True)
        for split, n in [("train", 64), ("val", 16), ("test", 16)]:
            adata = make_nmr_split(n, rng)
            adata.write_h5ad(nmr_dir / f"{split}.h5ad")
            print(f"wrote {nmr_dir / (split + '.h5ad')}")

        gomp_dir = Path(args.outdir) / "deep_gompertz_fake"
        os.makedirs(gomp_dir, exist_ok=True)
        for split, n in [("train", 128), ("val", 32), ("test", 32)]:
            adata = make_gompertz_split(n, rng)
            adata.write_h5ad(gomp_dir / f"{split}.h5ad")
            print(f"wrote {gomp_dir / (split + '.h5ad')}")

        blood_dir = Path(args.outdir) / "Blood_dataset_fake"
        os.makedirs(blood_dir, exist_ok=True)
        for split, n in [("train", 128), ("val", 32), ("test", 32)]:
            adata = make_blood_split(n, rng)
            adata.write_h5ad(blood_dir / f"{split}.h5ad")
            print(f"wrote {blood_dir / (split + '.h5ad')}")

    print("Done. These are FAKE data for demonstration only — never use for real analysis.")


if __name__ == "__main__":
    main()
