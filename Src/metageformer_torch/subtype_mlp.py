"""PyTorch MLP with focal loss for imbalanced metabolic subtype assignment.

This is the classifier shipped with the released subtype model
(`Model_Weights/SubtypeClassifier/subtype_mlp_classifier_focal.joblib`).
Despite the `.joblib` extension, that file is a PyTorch `torch.save` archive —
load it with `FocalMLPClassifier.load()`, never with `joblib.load()`.

Inference (`load` / `predict` / `predict_proba`) needs only torch + numpy.
`fit()` additionally needs scikit-learn.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

LossName = Literal["focal", "weighted_ce", "ce"]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class _MLPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_sizes: tuple[int, ...],
        n_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _class_weight_vector(y_enc: np.ndarray, n_classes: int, mode: str):
    if mode in (None, "none"):
        return None
    if mode == "balanced_sqrt":
        counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / np.sqrt(counts)
        weights /= weights.mean()
        return weights.astype(np.float32)
    if mode == "balanced":
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.arange(n_classes)
        return compute_class_weight("balanced", classes=classes, y=y_enc).astype(np.float32)
    raise ValueError(mode)


class FocalMLPClassifier:
    """sklearn-compatible API: fit / predict / predict_proba / save / load."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        focal_gamma: float = 2.0,
        class_weight: str = "balanced",
        balanced_sampler: bool = True,
        sampler_max_ratio: float = 30.0,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        max_epochs: int = 500,
        validation_fraction: float = 0.1,
        early_stopping: bool = True,
        patience: int = 30,
        dropout: float = 0.1,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.focal_gamma = focal_gamma
        self.class_weight = class_weight
        self.balanced_sampler = balanced_sampler
        self.sampler_max_ratio = sampler_max_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.classes_: Optional[np.ndarray] = None
        self._net: Optional[_MLPNet] = None
        self.n_iter_: int = 0
        self.best_val_loss_: Optional[float] = None
        self.class_counts_: Optional[dict[str, int]] = None

    def _encode(self, y: np.ndarray) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.classes_)}
        return np.array([lookup[int(v)] for v in y], dtype=np.int64)

    def _decode(self, y_enc: np.ndarray) -> np.ndarray:
        return self.classes_[y_enc]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FocalMLPClassifier":
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Need at least 2 classes")

        y_enc = self._encode(y)
        self.class_counts_ = {str(int(c)): int((y == c).sum()) for c in self.classes_}

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        if self.validation_fraction > 0 and len(y) >= 20:
            idx = np.arange(len(y))
            tr_idx, va_idx = train_test_split(
                idx,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_enc,
            )
        else:
            tr_idx = np.arange(len(y))
            va_idx = np.array([], dtype=int)

        X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
        X_va = X[va_idx] if len(va_idx) else None
        y_va = y_enc[va_idx] if len(va_idx) else None

        self._net = _MLPNet(
            in_dim=X.shape[1],
            hidden_sizes=self.hidden_layer_sizes,
            n_classes=n_classes,
            dropout=self.dropout,
        ).to(self.device)

        alpha_np = _class_weight_vector(y_tr, n_classes, self.class_weight)
        alpha_t = (
            torch.tensor(alpha_np, dtype=torch.float32, device=self.device)
            if alpha_np is not None
            else None
        )
        criterion = FocalLoss(gamma=self.focal_gamma, alpha=alpha_t)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)

        tr_x = torch.tensor(X_tr, dtype=torch.float32)
        tr_y = torch.tensor(y_tr, dtype=torch.long)
        dataset = TensorDataset(tr_x, tr_y)

        if self.balanced_sampler and len(tr_y) > self.batch_size:
            counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            sample_w = 1.0 / counts[y_tr]
            if self.sampler_max_ratio > 0:
                sample_w = np.minimum(sample_w, self.sampler_max_ratio / len(y_tr))
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True,
                generator=torch.Generator().manual_seed(self.random_state),
            )
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(tr_y)),
                sampler=sampler,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(tr_y)),
                shuffle=True,
                generator=torch.Generator().manual_seed(self.random_state),
            )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
            self.n_iter_ = epoch + 1

            if X_va is not None and len(va_idx):
                val_loss = self._eval_loss(X_va, y_va, criterion)
                if self.verbose and (epoch + 1) % 25 == 0:
                    print(f"epoch {epoch + 1}: val_loss={val_loss:.4f}")
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self._net.state_dict().items()
                    }
                    bad_epochs = 0
                elif self.early_stopping:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
            self.best_val_loss_ = float(best_val)
        elif X_va is not None and len(va_idx):
            self.best_val_loss_ = float(self._eval_loss(X_va, y_va, criterion))

        if self.verbose:
            print(
                f"Focal MLP done: epochs={self.n_iter_}, "
                f"best_val_loss={self.best_val_loss_}, device={self.device}"
            )
        return self

    def _eval_loss(self, X, y_enc, criterion) -> float:
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            yb = torch.tensor(y_enc, dtype=torch.long, device=self.device)
            return float(criterion(self._net(xb), yb).item())

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None and self.classes_ is not None
        self._net.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self._net(xb).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        return self._decode(np.argmax(logits, axis=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path: str) -> None:
        assert self._net is not None and self.classes_ is not None
        payload = {
            "state_dict": self._net.state_dict(),
            "classes": self.classes_,
            "config": {
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "focal_gamma": self.focal_gamma,
                "class_weight": self.class_weight,
                "balanced_sampler": self.balanced_sampler,
                "sampler_max_ratio": self.sampler_max_ratio,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "validation_fraction": self.validation_fraction,
                "early_stopping": self.early_stopping,
                "patience": self.patience,
                "dropout": self.dropout,
                "random_state": self.random_state,
                "in_dim": self._net.net[0].in_features,
                "n_classes": len(self.classes_),
            },
            "n_iter": self.n_iter_,
            "best_val_loss": self.best_val_loss_,
            "class_counts": self.class_counts_,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "FocalMLPClassifier":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        obj = cls(
            hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),
            focal_gamma=cfg["focal_gamma"],
            class_weight=cfg["class_weight"],
            balanced_sampler=cfg["balanced_sampler"],
            sampler_max_ratio=cfg.get("sampler_max_ratio", 30.0),
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            max_epochs=cfg["max_epochs"],
            validation_fraction=cfg["validation_fraction"],
            early_stopping=cfg["early_stopping"],
            patience=cfg["patience"],
            dropout=cfg["dropout"],
            random_state=cfg["random_state"],
            device=device,
        )
        obj.classes_ = np.asarray(payload["classes"])
        obj.n_iter_ = int(payload.get("n_iter", 0))
        obj.best_val_loss_ = payload.get("best_val_loss")
        obj.class_counts_ = payload.get("class_counts")
        obj._net = _MLPNet(
            in_dim=cfg["in_dim"],
            hidden_sizes=tuple(cfg["hidden_layer_sizes"]),
            n_classes=cfg["n_classes"],
            dropout=cfg["dropout"],
        ).to(obj.device)
        obj._net.load_state_dict(payload["state_dict"])
        obj._net.eval()
        return obj
