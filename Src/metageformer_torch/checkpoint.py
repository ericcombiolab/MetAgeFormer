import os
from typing import Any, Dict, Optional

import torch


def load_teacher_gompertz_config(checkpoint_path: str) -> Dict[str, Any]:
    """Read DeepGompertz head config + Gompertz baseline params from a teacher checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    baseline_params = checkpoint.get("baseline_params")
    if baseline_params is None:
        baseline_params = {
            "alpha_age_scale": config["alpha_age_scale"],
            "gamma_age_scale": config["gamma_age_scale"],
        }
    return {
        "gompertz_head_config": {
            "hidden_dim": config["hidden_dim"],
            "dropout": config["dropout"],
            "t_window": config["t_window"],
            "gamma_min": config.get("gamma_min", 1e-6),
        },
        "baseline_params": baseline_params,
    }


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location or "cpu")


def load_pretrained_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    if "METAGEFORMER" not in payload:
        raise KeyError(f"Pretrained checkpoint missing METAGEFORMER weights: {path}")
    return payload


def normalize_distilled_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize legacy distilled keys to MetAgeFormer_Lightweight_DeepGompertz layout."""
    if any(key.startswith("lightweight_model.fc1.") for key in state_dict):
        raise ValueError(
            "Legacy MLP distilled checkpoint detected (lightweight_model.fc1.*). "
            "Only blood-token Transformer lightweight checkpoints are supported in live code."
        )
    if not any(key.startswith("student.") for key in state_dict):
        return state_dict

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("student."):
            remapped["lightweight_model." + key[len("student.") :]] = value
        else:
            remapped[key] = value
    return remapped


def load_distilled_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    if "METAGEFORMER_DISTILLED" not in payload:
        raise KeyError(f"Distilled checkpoint missing METAGEFORMER_DISTILLED weights: {path}")
    payload["METAGEFORMER_DISTILLED"] = normalize_distilled_state_dict(payload["METAGEFORMER_DISTILLED"])
    return payload


def save_distilled_checkpoint(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """Atomically write distilled weights (tmp + replace) to avoid NFS overwrite races."""
    import tempfile

    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".model_weights_", suffix=".pth.tmp", dir=parent)
    os.close(fd)
    try:
        torch.save({"METAGEFORMER_DISTILLED": state_dict}, tmp_path)
        os.replace(tmp_path, abs_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
