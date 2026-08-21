import argparse
import os
from typing import Any, Dict, Optional

from torchsurv.metrics.cindex import ConcordanceIndex


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def in_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def resolve_num_workers(requested: Optional[int] = None, default: int = 0) -> int:
    if requested is not None:
        return int(requested)
    if in_slurm_job():
        return min(4, os.cpu_count() or 1)
    return default


def progress_iter(iterable, desc: str = "", total: Optional[int] = None, disable: Optional[bool] = None):
    from tqdm import tqdm

    if disable is None:
        disable = in_slurm_job()
    return tqdm(iterable, desc=desc, total=total, disable=disable)


def safe_cindex(risk, event, time) -> float:
    try:
        # torchsurv ConcordanceIndex is callable; it has no update()/compute().
        metric = ConcordanceIndex()
        value = metric(
            risk.reshape(-1),
            event.reshape(-1).bool(),
            time.reshape(-1),
        )
        return float(value.detach().cpu())
    except Exception:
        return float("nan")


def init_wandb(args, config: Dict[str, Any]):
    if not getattr(args, "use_wandb", True):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without wandb logging.")
        return None

    project = getattr(args, "wandb_project", "MetAgeFormer")
    run_name = getattr(args, "wandb_run_name", None) or getattr(args, "save_dir", "run")
    try:
        return wandb.init(project=project, name=os.path.basename(str(run_name)), config=config)
    except Exception as exc:
        print(f"wandb init failed ({exc}); continuing without wandb logging.")
        return None
