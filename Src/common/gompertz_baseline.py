import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class AgeOnlyGompertzBaseline(nn.Module):
    """Age-scale left-truncated Gompertz baseline (model B)."""

    def __init__(self, init_alpha: float = -10.5, init_gamma: float = 0.09, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.raw_gamma = nn.Parameter(torch.tensor(inverse_softplus(float(init_gamma))))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + self.eps

    def params_dict(self) -> Dict[str, float]:
        return {
            "alpha_age_scale": float(self.alpha.detach().cpu()),
            "gamma_age_scale": float(self.gamma.detach().cpu()),
        }

    def nll(self, age: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        age = age.reshape(-1).clamp_min(0.0)
        time = time.reshape(-1).clamp_min(self.eps)
        event = event.reshape(-1)
        gamma = self.gamma.clamp_min(self.eps)
        alpha = self.alpha

        age_end = age + time
        # Left-truncated Gompertz on the age scale (matches metabolomic_age inversion and
        # the stored fullcohort_107nonderived_mlm baseline ≈ α=-10.34, γ=0.0815):
        #   h(a)=exp(α+γa),  H(a)=exp(α)/γ * exp(γa)
        # Cumulative term MUST use -log(γ). Using +log(γ) yields a different optimum
        # (≈ α=-9.12, γ=0.12) and breaks metabolomic age scale on retrain/eval.
        log_hazard_end = alpha + gamma * age_end
        log_cum_end = alpha - torch.log(gamma) + gamma * age_end
        log_cum_start = alpha - torch.log(gamma) + gamma * age

        log_lik = event * log_hazard_end - (torch.exp(log_cum_end) - torch.exp(log_cum_start))
        return -log_lik.mean()

    def forward(self, age: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        return self.nll(age, time, event)


@torch.no_grad()
def evaluate_age_baseline(
    model: AgeOnlyGompertzBaseline,
    age: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> float:
    model.eval()
    return float(model.nll(age, time, event).detach().cpu())


def fit_age_baseline(
    train_data: Any,
    val_data: Any,
    device: str,
    args,
    wandb_run=None,
) -> Tuple[Dict[str, float], list, list]:
    model = AgeOnlyGompertzBaseline(init_alpha=args.init_alpha, init_gamma=args.init_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.baseline_lr, weight_decay=args.weight_decay)

    train_age = torch.from_numpy(train_data.age).to(device)
    train_event = torch.from_numpy(train_data.event).to(device)
    train_time = torch.from_numpy(train_data.time).to(device)
    val_age = torch.from_numpy(val_data.age).to(device)
    val_event = torch.from_numpy(val_data.event).to(device)
    val_time = torch.from_numpy(val_data.time).to(device)

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
        model.raw_gamma.copy_(torch.tensor(inverse_softplus(best_params["gamma_age_scale"]), device=device))

    return best_params, train_loss_epoch, val_loss_epoch
