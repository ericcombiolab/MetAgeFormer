"""Cox proportional hazards loss utilities."""

from __future__ import annotations

import torch


def cox_ph_loss(
    risk: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Breslow Cox partial-likelihood loss (mean over events)."""
    risk = risk.reshape(-1)
    event = event.reshape(-1).float()
    time = time.reshape(-1)
    n_events = event.sum()
    if n_events < 1:
        return risk.new_zeros(())

    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    event_mask = event > 0.5
    loss = (log_cum[event_mask] - risk[event_mask]).sum() / n_events.clamp_min(1.0)
    return loss
