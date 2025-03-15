import torch
import torch.nn as nn


class NLLSurvLoss(nn.Module):
    """
    Negative Log-Likelihood loss for discrete-time survival analysis with alpha and eps.
    This version now expects logits as input (like NLLSurvLoss) and applies sigmoid internally.
    """

    def __init__(self, alpha=0.0, eps=1e-7, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, event_times: torch.Tensor, censoring: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the NLL for discrete survival given model logits (raw outputs), event times, and censoring indicators.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (N, M). These are the raw model outputs (not probabilities).
        event_times : torch.Tensor
            Shape (N,). event_times[i] = T_i, the event or censor time interval.
        censoring : torch.Tensor
            Shape (N,). censoring[i] = 1 if event occurred at T_i (uncensored), 0 if censored at T_i.

        Returns
        -------
        torch.Tensor
            The NLL loss. If reduction='none', returns loss per sample. Otherwise, a scalar.
        """
        N, M = logits.shape  # noqa: N806

        # Convert logits to hazards
        hazards = torch.sigmoid(logits)

        # Create a time range matrix
        event_times_expanded = event_times.unsqueeze(1).expand(N, M)
        time_range = torch.arange(M, device=hazards.device).unsqueeze(0).expand(N, M)

        # event=1 (uncensored), censor=0
        uncensored_mask = censoring == 1
        censored_mask = censoring == 0

        # Clamp hazards to avoid log(0)
        hazards = hazards.clamp(min=self.eps, max=1.0 - self.eps)
        log_hazards = torch.log(hazards)
        log_one_minus_hazards = torch.log(1 - hazards)

        # Uncensored: survived up to T_i-1 and failed at T_i
        uncensored_survival_mask = (
            time_range < event_times_expanded
        ) & uncensored_mask.unsqueeze(1)
        uncensored_event_mask = (
            time_range == event_times_expanded
        ) & uncensored_mask.unsqueeze(1)
        uncensored_survival_sum = (
            log_one_minus_hazards * uncensored_survival_mask
        ).sum(dim=1)
        uncensored_event_sum = (log_hazards * uncensored_event_mask).sum(dim=1)
        uncensored_nll = -(uncensored_survival_sum + uncensored_event_sum)

        # Censored: survived through T_i
        censored_survival_mask = (
            time_range <= event_times_expanded
        ) & censored_mask.unsqueeze(1)
        censored_survival_sum = (log_one_minus_hazards * censored_survival_mask).sum(
            dim=1
        )
        censored_nll = -censored_survival_sum

        # Combine per-sample losses
        nll_per_sample = torch.zeros(N, device=hazards.device)
        nll_per_sample[uncensored_mask] = uncensored_nll[uncensored_mask]
        nll_per_sample[censored_mask] = censored_nll[censored_mask]

        # Apply alpha weighting
        uncensored_loss = torch.zeros_like(nll_per_sample)
        uncensored_loss[uncensored_mask] = uncensored_nll[uncensored_mask]
        neg_l = nll_per_sample
        loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss

        # Reduction
        if self.reduction == "mean":
            return loss.mean()  # type: ignore[no-any-return]
        elif self.reduction == "sum":
            return loss.sum()  # type: ignore[no-any-return]
        else:
            return loss  # type: ignore[no-any-return]
