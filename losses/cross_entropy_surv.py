import torch
import torch.nn as nn


class CrossEntropySurvLoss(nn.Module):
    """
    Cross-Entropy loss for discrete-time survival analysis.

    Parameters
    ----------
    eps : float, optional (default=1e-7)
        A small value to avoid log(0).
    reduction : str, optional (default='mean')
        Specifies the reduction to apply:
        - 'none': no reduction (returns a loss per sample)
        - 'mean': mean loss over the batch
        - 'sum': sum of losses over the batch
    """

    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, event_times: torch.Tensor, censoring: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross-entropy for discrete survival.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (N, M). Raw model outputs for each of M intervals.
        event_times : torch.Tensor
            Shape (N,). event_times[i] = T_i, the interval at which event/censoring occurred.
        censoring : torch.Tensor
            Shape (N,). censoring[i] = 1 if event occurred at T_i (uncensored),
            0 if censored at T_i.

        Returns
        -------
        torch.Tensor
            The computed cross-entropy loss. Scalar if reduction='mean' or 'sum', else (N,).
        """
        N, M = logits.shape  # noqa: N806
        # Convert logits to hazards
        hazards = torch.sigmoid(logits).clamp(min=self.eps, max=1 - self.eps)

        # Compute survival probabilities to form a full categorical distribution over M+1 outcomes
        # S_k = product of (1 - h_j) for j < k
        # We'll build cumulative survival: S for all intervals
        one_minus_h = 1 - hazards
        # Cumprod along intervals gives S_k for each k
        # S_0 = 1 by definition (survive before first interval)
        survival = torch.cumprod(one_minus_h, dim=1)
        # survival[:, t] = product_{j=0 to t} (1 - h_j)

        # Probability event at interval t:
        # p(event at t) = h_t * product_{j< t}(1 - h_j)
        # product_{j<t}(1 - h_j) = survival[:, t-1] if t>0 else 1
        # Let's build these probabilities:
        # We can prepend a column of ones for convenience
        survival_padded = torch.cat(
            [torch.ones((N, 1), device=hazards.device), survival], dim=1
        )
        # survival_padded[:, t] = product_{j< t}(1 - h_j), starting with t=1 (since padded)

        # p_event_t = h_t * survival_padded[:, t]
        # For t in [0..M-1], class t corresponds to event at interval t
        p_event = hazards * survival_padded[:, :-1]

        # Probability no event (censored) after last interval:
        # p_no_event = survival[:, M-1] = product_{j<M}(1 - h_j)
        p_no_event = survival[:, -1].unsqueeze(1)

        # Concatenate event probabilities and no-event probability into a distribution (N, M+1)
        # Classes: 0..M-1 for events, M for no event
        p_dist = torch.cat([p_event, p_no_event], dim=1)

        # Normalize distribution just in case (it should already sum to 1)
        # Theoretically p_dist sums to 1, but due to numerical issues, let's be safe.
        p_sum = p_dist.sum(dim=1, keepdim=True)
        p_dist = p_dist / p_sum.clamp(min=self.eps)

        # Determine target classes:
        # If event: class = event_times[i]
        # If censored: class = M (the no-event class)
        targets = torch.where(
            censoring == 1, event_times, torch.full_like(event_times, M)
        )

        # Compute cross-entropy:
        # CE = -log p_dist[i, targets[i]]
        # Use gather to pick out the correct probability for each sample
        chosen_p = p_dist.gather(dim=1, index=targets.unsqueeze(1)).clamp(min=self.eps)
        ce_loss = -torch.log(chosen_p)

        # Apply reduction
        if self.reduction == "mean":
            return ce_loss.mean()
        elif self.reduction == "sum":
            return ce_loss.sum()
        else:
            return ce_loss
