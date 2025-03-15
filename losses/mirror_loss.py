"""MIRROR Pre-training Loss
Copyright (c) 2025, Tianyi Wang @ The University of Sydney
All rights reserved.

Based on the OpenCLIP codebase by Ross Wightman
https://github.com/mlfoundations/open_clip

Licensed under the GNU General Public License v3.0, see LICENSE for details
"""

import torch
import torch.nn.functional as F
from torch import nn


class ClipLoss(nn.Module):

    def __init__(self, cache_labels=False):
        super().__init__()
        self.cache_labels = cache_labels

        # Cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # Calculate ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, wsi_features, rna_features, logit_scale, output_dict=False):
        # Compute logits
        logits_per_image = logit_scale * wsi_features @ rna_features.T
        logits_per_text = logit_scale * rna_features @ wsi_features.T

        # Generate ground-truth labels
        device = wsi_features.device
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        # Compute contrastive loss
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class MIRRORLoss(nn.Module):
    def __init__(
        self,
        clip_loss_cache_labels=True,
        alignment_loss_weight=0.5,
        wsi_retention_loss_weight=0.1,
        rna_retention_loss_weight=0.1,
        style_loss_weight=0.1,
        cluster_loss_weight=0.2,
    ):
        super().__init__()

        self.clip_loss = ClipLoss(cache_labels=clip_loss_cache_labels)
        self.alignment_loss_weight = alignment_loss_weight
        self.wsi_retention_loss_weight = wsi_retention_loss_weight
        self.rna_retention_loss_weight = rna_retention_loss_weight
        self.style_loss_weight = style_loss_weight
        self.cluster_loss_weight = cluster_loss_weight

    def forward(
        self,
        wsi_alignment_emb,
        wsi_retention_emb,
        wsi_retention_target,
        wsi_mask,
        wsi_score,
        wsi_mu,
        wsi_logstd,
        rna_alignment_emb,
        rna_retention_emb,
        rna_retention_target,
        rna_mask,
        rna_score,
        rna_mu,
        rna_logstd,
        logit_scale,
    ):
        alignment_loss = self.clip_loss(
            wsi_alignment_emb,
            rna_alignment_emb,
            logit_scale,
        )

        wsi_retention_loss = (wsi_retention_emb - wsi_retention_target) ** 2
        wsi_retention_loss = wsi_retention_loss.mean(dim=-1)
        wsi_retention_loss = (wsi_retention_loss * wsi_mask).sum() / wsi_mask.sum()

        rna_retention_loss = (rna_retention_emb - rna_retention_target) ** 2
        rna_retention_loss = (rna_retention_loss * rna_mask).sum() / rna_mask.sum()

        style_loss = 0.5 * (
            torch.sum(
                torch.exp(wsi_logstd) + wsi_mu**2 - 1.0 - wsi_logstd, dim=1
            ).mean()
            + torch.sum(
                torch.exp(rna_logstd) + rna_mu**2 - 1.0 - rna_logstd, dim=1
            ).mean()
        )

        wsi_prob = F.softmax(wsi_score, dim=-1)
        rna_prob = F.softmax(rna_score, dim=-1)
        cluster_loss = 0.5 * (
            F.kl_div(wsi_prob.log(), rna_prob, reduction="batchmean")
            + F.kl_div(rna_prob.log(), wsi_prob, reduction="batchmean")
        )

        total_loss = (
                self.alignment_loss_weight * alignment_loss
                + self.wsi_retention_loss_weight * wsi_retention_loss
                + self.rna_retention_loss_weight * rna_retention_loss
                + self.style_loss_weight * style_loss
                + self.cluster_loss_weight * cluster_loss
        )
        return (
            total_loss,
            alignment_loss,
            wsi_retention_loss,
            rna_retention_loss,
            style_loss,
            cluster_loss,
        )
