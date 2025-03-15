"""Loader Sampler
Copyright (c) 2025, Tianyi Wang @ The University of Sydney
All rights reserved.

Based on the CLAM codebase by Mahmood Lab
https://github.com/mahmoodlab/CLAM

Licensed under the GNU General Public License v3.0, see LICENSE for details
"""

import torch
from torch.utils.data import WeightedRandomSampler


def class_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(dataset):
        class_counts = [len(cls_ids) for cls_ids in dataset.slide_cls_ids]
        total_samples = len(dataset)
        weight_per_class = [total_samples / count for count in class_counts]
        weights = [
            weight_per_class[dataset.get_label(idx)] for idx in range(total_samples)
        ]
        return torch.DoubleTensor(weights)

    weights = make_weights_for_balanced_classes(dataset)
    return WeightedRandomSampler(weights, len(weights))
