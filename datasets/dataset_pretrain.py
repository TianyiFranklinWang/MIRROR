"""TCGA Dataset for WSI-RNA Pretraining
Copyright (c) 2025, Tianyi Wang @ The University of Sydney
All rights reserved.

Licensed under the GNU General Public License v3.0, see LICENSE for details
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_logger = logging.getLogger(__name__)


class TCGAWSIRNAPretrainDataset(Dataset):
    def __init__(
        self,
        wsi_feature_dir: str,
        rna_feature_csv: str,
        num_wsi_feature_tokens: int,
        splits: Optional[str] = None,
        k: int = 5,
        cache: bool = False,
    ) -> None:
        """
        Args:
            wsi_feature_dir: Directory containing WSI features
            rna_feature_csv: Path to CSV containing RNA features
            num_wsi_feature_tokens: Number of WSI patch features to sample
            splits: Path to CSV containing train/val splits. Defaults to None.
            k: Number of folds for cross-validation. Defaults to 5.
            cache: Whether to cache data
        """
        super().__init__()

        self.wsi_feature_dir = wsi_feature_dir
        self.rna_feature_csv = rna_feature_csv
        self.num_wsi_feature_tokens = num_wsi_feature_tokens
        self.splits = splits
        self.cache = cache
        self.k = k
        self.fold_nb = 0

        self.wsi_feature_files = os.listdir(self.wsi_feature_dir)
        self.rna_feature_df = pd.read_csv(
            self.rna_feature_csv, header=0, index_col=0, sep=","
        ).fillna(0)
        self._filter_data()

        if self.splits is not None:
            self.train_feature_ids: List[str] = []
            self.val_feature_ids: List[str] = []
            self.used_feature_ids: List[str] = []
            self.update_fold_nb(0)
        else:
            self.used_feature_ids = [f.split(".")[0] for f in self.wsi_feature_files]
        self.train()

    def _filter_data(self) -> None:
        # Drop duplicated rna features
        self.rna_feature_df = self.rna_feature_df.loc[
            ~self.rna_feature_df.index.duplicated(keep="first")
        ]

        orig_num_wsi_feature_files = len(self.wsi_feature_files)
        orig_num_rna_feature_ids = len(self.rna_feature_df)

        # Filter out features that do not have corresponding data
        wsi_feature_ids = set([f.split(".")[0][:15] for f in self.wsi_feature_files])
        rna_feature_ids = set(self.rna_feature_df.index.tolist())
        common_ids = wsi_feature_ids & rna_feature_ids
        self.wsi_feature_files = [
            f for f in self.wsi_feature_files if f.split(".")[0][:15] in common_ids
        ]
        self.rna_feature_df = self.rna_feature_df.loc[list(common_ids)]

        filtered_wsi_feature_ids = orig_num_wsi_feature_files - len(
            self.wsi_feature_files
        )
        if filtered_wsi_feature_ids > 0:
            _logger.warning(
                f"WSI features for {filtered_wsi_feature_ids} slides are missing"
            )
        filtered_rna_feature_ids = orig_num_rna_feature_ids - len(self.rna_feature_df)
        if filtered_rna_feature_ids > 0:
            _logger.warning(
                f"RNA features for {filtered_rna_feature_ids} slides are missing"
            )

    def update_fold_nb(self, fold_nb: int) -> "TCGAWSIRNAPretrainDataset":
        """Update fold number for cross-validation
        args:
            fold_nb: Fold number
        """
        self.fold_nb = fold_nb

        fold_csv = pd.read_csv(
            os.path.join(self.splits, f"splits_{fold_nb}.csv"),  # type: ignore[arg-type]
            header=0,
            index_col=0,
            sep=",",
        )
        train_patients = fold_csv["train"].dropna().tolist()
        val_patients = fold_csv["val"].dropna().tolist()
        self.train_feature_ids = [
            f.split(".")[0]
            for f in self.wsi_feature_files
            if f.split(".")[0][:12] in train_patients
        ]
        self.val_feature_ids = [
            f.split(".")[0]
            for f in self.wsi_feature_files
            if f.split(".")[0][:12] in val_patients
        ]

        return self

    def train(self) -> "TCGAWSIRNAPretrainDataset":
        if self.splits is not None:
            self.used_feature_ids = self.train_feature_ids
        if self.cache:
            self._cache_data()

        return self

    def val(self) -> "TCGAWSIRNAPretrainDataset":
        if self.splits is not None:
            self.used_feature_ids = self.val_feature_ids
        if self.cache:
            self._cache_data()

        return self

    def _cache_data(self) -> None:
        self.used_feature_data = {}
        for slide in self.used_feature_ids:
            self.used_feature_data[slide] = torch.load(
                os.path.join(self.wsi_feature_dir, f"{slide}.pt")
            )

    def __len__(self) -> int:
        return len(self.used_feature_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        slide = self.used_feature_ids[idx]

        if self.cache:
            wsi_feature = self.used_feature_data[slide]
        else:
            wsi_feature = torch.load(os.path.join(self.wsi_feature_dir, f"{slide}.pt"))
        is_replace = not wsi_feature.shape[0] >= self.num_wsi_feature_tokens
        sampled_indices = np.random.choice(
            wsi_feature.shape[0], self.num_wsi_feature_tokens, replace=is_replace
        )
        wsi_feature = wsi_feature[sampled_indices]

        rna_feature = torch.tensor(
            self.rna_feature_df.loc[slide[:15]].to_numpy(), dtype=torch.float32
        )

        return wsi_feature, rna_feature
