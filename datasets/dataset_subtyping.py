import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_logger = logging.getLogger(__name__)


class TCGAWSIRNASubtypingDataset(Dataset):
    def __init__(
        self,
        wsi_feature_dir,
        rna_feature_csv,
        classes,
        num_wsi_feature_tokens,
        splits,
        k=5,
        wsi_feature_only=False,
        cache=False,
    ):
        self.wsi_feature_dir = wsi_feature_dir
        self.rna_feature_csv = rna_feature_csv
        self.classes = classes
        self.num_wsi_feature_tokens = num_wsi_feature_tokens
        self.splits = splits
        self.k = k
        self.fold_nb = 0
        self.wsi_feature_only = wsi_feature_only
        self.cache = cache

        self.class_label = {
            class_idx: idx for idx, class_idx in enumerate(self.classes)
        }
        self.class_dict = dict()
        for class_idx in self.classes:
            class_slides = [
                slide.split(".")[0]
                for slide in os.listdir(os.path.join(self.wsi_feature_dir, class_idx))
            ]
            for slide in class_slides:
                self.class_dict[slide] = self.class_label[class_idx]
        self.wsi_feature_files = list(self.class_dict.keys())
        self.rna_feature_df = pd.read_csv(
            self.rna_feature_csv, header=0, index_col=0, sep=","
        ).fillna(0)
        self._filter_data()

        self.train_feature_ids = list()
        self.val_feature_ids = list()
        self.used_feature_ids = list()
        self.update_fold_nb(0)
        self.train()

    def _filter_data(self):
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

    def update_fold_nb(self, fold_nb):
        self.fold_nb = fold_nb

        fold_csv = pd.read_csv(
            os.path.join(self.splits, f"splits_{fold_nb}.csv"),
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

    def train(self):
        self.used_feature_ids = self.train_feature_ids
        if self.cache:
            self._cache_data()

        return self

    def val(self):
        self.used_feature_ids = self.val_feature_ids
        if self.cache:
            self._cache_data()

        return self

    def _cache_data(self):
        self.used_feature_data = {}
        for slide in self.used_feature_ids:
            label = self.class_dict[slide]
            self.used_feature_data[slide] = torch.load(
                os.path.join(
                    self.wsi_feature_dir,
                    list(self.class_label.keys())[label],
                    slide + ".pt",
                )
            )

    def __len__(self):
        return len(self.used_feature_ids)

    def __getitem__(self, idx):
        slide = self.used_feature_ids[idx]

        label = torch.tensor(self.class_dict[slide])

        if self.cache:
            wsi_feature = self.used_feature_data[slide]
        else:
            wsi_feature = torch.load(
                os.path.join(
                    self.wsi_feature_dir,
                    list(self.class_label.keys())[label],
                    slide + ".pt",
                )
            )
        is_replace = not wsi_feature.shape[0] >= self.num_wsi_feature_tokens
        sampled_indices = np.random.choice(
            wsi_feature.shape[0], self.num_wsi_feature_tokens, replace=is_replace
        )
        wsi_feature = wsi_feature[sampled_indices]

        rna_feature = torch.tensor(
            self.rna_feature_df.loc[slide[:15]].to_numpy(), dtype=torch.float32
        )

        if not self.wsi_feature_only:
            return wsi_feature, rna_feature, label
        else:
            return wsi_feature, label
