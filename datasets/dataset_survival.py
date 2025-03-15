import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_logger = logging.getLogger(__name__)


class TCGAWSIRNASurvivalDataset(Dataset):
    def __init__(
        self,
        wsi_feature_dir,
        rna_feature_csv,
        survival_csv,
        num_wsi_feature_tokens,
        splits,
        k=5,
        num_bins=4,
        case_id_column="Patient ID",
        slide_id_column="Sample ID",
        label_column="Overall Survival (Months)",
        censor_column="Overall Survival Status",
        wsi_feature_only=False,
        cache=False,
    ):
        self.wsi_feature_dir = wsi_feature_dir
        self.rna_feature_csv = rna_feature_csv
        self.survival_csv = survival_csv
        self.num_wsi_feature_tokens = num_wsi_feature_tokens
        self.splits = splits
        self.k = k
        self.fold_nb = 0
        self.num_bins = num_bins
        self.case_id_column = case_id_column
        self.slide_id_column = slide_id_column
        self.label_column = label_column
        self.censor_column = censor_column
        self.wsi_feature_only = wsi_feature_only
        self.cache = cache
        self.eps = 1e-6

        self.wsi_feature_files = os.listdir(self.wsi_feature_dir)
        self.rna_feature_df = pd.read_csv(
            self.rna_feature_csv, header=0, index_col=0, sep=","
        ).fillna(0)
        self.survival_data = pd.read_csv(self.survival_csv, sep=",").fillna(0)
        self._filter_data()
        self.num_classes = None
        self._gen_disc_label()

        # For class balanced sampler protol
        self.slide_cls_ids = [
            np.where(self.survival_data["label"] == i)[0]  # type: ignore[call-overload]
            for i in range(self.num_classes)  # type: ignore[call-overload]
        ]

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

        # Drop duplicated survival data
        self.survival_data = self.survival_data.loc[
            ~self.survival_data[self.slide_id_column]
            .apply(lambda x: x.split(".")[0])
            .duplicated(keep="first")
        ]
        self.survival_data = self.survival_data.drop_duplicates(
            subset=self.case_id_column, keep="first"
        )

        orig_num_wsi_feature_files = len(self.wsi_feature_files)
        orig_num_rna_feature_ids = len(self.rna_feature_df)
        orig_num_survival_ids = len(self.survival_data)

        # Filter out features that do not have corresponding data
        wsi_feature_ids = set([f.split(".")[0][:15] for f in self.wsi_feature_files])
        survival_ids = set(
            [f.split(".")[0] for f in self.survival_data[self.slide_id_column].tolist()]
        )
        common_ids = wsi_feature_ids & survival_ids
        self.wsi_feature_files = [
            f for f in self.wsi_feature_files if f.split(".")[0][:15] in common_ids
        ]
        self.survival_data = self.survival_data[
            self.survival_data[self.slide_id_column]
            .apply(lambda x: x.split(".")[0])
            .isin(common_ids)
        ]

        wsi_feature_ids = set([f.split(".")[0][:15] for f in self.wsi_feature_files])
        rna_feature_ids = set(self.rna_feature_df.index.tolist())
        survival_ids = set(
            [
                f.split(".")[0][:15]
                for f in self.survival_data[self.slide_id_column].tolist()
            ]
        )
        common_ids = wsi_feature_ids & rna_feature_ids & survival_ids
        self.wsi_feature_files = [
            f for f in self.wsi_feature_files if f.split(".")[0][:15] in common_ids
        ]
        self.rna_feature_df = self.rna_feature_df.loc[list(common_ids)]
        self.survival_data = self.survival_data[
            self.survival_data[self.slide_id_column]
            .apply(lambda x: x.split(".")[0][:15])
            .isin(common_ids)
        ]

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
        filtered_survival_ids = orig_num_survival_ids - len(self.survival_data)
        if filtered_survival_ids > 0:
            _logger.warning(
                f"Survival data for {filtered_survival_ids} slides are missing"
            )

    def _gen_disc_label(self):
        patients_df = self.survival_data.copy()
        event_df = self.survival_data[
            self.survival_data[self.censor_column] == "1:DECEASED"
        ]

        if len(event_df) > 0:
            disc_labels, q_bins = pd.qcut(
                event_df[self.label_column], q=self.num_bins, retbins=True, labels=False
            )
            q_bins[-1] = self.survival_data[self.label_column].max() + self.eps
            q_bins[0] = self.survival_data[self.label_column].min() - self.eps
            disc_labels, q_bins = pd.cut(
                patients_df[self.label_column],
                bins=q_bins,
                retbins=True,
                labels=False,
                right=False,
                include_lowest=True,
            )
        else:
            disc_labels, q_bins = pd.cut(
                patients_df[self.label_column],
                bins=self.num_bins,
                retbins=True,
                labels=False,
                right=False,
                include_lowest=True,
            )
        patients_df.insert(
            len(patients_df.columns), "disc_label", disc_labels.values.astype(int)
        )

        label_dict = dict()
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.survival_data = self.survival_data.merge(
            patients_df[[self.case_id_column, "disc_label"]],
            on=self.case_id_column,
            how="left",
        )
        self.survival_data[self.censor_column] = self.survival_data[
            self.censor_column
        ].astype(
            str
        )  # Some entries may be NaN
        self.survival_data["censorship"] = (
            self.survival_data[self.censor_column].str[0].astype(int)
        )
        self.survival_data["label"] = self.survival_data.apply(
            lambda row: label_dict[(row["disc_label"], row["censorship"])], axis=1
        )

        self.num_classes = len(label_dict)

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
            self.used_feature_data[slide] = torch.load(
                os.path.join(self.wsi_feature_dir, f"{slide}.pt")
            )

    # For class balanced sampler protol
    def get_label(self, idx):
        return int(self.survival_data["label"][idx])

    def __len__(self):
        return len(self.used_feature_ids)

    def __getitem__(self, idx):
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

        slide_series = self.survival_data.loc[
            self.survival_data[self.slide_id_column].apply(lambda x: x.split(".")[0])
            == slide[:15]
        ]
        assert len(slide_series) == 1, f"Multiple records exist for slide {slide}"
        label = torch.tensor(slide_series["disc_label"].to_numpy(), dtype=torch.int)
        event_time = torch.tensor(slide_series[self.label_column].to_numpy())
        c = torch.tensor(slide_series["censorship"].to_numpy())

        if not self.wsi_feature_only:
            return wsi_feature, rna_feature, label, event_time, c
        else:
            return wsi_feature, label, event_time, c
