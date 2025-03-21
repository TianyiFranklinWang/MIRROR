import argparse
import logging
import os
import random
import sys

import pandas as pd
import torch
from timm import utils


sys.path.append(os.getcwd())

from datasets import TCGAWSIRNASubtypingDataset, TCGAWSIRNASurvivalDataset


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate few-shot files")
parser.add_argument(
    "--class-name",
    required=True,
    type=str,
    help="TCGA class to generate few-shot files",
)
parser.add_argument(
    "--survival-wsi-feature-dir",
    default="./input/wsi_feature/phikon/TCGA_FEATURE/TCGA_BRCA",
    type=str,
    help="Path to WSI feature directory",
)
parser.add_argument(
    "--subtyping-wsi-feature-dir",
    default="./input/wsi_feature/phikon/TCGA_FEATURE",
    type=str,
    help="Path to WSI feature directory",
)
parser.add_argument(
    "--rna-feature-csv",
    default="./input/pruned_rna_feature/TCGA_BRCA_pruned_rna.csv",
    type=str,
    help="Path to omics csv file",
)
parser.add_argument(
    "--survival-csv",
    default="./input/survival/mutsig/tcga_brca_all_clean.csv.zip",
    type=str,
    help="Path to survival csv file",
)
parser.add_argument(
    "--split-dir",
    default="./splits/5foldcv/tcga_brca",
    type=str,
    help="Path to cross validation split files",
)
parser.add_argument(
    "--num-wsi-feature-tokens",
    default=2048,
    type=int,
    help="Number of WSI feature tokens",
)
parser.add_argument(
    "--num-bins", default=4, type=int, help="Number of bins for survival task"
)
parser.add_argument(
    "--subtyping-classes",
    default=["TCGA-BRCA-IDC", "TCGA-BRCA-ILC"],
    type=str,
    nargs="+",
    help="Subtyping classes",
)
parser.add_argument(
    "--tasks",
    default=["survival", "subtyping"],
    type=str,
    nargs="+",
    help="Tasks to generate few-shot files",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument(
    "--shots", default=[10], type=int, nargs="+", help="Number of shots"
)
parser.add_argument("-k", "--k", default=5, type=int, help="Number of folds")
parser.add_argument(
    "--output-dir", default="./splits", type=str, help="Output directory"
)


def main():
    utils.setup_default_logging()
    args = parser.parse_args()
    utils.random_seed(args.seed)

    for task in args.tasks:
        if task == "survival":
            dataset = TCGAWSIRNASurvivalDataset(
                wsi_feature_dir=args.survival_wsi_feature_dir,
                rna_feature_csv=args.rna_feature_csv,
                survival_csv=args.survival_csv,
                num_wsi_feature_tokens=args.num_wsi_feature_tokens,
                splits=args.split_dir,
                k=args.k,
                num_bins=args.num_bins,
                cache=False,
            )
        elif task == "subtyping":
            dataset = TCGAWSIRNASubtypingDataset(  # type: ignore[assignment]
                wsi_feature_dir=args.subtyping_wsi_feature_dir,
                rna_feature_csv=args.rna_feature_csv,
                classes=args.subtyping_classes,
                num_wsi_feature_tokens=args.num_wsi_feature_tokens,
                splits=args.split_dir,
                k=args.k,
                cache=False,
            )
        else:
            raise ValueError(f"Invalid task: {task}")
        for k in range(args.k):
            dataset.update_fold_nb(k)
            for shot in args.shots:
                _logger.info(
                    f"Generating {shot}-shot files for {task} task on fold {k}"
                )

                if task == "survival":
                    labels = [_ for _ in range(args.num_bins)]
                    train_feature_ids_by_label: dict[int, list[int]] = {
                        label: [] for label in labels
                    }
                    for idx in dataset.train_feature_ids:
                        label = int(
                            dataset.survival_data.loc[
                                dataset.survival_data[dataset.slide_id_column]
                                == idx[:15]
                            ]["disc_label"].iloc[0]
                        )
                        train_feature_ids_by_label[label].append(idx)  # type: ignore[arg-type]
                elif task == "subtyping":
                    train_feature_ids_by_label = {
                        label: [] for label in dataset.class_label.values()  # type: ignore[attr-defined]
                    }
                    for idx in dataset.train_feature_ids:
                        label = dataset.class_dict[idx]  # type: ignore[attr-defined]
                        train_feature_ids_by_label[label].append(idx)  # type: ignore[arg-type]
                else:
                    raise ValueError(f"Invalid task: {task}")

                support_set = []
                for label in train_feature_ids_by_label:
                    while True:
                        flag = False
                        support_set_cur_label = random.choices(
                            train_feature_ids_by_label[label], k=shot
                        )
                        for idx in support_set_cur_label:  # type: ignore[assignment]
                            if task == "survival":
                                wsi_feature = torch.load(
                                    os.path.join(dataset.wsi_feature_dir, f"{idx}.pt")
                                )
                            elif task == "subtyping":
                                wsi_feature = torch.load(
                                    os.path.join(
                                        dataset.wsi_feature_dir,
                                        list(dataset.class_label.keys())[label],  # type: ignore[attr-defined]
                                        f"{idx}.pt",
                                    )
                                )
                            if len(wsi_feature) < args.num_wsi_feature_tokens:
                                flag = True
                                break
                        if not flag:
                            break
                    support_set.extend(support_set_cur_label)

                support_set = [_[:12] for _ in support_set]
                query_set = [_[:12] for _ in dataset.val_feature_ids]
                df = pd.DataFrame(
                    {"train": pd.Series(support_set), "val": pd.Series(query_set)}
                )

                output_file = os.path.join(
                    args.output_dir,
                    task,
                    f"{args.k}foldcv",
                    f"{shot}-shot",
                    args.class_name,
                    f"splits_{k}.csv",
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file)

    _logger.info("Done")


if __name__ == "__main__":
    main()
