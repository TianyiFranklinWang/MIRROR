import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import KFold
from timm import utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Generate 5-fold CV splits from a directory of files."
)
parser.add_argument(
    "--root", type=str, required=True, help="Path to features directory."
)
parser.add_argument(
    "--class-name",
    required=True,
    type=str,
    help="TCGA class to generate few-shot files",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./splits/5foldcv",
    help="Path to save the generated CSV splits.",
)
parser.add_argument(
    "--n-splits",
    type=int,
    default=5,
    help="Number of folds for cross-validation (default: 5)",
)
parser.add_argument(
    "--random-seed", type=int, default=42, help="Random seed (default: 42)"
)


def main():
    utils.setup_default_logging()
    args = parser.parse_args()

    _logger.info(f"Starting 5-fold CV split generation for class: {args.class_name}")
    _logger.info(f"Feature root: {args.root}")
    _logger.info(f"Output directory: {args.output_dir}")
    _logger.info(f"Number of splits: {args.n_splits}")
    _logger.info(f"Random seed: {args.random_seed}")

    features = os.listdir(args.root)
    _logger.info(f"Found {len(features)} feature files.")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)

    if not os.path.exists(os.path.join(args.output_dir, args.class_name)):
        os.makedirs(os.path.join(args.output_dir, args.class_name))
        _logger.info(
            f"Created output directory: {os.path.join(args.output_dir, args.class_name)}"
        )

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        support_set = [features[i][:12] for i in train_idx]
        query_set = [features[i][:12] for i in val_idx]

        df_split = pd.DataFrame(
            {"train": pd.Series(support_set), "val": pd.Series(query_set)}
        )
        df_split.to_csv(
            os.path.join(args.output_dir, args.class_name, f"splits_{fold}.csv")
        )

        _logger.info(f"Fold {fold}: Train {len(support_set)}, Val {len(query_set)}")

    _logger.info("All folds generated successfully.")


if __name__ == "__main__":
    main()
