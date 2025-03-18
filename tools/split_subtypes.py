import argparse
import glob
import logging
import os

import pandas as pd
from timm import utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Split TCGA dataset based on oncotree codes"
)
parser.add_argument("--input-folder", type=str, required=True, help="Input folder")
parser.add_argument(
    "--oncotree-code-csv",
    type=str,
    default="./input/survival/brca_tcga_pan_can_atlas_2018_clinical_data.csv",
    help="Oncotree code csv",
)
parser.add_argument(
    "--target-oncotree-codes",
    type=str,
    nargs="+",
    default=["IDC", "ILC"],
    help="Target oncotree codes",
)


def main():
    utils.setup_default_logging()
    args = parser.parse_args()

    args.input_folder = os.path.abspath(args.input_folder)

    project_code = os.path.basename(args.input_folder)
    base_folder = os.path.dirname(args.input_folder)
    _logger.info(f"Project to split: {project_code}")

    df = pd.read_csv(args.oncotree_code_csv)
    df_oncotree_code = df["Oncotree Code"].unique()
    if not set(args.target_oncotree_codes).issubset(set(df_oncotree_code)):
        raise ValueError("Invalid oncotree codes")

    for target_oncotree_code in args.target_oncotree_codes:
        os.makedirs(
            os.path.join(
                base_folder, f"{project_code}{project_code[4]}{target_oncotree_code}"
            ),
            exist_ok=True,
        )
        _logger.info(
            f"Created folder: {project_code}{project_code[4]}{target_oncotree_code}"
        )

    for slide_id, oncotree_code in zip(df["Sample ID"], df["Oncotree Code"]):
        if oncotree_code in args.target_oncotree_codes:
            feature_file = slide_id + "-DX*.pt"

            target_file = glob.glob(os.path.join(args.input_folder, feature_file))[0]
            dest_file = os.path.join(
                base_folder,
                f"{project_code}{project_code[4]}{oncotree_code}",
                feature_file,
            )
            if os.path.exists(target_file):
                os.symlink(target_file, dest_file)

    _logger.info("Done")


if __name__ == "__main__":
    main()
