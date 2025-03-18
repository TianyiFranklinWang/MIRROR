import argparse
import glob
import logging
import os
from datetime import datetime

import torch
from timm import utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Extract teacher weights from latest CV")
parser.add_argument(
    "--result-dir", type=str, required=True, help="The directory of the training output"
)
parser.add_argument("-k", "--k", type=int, default=5, help="The number of folds in CV")
parser.add_argument(
    "--weight-file", type=str, default="last.pth.tar", help="Path to the weight file"
)


def parse_folder_name(folder_name):
    date_part, time_part = folder_name.split("/")[-1].split("-")[:2]
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")


def main(args):
    utils.setup_default_logging()

    device = torch.device("cpu")

    folders = []
    for i in range(args.k):
        current_fold_folders = glob.glob(os.path.join(args.result_dir, f"*fold{i}*"))
        sorted_folders = sorted(
            current_fold_folders, key=parse_folder_name, reverse=True
        )
        folders.append(sorted_folders[0])
    _logger.info(f"Folders: {folders}")

    for folder in folders:
        weight_file = os.path.join(folder, args.weight_file)
        _logger.info(f"Loading global weights from {weight_file}")
        global_weight = torch.load(weight_file, map_location=device)
        wsi_state_dict = {}
        rna_state_dict = {}
        for key, val in global_weight["state_dict"].items():
            if "wsi_encoder" in key:
                if "mask_token" not in key:
                    wsi_state_dict[key.replace("wsi_encoder.", "")] = val
            if "rna_encoder" in key:
                if "mask_token" not in key:
                    rna_state_dict[key.replace("rna_encoder.", "")] = val
        wsi_weight = {
            "epoch": global_weight["epoch"],
            "arch": global_weight["arch"],
            "state_dict": wsi_state_dict,
            "version": global_weight["version"],
            "args": global_weight["args"],
            "metric": global_weight["metric"],
        }
        rna_weight = {
            "epoch": global_weight["epoch"],
            "arch": global_weight["arch"],
            "state_dict": rna_state_dict,
            "version": global_weight["version"],
            "args": global_weight["args"],
            "metric": global_weight["metric"],
        }

        orig_file_path, orig_file_name = os.path.split(weight_file)
        wsi_output_file = os.path.join(orig_file_path, f"wsi_{orig_file_name}")
        rna_output_file = os.path.join(orig_file_path, f"rna_{orig_file_name}")
        torch.save(wsi_weight, wsi_output_file)
        torch.save(rna_weight, rna_output_file)
        _logger.info(f"Saved WSI weights to {wsi_output_file}")
        _logger.info(f"Saved RNA weights to {rna_output_file}")
    _logger.info("Done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
