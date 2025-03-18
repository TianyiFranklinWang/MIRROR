import argparse
import gc
import glob
import logging
import os

import cv2
import numpy as np
import openslide
import skimage
from timm import utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate patch for Whole Slide Image")
parser.add_argument(
    "--input-dir",
    type=str,
    default="./input/wsi/TCGA",
    help="The directory of the input WSIs",
)
parser.add_argument(
    "--cohorts",
    nargs="+",
    type=str,
    default=["TCGA_BRCA"],
    help="The cohort of the input WSIs",
)
parser.add_argument(
    "--target-mag", type=int, default=20, help="The target magnification of the wsi"
)
parser.add_argument(
    "--patch-size", type=int, default=512, help="The size of the output patch"
)
parser.add_argument(
    "--pad-value", type=int, default=255, help="The value of the padding"
)
parser.add_argument(
    "--blur-ksize", type=int, default=7, help="The kernel size of the blur"
)
parser.add_argument(
    "--close-ksize", type=int, default=5, help="The kernel size of the closing"
)
parser.add_argument(
    "--erode-ksize", type=int, default=10, help="The kernel size of the erosion"
)
parser.add_argument(
    "--area-small-holes", type=int, default=16384, help="The area of the small holes"
)
parser.add_argument(
    "--min-size-small-objects",
    type=int,
    default=8192,
    help="The min size of the small objects",
)
parser.add_argument(
    "--connectivity", type=int, default=8, help="The connectivity of the small objects"
)
parser.add_argument(
    "--output-type", type=str, default="jpeg", help="The type of the output patch"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./input/wsi_patch/TCGA",
    help="The directory of the output patch",
)


def pad_slide(image, patch_size, pad_value):
    shape = image.shape
    pad0, pad1 = (
        int(patch_size - (shape[0] % patch_size)),
        int(patch_size - (shape[1] % patch_size)),
    )
    if len(shape) == 3:
        pad_image = np.pad(
            image,
            [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
            constant_values=pad_value,
        )
    elif len(shape) == 2:
        pad_image = np.pad(
            image,
            [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]],
            constant_values=pad_value,
        )
    else:
        raise ValueError("Invalid shape")
    return pad_image


def segment_foreground(
    image,
    blur_ksize,
    close_ksize,
    erode_ksize,
    area_small_holes,
    min_size_small_objects,
    connectivity,
):
    mask = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.medianBlur(mask[:, :, 1], ksize=blur_ksize)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8)
    )
    mask = cv2.erode(mask, np.ones((erode_ksize, erode_ksize), np.uint8))
    mask = skimage.morphology.remove_small_holes(
        mask > 0, area_threshold=area_small_holes, connectivity=connectivity
    )
    mask = skimage.morphology.remove_small_objects(
        mask, min_size=min_size_small_objects, connectivity=connectivity
    )
    mask = mask.astype(np.uint8) * 255
    return mask


def patchify(image, patch_size):
    shape = image.shape
    if len(shape) == 2:
        patches = image.reshape(
            shape[0] // patch_size, patch_size, shape[1] // patch_size, patch_size
        )
        patches = patches.transpose(0, 2, 1, 3)
        patches = patches.reshape(-1, patch_size, patch_size)
    elif len(shape) == 3:
        patches = image.reshape(
            shape[0] // patch_size, patch_size, shape[1] // patch_size, patch_size, 3
        )
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, patch_size, patch_size, 3)
    else:
        raise ValueError("Invalid shape")
    return patches


def main():
    utils.setup_default_logging()
    args = parser.parse_args()

    for cohort in args.cohorts:
        _logger.info(f"Processing {cohort}")

        cohort_dir = os.path.join(args.input_dir, cohort)
        if not os.path.exists(cohort_dir):
            raise ValueError(f"Input directory {cohort_dir} does not exist")
        _logger.info(f"Input directory: {cohort_dir}")

        cohort_output_dir = os.path.join(args.output_dir, cohort)
        os.makedirs(cohort_output_dir, exist_ok=True)

        wsi_files = glob.glob(os.path.join(cohort_dir, "*.svs"))
        _logger.info(f"Found {len(wsi_files)} WSI files")

        for i, wsi_file in enumerate(wsi_files):
            _logger.info(f"Processing {wsi_file} ({i + 1}/{len(wsi_files)})")

            slide_output_dir = os.path.join(
                cohort_output_dir, os.path.basename(wsi_file).removesuffix(".svs")
            )
            os.makedirs(slide_output_dir, exist_ok=True)

            slide = openslide.OpenSlide(wsi_file)
            size = slide.level_dimensions[0]
            factor = int(slide.properties.get("aperio.AppMag")) / args.target_mag  # type: ignore[arg-type]
            target_size = (size[0] // factor, size[1] // factor)
            slide = slide.get_thumbnail(target_size)  # type: ignore[assignment]
            slide = np.asarray(slide)  # type: ignore[assignment]

            slide = pad_slide(slide, args.patch_size, args.pad_value)
            mask = segment_foreground(
                slide,
                args.blur_ksize,
                args.close_ksize,
                args.erode_ksize,
                args.area_small_holes,
                args.min_size_small_objects,
                args.connectivity,
            )

            patches = patchify(slide, args.patch_size)
            mask_patches = patchify(mask, args.patch_size)
            selected_idx = [
                idx
                for idx, thresh_patch in enumerate(mask_patches)
                if thresh_patch.sum() > 0
            ]
            if len(selected_idx) == 0:
                _logger.info(f"Segment slide {wsi_file} failed, select all patches")
                selected_idx = [idx for idx, thresh_patch in enumerate(mask_patches)]
            for idx, patch in enumerate(patches):
                if idx in selected_idx:
                    coord = (
                        idx // (size[0] // args.patch_size),
                        idx % (size[0] // args.patch_size),
                    )
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)  # noqa: PLW2901
                    cv2.imwrite(
                        os.path.join(
                            slide_output_dir,
                            f"{coord[0]}_{coord[1]}.{args.output_type}",
                        ),
                        patch,
                    )
            del slide, mask, patches, mask_patches
            gc.collect()
    _logger.info("Done")


if __name__ == "__main__":
    main()
