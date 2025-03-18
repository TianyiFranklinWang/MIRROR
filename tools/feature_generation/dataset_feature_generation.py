import concurrent.futures
import copy
import logging
import os

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset


_logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    def __init__(
        self,
        root,
        classes,
        input_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=1,
        parallel=True,
    ):
        self.root = root
        self.classes = classes
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.num_workers = num_workers
        self.parallel = parallel
        self.slide_labels, self.slides = self.gather_slides()
        if self.parallel:
            self.gatherer = self.parallel_gather_patches
        else:
            self.gatherer = self.gather_patches

    def gather_slides(self):
        slide_labels = list()
        slides = list()

        for class_name in self.classes:
            slides_per_class = os.listdir(os.path.join(self.root, class_name))
            slide_labels_per_class = [class_name for _ in range(len(slides_per_class))]

            slides += slides_per_class
            slide_labels += slide_labels_per_class

        return slide_labels, slides

    def __len__(self):
        return len(self.slides)

    def transform(self, patch):
        patch = cv2.resize(
            patch,
            (self.input_size, self.input_size),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_AREA,
        )
        patch = A.Normalize(
            mean=self.mean,
            std=self.std,
            max_pixel_value=255.0,
            always_apply=True,
            p=1.0,
        )(image=patch)["image"]
        patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch).float()
        return patch

    def gather_patches(self, slide_label, slide):
        patch_files = os.listdir(os.path.join(self.root, slide_label, slide))
        patches = list()
        for patch_file in patch_files:
            patch = cv2.imread(os.path.join(self.root, slide_label, slide, patch_file))
            if patch is None:
                _logger.warning(
                    f"Patch {os.path.join(self.root, slide_label, slide, patch_file)} empty"
                )
                continue
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = self.transform(patch)
            patches.append(patch)
        return patches

    def parallel_gather_patches(self, slide_label, slide):
        patch_files = os.listdir(os.path.join(self.root, slide_label, slide))
        patches = list()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(
                    cv2.imread, os.path.join(self.root, slide_label, slide, patch_file)
                )
                for patch_file in patch_files
            ]
            for patch_file, future in zip(
                patch_files, concurrent.futures.as_completed(futures)
            ):
                patch = future.result()
                if patch is None:
                    _logger.warning(
                        f"Patch {os.path.join(self.root, slide_label, slide, patch_file)} empty"
                    )
                    continue
                patches.append(patch)
        patches = [self.transform(_) for _ in patches]
        return patches

    def __getitem__(self, index):
        slide_label = self.slide_labels[index]
        slide = self.slides[index]
        patches = self.gatherer(slide_label, slide)
        return slide_label, slide, patches


class KFoldPatchDataset(PatchDataset):
    def __init__(self, k=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k = k
        self.global_slide_labels = copy.deepcopy(self.slide_labels)
        self.global_slides = copy.deepcopy(self.slides)
        self.update_fold(0)

    def update_fold(self, fold_nb):
        self.slide_labels = self.global_slide_labels[fold_nb :: self.k]
        self.slides = self.global_slides[fold_nb :: self.k]


class SinglePatchDataset(Dataset):
    def __init__(self, root, input_size, mean, std):
        self.root = root
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.patch_files = os.listdir(self.root)

    def transform(self, patch):
        patch = cv2.resize(
            patch,
            (self.input_size, self.input_size),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_AREA,
        )
        patch = A.Normalize(
            mean=self.mean,
            std=self.std,
            max_pixel_value=255.0,
            always_apply=True,
            p=1.0,
        )(image=patch)["image"]
        patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch).float()
        return patch

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch_file = self.patch_files[idx]
        patch = cv2.imread(os.path.join(self.root, patch_file))
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = self.transform(patch)
        return patch_file, patch
