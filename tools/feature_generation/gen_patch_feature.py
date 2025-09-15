import gc
import json
import logging
import os

import torch
from dataset_feature_generation import KFoldPatchDataset, PatchDataset
from timm import utils


class Config:
    def __init__(self):
        self.model = "resnet50"

        self.input_folder = "./input/wsi_patch/TCGA"
        self.classes = ["TCGA_BRCA"]
        self.output_folder = "./input/wsi_feature/phikon/TCGA_FEATURE"

        self.dataset_type = "patch"
        self.parallel = True
        self.k = None
        self.fold_nb = None
        self.dataset_num_workers = 8
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.batch_size = 64
        self.loader_num_workers = 2

        if self.model == "resnet50":
            self.device_type = "cuda:0"
            self.model = "custom_resnet50"
            self.pretrained = True
            self.input_size = 224
        elif self.model == "phikon":
            self.device_type = "cuda:0"
            self.input_size = 224
            self.model = "phikon"
        else:
            raise ValueError(f"Model {self.model} not found")


_logger = logging.getLogger(__name__)


def main():
    utils.setup_default_logging()
    config = Config()

    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)
        _logger.info(f"Create output folder at {config.output_folder}")
    for class_name in config.classes:
        if not os.path.exists(os.path.join(config.output_folder, class_name)):
            os.makedirs(os.path.join(config.output_folder, class_name))
            _logger.info(
                f"Create output folder for {class_name} at {os.path.join(config.output_folder, class_name)}"
            )

    with open(os.path.join(config.output_folder, "config.json"), "w") as f:
        config_json = json.dumps(config.__dict__)
        json.dump(config_json, f)
        _logger.info(
            f"Save configuration file at {os.path.join(config.output_folder, 'config.json')}"
        )

    if config.k is not None and config.fold_nb is not None:
        dataset = KFoldPatchDataset(
            k=config.k,
            root=config.input_folder,
            classes=config.classes,
            input_size=config.input_size,
            mean=config.mean,
            std=config.std,
            num_workers=config.dataset_num_workers,
            parallel=config.parallel,
        )
        dataset.update_fold(config.fold_nb)
    else:
        dataset = PatchDataset(  # type: ignore[assignment]
            root=config.input_folder,
            classes=config.classes,
            input_size=config.input_size,
            mean=config.mean,
            std=config.std,
            num_workers=config.dataset_num_workers,
            parallel=config.parallel,
        )
    _logger.info(f"Create dataset with {len(dataset)} slides")

    device = torch.device(config.device_type)
    model = None
    if config.model == "custom_resnet50":
        from feature_models import custom_resnet50  # noqa: PLC0415

        model = custom_resnet50(pretrained=config.pretrained)
    elif config.model == "phikon":
        from feature_models import Phikon  # noqa: PLC0415

        model = Phikon()
    if model:
        if hasattr(config, "checkpoint") and config.checkpoint:
            model.load_state_dict(torch.load(config.checkpoint), strict=False)
        model = model.to(device)
        model.eval()
    else:
        raise ValueError(f"Model {config.model} not found")
    _logger.info(f"Create {config.model} model to device {config.device_type}")
    if hasattr(config, "checkpoint") and config.checkpoint:
        _logger.info(f"Load checkpoint from {config.checkpoint}")

    hook = None
    if hasattr(config, "hook_layer_name") and config.hook_layer_name is not None:
        msg = f"Register forward hook to {config.hook_layer_name}"
        intermediate_outputs = list()
        layer = getattr(model, config.hook_layer_name)
        if hasattr(config, "hook_layer_idx") and config.hook_layer_idx is not None:
            msg += f"[{config.hook_layer_idx}]"
            layer = layer[config.hook_layer_idx]

        def forward_hook_fn(module, input, output):
            intermediate_outputs.append(output[:, 0])

        hook = layer.register_forward_hook(forward_hook_fn)
        _logger.info(msg)

    index_to_remove = list()
    for idx, (slide_label, slide) in enumerate(
        zip(dataset.slide_labels, dataset.slides)
    ):
        save_file = os.path.join(
            config.output_folder, slide_label, f"{slide.split('.')[0]}.pt"
        )
        if os.path.exists(save_file):
            index_to_remove.append(idx)
            _logger.info(f"Skip {slide_label} slide: {slide} features existed")
    for idx in sorted(index_to_remove, reverse=True):
        del dataset.slide_labels[idx]
        del dataset.slides[idx]

    with torch.no_grad():
        for idx, (slide_label, slide, patches) in enumerate(dataset):  # type: ignore[misc, arg-type]
            if len(patches) == 0:  # type: ignore[has-type]
                _logger.warning(f"Empty slide detected: {slide}    type: {slide_label}")
                continue
            save_file = os.path.join(
                config.output_folder, slide_label, f"{slide.split('.')[0]}.pt"
            )
            _logger.info(
                f"Process on: {slide}    type: {slide_label}    total_patches: {len(patches)}    progress: [{idx + 1}/{len(dataset)}]"  # type: ignore[has-type]
            )
            features = list()
            loader = torch.utils.data.DataLoader(
                patches,  # type: ignore[has-type]
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.loader_num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for batch in loader:
                batch = batch.to(device)  # noqa: PLW2901
                feature = model(batch)
                if hook is not None:
                    feature = intermediate_outputs.pop()
                features.append(feature.cpu())
            features = torch.cat(features, dim=0)  # type: ignore[assignment]
            torch.save(features, save_file)

            del slide_label, slide, patches, batch, feature, features  # type: ignore[has-type]
            gc.collect()
            torch.cuda.empty_cache()

    if hook is not None:
        hook.remove()

    _logger.info("Complete feature generation!")


if __name__ == "__main__":
    main()
