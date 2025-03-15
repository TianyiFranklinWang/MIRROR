"""Subtyping Training Script
Copyright (c) 2024, Tianyi Wang @ The University of Sydney
All rights reserved.

Based on the timm codebase by Ross Wightman
https://github.com/huggingface/pytorch-image-models

Licensed under the GNU General Public License v3.0, see LICENSE for details
"""

import argparse
import importlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import yaml
from timm import utils
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.loader import _worker_init
from timm.layers import convert_sync_batchnorm, set_fast_norm, trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import (
    create_model,
    load_checkpoint,
    model_parameters,
    resume_checkpoint,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAUROC, MulticlassF1Score
from torcheval.metrics.toolkit import sync_and_compute

import models  # noqa: F401
from datasets import TCGAWSIRNASubtypingDataset


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP, convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if torch.cuda.amp.autocast is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

has_compile = hasattr(torch, "compile")

_logger = logging.getLogger("train")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="PyTorch Survival Prediction Training")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")
# Keep this argument outside the dataset group because it is positional.
parser.add_argument(
    "data",
    nargs="?",
    metavar="DIR",
    const=None,
    help="path to dataset (positional is *deprecated*, use --data-dir)",
)
parser.add_argument(
    "--wsi-feature-dir", metavar="DIR", help="path to wsi feature dataset"
)
parser.add_argument("--rna-feature-csv", metavar="PATH", help="path to omics csv file")
parser.add_argument("--classes", nargs="+", metavar="CLASS", help="list of classes")
parser.add_argument(
    "--split-dir", metavar="DIR", help="path to cross validation split files"
)
parser.add_argument(
    "--num-wsi-feature-tokens",
    type=int,
    default=2048,
    metavar="N",
    help="number of wsi feature tokens sampled",
)
parser.add_argument(
    "--k", "-k", type=int, default=0, metavar="N", help="total fold number"
)
parser.add_argument("--fold-nb", type=int, default=0, metavar="N", help="fold number")
parser.add_argument(
    "--wsi-feature-only",
    action="store_true",
    default=False,
    help="use only wsi features for training",
)
parser.add_argument(
    "--cache", action="store_true", default=False, help="cache dataset in memory"
)
group.add_argument(
    "--val", action="store_true", default=False, help="enable validation"
)

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument("--model", type=str, metavar="MODEL", help="Name of model to train")
group.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Load this checkpoint into model after initialization (default: none)",
)
group.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
group.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
group.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
group.add_argument(
    "--in-chans",
    type=int,
    default=None,
    metavar="N",
    help="Image input channels (default: None => 3)",
)
group.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="Input batch size for training (default: 128)",
)
group.add_argument(
    "-vb",
    "--validation-batch-size",
    type=int,
    default=None,
    metavar="N",
    help="Validation batch size override (default: None)",
)
group.add_argument(
    "--fuser",
    default="",
    type=str,
    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')",
)
group.add_argument(
    "--grad-accum-steps",
    type=int,
    default=1,
    metavar="N",
    help="The number of steps to accumulate gradients (default: 1)",
)
group.add_argument(
    "--grad-checkpointing",
    action="store_true",
    default=False,
    help="Enable gradient checkpointing through model blocks/stages",
)
group.add_argument(
    "--fast-norm",
    default=False,
    action="store_true",
    help="enable experimental fast-norm",
)
group.add_argument("--model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument(
    "--init-head",
    action="store_true",
    default=False,
    help="initialize head layer parameters",
)
group.add_argument(
    "--head-init-scale", default=None, type=float, help="Head initialization scale"
)
group.add_argument(
    "--head-init-bias", default=None, type=float, help="Head initialization bias value"
)

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="torch.jit.script the full model",
)
scripting_group.add_argument(
    "--torchcompile",
    nargs="?",
    type=str,
    default=None,
    const="inductor",
    help="Enable compilation w/ specified backend (default: inductor).",
)

# Device & distributed
group = parser.add_argument_group("Device parameters")
group.add_argument(
    "--device", default="cuda", type=str, help="Device (accelerator) to use."
)
group.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
group.add_argument(
    "--amp-dtype",
    default="float16",
    type=str,
    help="lower precision AMP dtype (default: float16)",
)
group.add_argument(
    "--amp-impl",
    default="native",
    type=str,
    help='AMP impl to use, "native" or "apex" (default: native)',
)
group.add_argument(
    "--no-ddp-bb",
    action="store_true",
    default=False,
    help="Force broadcast buffers for native DDP to off.",
)
group.add_argument(
    "--synchronize-step",
    action="store_true",
    default=False,
    help="torch.cuda.synchronize() end of each step",
)
group.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--device-modules",
    default=None,
    type=str,
    nargs="+",
    help="Python imports for device backend modules.",
)

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
group.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
group.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
group.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
group.add_argument(
    "--weight-decay", type=float, default=2e-5, help="weight decay (default: 2e-5)"
)
group.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
group.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)
group.add_argument(
    "--layer-decay",
    type=float,
    default=None,
    help="layer-wise learning rate decay (default: None)",
)
group.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument(
    "--use-sched", action="store_true", default=False, help="scheduler on/off"
)
group.add_argument(
    "--sched",
    type=str,
    default="cosine",
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
group.add_argument(
    "--sched-on-updates",
    action="store_true",
    default=False,
    help="Apply LR scheduler step on update instead of epoch end.",
)
group.add_argument(
    "--lr",
    type=float,
    default=None,
    metavar="LR",
    help="learning rate, overrides lr-base if set (default: None)",
)
group.add_argument(
    "--lr-base",
    type=float,
    default=0.1,
    metavar="LR",
    help="base learning rate: lr = lr_base * global_batch_size / base_size",
)
group.add_argument(
    "--lr-base-size",
    type=int,
    default=256,
    metavar="DIV",
    help="base learning rate batch size (divisor, default: 256).",
)
group.add_argument(
    "--lr-base-scale",
    type=str,
    default="",
    metavar="SCALE",
    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)',
)
group.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
group.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
group.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
group.add_argument(
    "--lr-cycle-decay",
    type=float,
    default=0.5,
    metavar="MULT",
    help="amount to decay each learning rate cycle (default: 0.5)",
)
group.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit, cycles enabled if > 1",
)
group.add_argument(
    "--lr-k-decay",
    type=float,
    default=1.0,
    help="learning rate k-decay for cosine/poly (default: 1.0)",
)
group.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="warmup learning rate (default: 1e-5)",
)
group.add_argument(
    "--min-lr",
    type=float,
    default=0,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (default: 0)",
)
group.add_argument(
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 300)",
)
group.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
group.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
group.add_argument(
    "--decay-milestones",
    default=[90, 180, 270],
    type=int,
    nargs="+",
    metavar="MILESTONES",
    help="list of decay epoch indices for multistep lr. must be increasing",
)
group.add_argument(
    "--decay-epochs",
    type=float,
    default=90,
    metavar="N",
    help="epoch interval to decay LR",
)
group.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
group.add_argument(
    "--warmup-prefix",
    action="store_true",
    default=False,
    help="Exclude warmup period from decay schedule.",
),
group.add_argument(
    "--cooldown-epochs",
    type=int,
    default=0,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
group.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10)",
)
group.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument(
    "--loss",
    type=str,
    choices=["ce_loss"],
    default="ce_loss",
    help="Loss function (default: ce_loss)",
)
group.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group(
    "Batch norm parameters", "Only works with gen_efficientnet based models currently."
)
group.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
group.add_argument(
    "--dist-bn",
    type=str,
    default="reduce",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)

# Model Exponential Moving Average
group = parser.add_argument_group("Model exponential moving average parameters")
group.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights.",
)
group.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
group.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="Decay factor for model weights moving average (default: 0.9998)",
)
group.add_argument(
    "--model-ema-warmup", action="store_true", help="Enable warmup for model EMA decay."
)

# Misc
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
group.add_argument(
    "--worker-seeding", type=str, default="all", help="worker seed mode (default: all)"
)
group.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status",
)
group.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
group.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
group.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
group.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
group.add_argument(
    "--drop-last",
    action="store_true",
    default=False,
    help="drop last batch in train DataLoader",
)
group.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
group.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
group.add_argument(
    "--eval-metric",
    default="acc",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "acc")',
)
group.add_argument(
    "--eval-metric-average",
    default="weighted",
    help='Average method for multiclass metrics (default: "weighted")',
)
group.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)
group.add_argument(
    "--wandb-project",
    default="",
    type=str,
    metavar="NAME",
    help="Wandb project name (default: None)",
)
parser.add_argument(
    "--linear_probe", action="store_true", default=False, help="linear probe mode"
)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config) as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            use_amp = "apex"
            assert args.amp_dtype == "float16"
        else:
            assert (
                has_native_amp
            ), "Please update PyTorch to a version with native AMP (or use APEX)."
            use_amp = "native"
            assert args.amp_dtype in ("float16", "bfloat16")
        if args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        scriptable=args.torchscript,
        checkpoint_path=None,
        **args.model_kwargs,
    )
    if args.initial_checkpoint:
        incompatible_keys = load_checkpoint(
            model, args.initial_checkpoint, strict=False
        )
        if incompatible_keys is not None:
            _logger.warning(f"Incompatible keys: {incompatible_keys}")
    if args.init_head:
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.linear_probe:
        model.head.weight.data.normal_(mean=0.0, std=0.01)
        model.head.bias.data.zero_()

        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly
    assert args.num_classes == len(
        args.classes
    ), "Number of classes must match number of class names."

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}."
        )

    # move model to GPU
    model.to(device=device)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ""  # disable dist_bn when sync BN active
        if has_apex and use_amp == "apex":
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not args.torchcompile
        assert use_amp != "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = (
                "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
            )
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
            )

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        assert device.type == "cuda"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        try:
            amp_autocast = partial(  # type: ignore[assignment]
                torch.autocast, device_type=device.type, dtype=amp_dtype
            )
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == "cuda"
            amp_autocast = torch.cuda.amp.autocast  # type: ignore[assignment]
        if device.type == "cuda" and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()  # type: ignore[assignment]
        if utils.is_primary(args):
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,  # type: ignore[arg-type]
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device="cpu" if args.model_ema_force_cpu else None,  # type: ignore[arg-type]
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)  # type: ignore[assignment]

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb
            )
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert (
            has_compile
        ), "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        model = torch.compile(model, backend=args.torchcompile)

    dataset_train = TCGAWSIRNASubtypingDataset(
        wsi_feature_dir=args.wsi_feature_dir,
        rna_feature_csv=args.rna_feature_csv,
        classes=args.classes,
        num_wsi_feature_tokens=args.num_wsi_feature_tokens,
        splits=args.split_dir,
        k=args.k,
        wsi_feature_only=args.wsi_feature_only,
        cache=args.cache,
    )
    dataset_train.update_fold_nb(args.fold_nb)
    dataset_train.train()
    if args.val:
        dataset_eval = TCGAWSIRNASubtypingDataset(
            wsi_feature_dir=args.wsi_feature_dir,
            rna_feature_csv=args.rna_feature_csv,
            classes=args.classes,
            num_wsi_feature_tokens=args.num_wsi_feature_tokens,
            splits=args.split_dir,
            k=args.k,
            wsi_feature_only=args.wsi_feature_only,
            cache=args.cache,
        )
        dataset_eval.update_fold_nb(args.fold_nb)
        dataset_eval.val()

    # create data loaders w/ augmentation pipeline
    sampler_train = None
    if args.distributed and not isinstance(
        dataset_train, torch.utils.data.IterableDataset
    ):
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    args.batch_size = (
        args.batch_size
        if args.batch_size * args.world_size < len(dataset_train)
        else max(2 ** int(math.log2(len(dataset_train))) // args.world_size, 2)
    )
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        shuffle=True if sampler_train is None else False,  # noqa: SIM210
        sampler=sampler_train,
        worker_init_fn=partial(_worker_init, worker_seeding=args.worker_seeding),
        persistent_workers=True,
    )

    loader_eval = None
    if args.val:
        sampler_eval = None
        if args.distributed and not isinstance(
            dataset_eval, torch.utils.data.IterableDataset
        ):
            sampler_eval = OrderedDistributedSampler(dataset_eval)
        if args.validation_batch_size is not None:
            args.validation_batch_size = (
                args.validation_batch_size
                if args.validation_batch_size * args.world_size < len(dataset_eval)
                else max(2 ** int(math.log2(len(dataset_eval))) // args.world_size, 2)
            )
        loader_eval = DataLoader(
            dataset_eval,
            batch_size=(
                args.validation_batch_size
                if args.validation_batch_size
                else args.batch_size
            ),
            drop_last=args.drop_last,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            shuffle=False,
            sampler=sampler_eval,
            worker_init_fn=partial(_worker_init, worker_seeding=args.worker_seeding),
            persistent_workers=True,
        )

    # setup loss function
    if args.loss == "ce_loss":
        if args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()  # type: ignore[assignment]
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else "loss"
    decreasing_metric = eval_metric == "loss"
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    f"fold{args.fold_nb}",
                    f"k{args.k}",
                ]
            )
        output_dir = utils.get_outdir(
            args.output if args.output else "./output/train", exp_name
        )
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(
                project=args.wandb_project,
                name=exp_name,
                config=args,
                dir=output_dir,
                save_code=True,
            )
            wandb.watch(
                model,
                log="all",
                log_freq=max(50, args.log_interval),
            )
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )
            raise ValueError("wandb package not found")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (
        len(loader_train) + args.grad_accum_steps - 1
    ) // args.grad_accum_steps
    lr_scheduler = None
    if args.use_sched:
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
            updates_per_epoch=updates_per_epoch,
        )
    else:
        num_epochs = args.epochs
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(f"Scheduled epochs: {num_epochs}.")
        if lr_scheduler is not None:
            _logger.info(
                f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
            )

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                device=device,
                lr_scheduler=lr_scheduler,
                saver=saver,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
            )

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                        utils.distribute_bn(
                            model_ema, args.world_size, args.dist_bn == "reduce"
                        )

                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=" (EMA)",
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, "summary.csv"),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=latest_metric
                )

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            results.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": eval_metrics,
                }
            )

    except KeyboardInterrupt:
        pass

    results = {"all": results}  # type: ignore[assignment]
    if best_metric is not None:
        results["best"] = results["all"][best_epoch - start_epoch]  # type: ignore[call-overload, operator]
        _logger.info(f"*** Best metric: {best_metric} (epoch {best_epoch})")
    print(f"--result\n{json.dumps(results, indent=4)}")
    if args.log_wandb and has_wandb:
        wandb.finish()


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device=torch.device("cuda"),  # noqa: B008
    lr_scheduler=None,
    saver=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
):
    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, batch in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        rna_features = None
        if not args.wsi_feature_only:
            wsi_features, rna_features, labels = batch
        else:
            wsi_features, labels = batch

        wsi_features = wsi_features.to(device)
        rna_features = rna_features.to(device)  # type: ignore[union-attr]
        rna_features = rna_features.to(device) if rna_features is not None else None

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                if rna_features is None:  # noqa: B023
                    output = model(wsi_features)  # noqa: B023
                else:
                    output = model(wsi_features, rna_features)  # noqa: B023
                loss = loss_fn(output, labels)  # noqa: B023
            if accum_steps > 1:  # noqa: B023
                loss /= accum_steps  # noqa: B023
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(
                        model, exclude_head="agc" in args.clip_mode
                    ),
                    create_graph=second_order,
                    need_update=need_update,  # noqa: B023
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:  # noqa: B023
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(
                                model, exclude_head="agc" in args.clip_mode
                            ),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, wsi_features.shape[0])
        update_sample_count += wsi_features.shape[0]

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step and device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(
                    reduced_loss.item() * accum_steps, wsi_features.shape[0]
                )
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch - 1} "
                    f"({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

        if (
            saver is not None
            and args.recovery_interval
            and ((update_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    model,
    loader,
    loss_fn,
    args,
    device=torch.device("cuda"),  # noqa: B008
    amp_autocast=suppress,
    log_suffix="",
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    acc_m = utils.AverageMeter()
    auc_m = MulticlassAUROC(
        num_classes=args.num_classes, average=args.eval_metric_average, device=device
    )
    f1_m = MulticlassF1Score(
        num_classes=args.num_classes, average=args.eval_metric_average, device=device
    )

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            last_batch = batch_idx == last_idx

            rna_features = None
            if not args.wsi_feature_only:
                wsi_features, rna_features, labels = batch
            else:
                wsi_features, labels = batch

            wsi_features = wsi_features.to(device)
            rna_features = rna_features.to(device)  # type: ignore[union-attr]
            rna_features = rna_features.to(device) if rna_features is not None else None

            with amp_autocast():
                if rna_features is None:
                    output = model(wsi_features)
                else:
                    output = model(wsi_features, rna_features)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = loss_fn(output, labels)
            acc = utils.accuracy(output, labels, topk=(1,))[0] / 100
            auc_m.update(output, labels)
            f1_m.update(output, labels)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc = utils.reduce_tensor(acc, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), wsi_features.shape[0])
            acc_m.update(acc.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc: {acc_m.val:>7.3f} ({acc_m.avg:>7.3f})"
                )

    if args.distributed:
        auc = sync_and_compute(auc_m).item()
        f1 = sync_and_compute(f1_m).item()
    else:
        auc = auc_m.compute().item()
        f1 = f1_m.compute().item()

    if utils.is_primary(args):
        _logger.info(
            f" * Loss: {losses_m.avg:>7.3f}  Acc: {acc_m.avg:>7.3f}  AUC: {auc:>7.3f}  F1: {f1:>7.3f}"
        )

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("acc", acc_m.avg), ("auc", auc), ("f1", f1)]
    )

    return metrics


if __name__ == "__main__":
    main()
