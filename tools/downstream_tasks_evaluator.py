import argparse
import glob
import logging
import os
import subprocess
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from timm import utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Bulk evaluator for downstream tasks")
parser.add_argument("--gpu-count", type=int, default=1, help="Number of physical GPUs")
parser.add_argument(
    "--virtual-gpu-factor",
    type=int,
    default=1,
    help="Number of virtual GPUs per physical GPU (to run multiple jobs concurrently on the same GPU)",
)
parser.add_argument(
    "--result-dir", type=str, default=None, help="The directory of the training output"
)
parser.add_argument("-k", "--k", type=int, default=5, help="The number of folds in CV")
parser.add_argument(
    "--checkpoint-file", type=str, default=None, help="Checkpoint file name"
)
parser.add_argument(
    "--subtyping-launch-script",
    type=str,
    default="./scripts/run_train_subtyping.sh",
    help="Subtyping launch script",
)
parser.add_argument(
    "--survival-launch-script",
    type=str,
    default="./scripts/run_train_survival.sh",
    help="Survival launch script",
)
parser.add_argument(
    "--subtyping-linprob-config",
    type=str,
    default=None,
    help="Subtyping linear probing config file",
)
parser.add_argument(
    "--subtyping-finetune-config",
    type=str,
    default=None,
    help="Subtyping finetuning config file",
)
parser.add_argument(
    "--subtyping-1shot-config",
    type=str,
    default=None,
    help="Subtyping 1-shot config file",
)
parser.add_argument(
    "--subtyping-5shot-config",
    type=str,
    default=None,
    help="Subtyping 5-shot config file",
)
parser.add_argument(
    "--subtyping-10shot-config",
    type=str,
    default=None,
    help="Subtyping 10-shot config file",
)
parser.add_argument(
    "--survival-linprob-config",
    type=str,
    default=None,
    help="Survival linear probing config file",
)
parser.add_argument(
    "--survival-finetune-config",
    type=str,
    default=None,
    help="Survival finetuning config file",
)
parser.add_argument(
    "--survival-1shot-config",
    type=str,
    default=None,
    help="Survival 1-shot config file",
)
parser.add_argument(
    "--survival-5shot-config",
    type=str,
    default=None,
    help="Survival 5-shot config file",
)
parser.add_argument(
    "--survival-10shot-config",
    type=str,
    default=None,
    help="Survival 10-shot config file",
)


class GPUResourceManager:
    def __init__(self, num_gpus, virtual_gpu_factor=1):
        self.gpus = deque(
            gpu_id for _ in range(virtual_gpu_factor) for gpu_id in range(num_gpus)
        )
        self.condition = threading.Condition()
        self.gpu_status: dict[int, list[str]] = {
            gpu_id: [] for gpu_id in range(num_gpus)
        }

    def acquire_gpu(self, job_name):
        with self.condition:
            while not self.gpus:
                self.condition.wait()
            gpu_id = self.gpus.popleft()
            self.gpu_status[gpu_id].append(job_name)
            self.log_gpu_status()
            return gpu_id

    def release_gpu(self, gpu_id, job_name):
        with self.condition:
            self.gpus.append(gpu_id)
            self.gpu_status[gpu_id].remove(job_name)
            self.log_gpu_status()
            self.condition.notify()

    def get_gpu_status(self):
        return {
            gpu_id: jobs if jobs else ["free"]
            for gpu_id, jobs in self.gpu_status.items()
        }

    def log_gpu_status(self):
        status = self.get_gpu_status()
        status_str = ", ".join(
            (
                f"GPU {gpu_id}: ({len(jobs)} job{'s' if len(jobs) > 1 else ''} - {', '.join(jobs)})"
                if jobs != ["free"]
                else f"GPU {gpu_id}: (free)"
            )
            for gpu_id, jobs in status.items()
        )
        _logger.info(f"GPU Status: [{status_str}]")


def parse_folder_name(folder_name):
    date_part, time_part = folder_name.split("/")[-1].split("-")[:2]
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")


def training_task(gpu_id, task_name, command):
    _logger.info(f"Task {task_name} starting on GPU {gpu_id}...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            command,
            env=env,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Task {task_name} failed") from e

    _logger.info(f"Task {task_name} finished on GPU {gpu_id} successfully!")


def main():
    args = parser.parse_args()
    utils.setup_default_logging()

    if args.result_dir is not None and args.checkpoint_file is not None:
        checkpoints = []
        for i in range(args.k):
            current_fold_folders = glob.glob(
                os.path.join(args.result_dir, f"*fold{i}*")
            )
            sorted_folders = sorted(
                current_fold_folders, key=parse_folder_name, reverse=True
            )
            checkpoints.append(os.path.join(sorted_folders[0], args.checkpoint_file))
        _logger.info(f"Checkpoints: {checkpoints}")
    else:
        checkpoints = None
        _logger.info("Result directory or checkpoint file not provided")

    tasks = OrderedDict()
    if args.subtyping_linprob_config:
        _logger.info(
            f"Subtyping linear probing configuration file at: {args.subtyping_linprob_config}"
        )
        for i in range(args.k):
            tasks[f"subtyping_linprob_fold{i}"] = [
                args.subtyping_launch_script,
                str(1),
                str(1),
                "c10d",
                "localhost:0",
                args.subtyping_linprob_config,
                str(i),
            ]
            if checkpoints is not None:
                tasks[f"subtyping_linprob_fold{i}"].append(checkpoints[i])
    else:
        _logger.info("Subtyping linear probing configuration file not provided")

    if args.subtyping_finetune_config:
        _logger.info(
            f"Subtyping finetuning configuration file at: {args.subtyping_finetune_config}"
        )
        for i in range(args.k):
            tasks[f"subtyping_finetune_fold{i}"] = [
                args.subtyping_launch_script,
                str(1),
                str(1),
                "c10d",
                "localhost:0",
                args.subtyping_finetune_config,
                str(i),
            ]
            if checkpoints is not None:
                tasks[f"subtyping_finetune_fold{i}"].append(checkpoints[i])
    else:
        _logger.info("Subtyping finetuning configuration file not provided")

    for shot in [1, 5, 10]:
        if getattr(args, f"subtyping_{shot}shot_config"):
            _logger.info(
                f"Subtyping {shot}-shot configuration file at: {getattr(args, f'subtyping_{shot}shot_config')}"
            )
            for i in range(args.k):
                tasks[f"subtyping_{shot}shot_fold{i}"] = [
                    args.subtyping_launch_script,
                    str(1),
                    str(1),
                    "c10d",
                    "localhost:0",
                    getattr(args, f"subtyping_{shot}shot_config"),
                    str(i),
                ]
                if checkpoints is not None:
                    tasks[f"subtyping_{shot}shot_fold{i}"].append(checkpoints[i])
        else:
            _logger.info(f"Subtyping {shot}-shot configuration file not provided")

    if args.survival_linprob_config:
        _logger.info(
            f"Survival linear probing configuration file at: {args.survival_linprob_config}"
        )
        for i in range(args.k):
            tasks[f"survival_linprob_fold{i}"] = [
                args.survival_launch_script,
                str(1),
                str(1),
                "c10d",
                "localhost:0",
                args.survival_linprob_config,
                str(i),
            ]
            if checkpoints is not None:
                tasks[f"survival_linprob_fold{i}"].append(checkpoints[i])
    else:
        _logger.info("Survival linear probing configuration file not provided")

    if args.survival_finetune_config:
        _logger.info(
            f"Survival finetuning configuration file at: {args.survival_finetune_config}"
        )
        for i in range(args.k):
            tasks[f"survival_finetune_fold{i}"] = [
                args.survival_launch_script,
                str(1),
                str(1),
                "c10d",
                "localhost:0",
                args.survival_finetune_config,
                str(i),
            ]
            if checkpoints is not None:
                tasks[f"survival_finetune_fold{i}"].append(checkpoints[i])
    else:
        _logger.info("Survival finetuning configuration file not provided")

    for shot in [1, 5, 10]:
        if getattr(args, f"survival_{shot}shot_config"):
            _logger.info(
                f"Survival {shot}-shot configuration file at: {getattr(args, f'survival_{shot}shot_config')}"
            )
            for i in range(args.k):
                tasks[f"survival_{shot}shot_fold{i}"] = [
                    args.survival_launch_script,
                    str(1),
                    str(1),
                    "c10d",
                    "localhost:0",
                    getattr(args, f"survival_{shot}shot_config"),
                    str(i),
                ]
                if checkpoints is not None:
                    tasks[f"survival_{shot}shot_fold{i}"].append(checkpoints[i])
        else:
            _logger.info(f"Survival {shot}-shot configuration file not provided")

    _logger.info(f"Total tasks collected: {len(tasks)}")

    # Calculate total number of virtual GPUs
    total_virtual_gpus = args.gpu_count * args.virtual_gpu_factor
    gpu_resource_manager = GPUResourceManager(args.gpu_count, args.virtual_gpu_factor)

    if len(tasks) > 0:
        with ThreadPoolExecutor(max_workers=total_virtual_gpus) as executor:
            futures = []
            for task_name, command in tasks.items():
                gpu_id = gpu_resource_manager.acquire_gpu(task_name)
                future = executor.submit(training_task, gpu_id, task_name, command)
                future.add_done_callback(
                    lambda fut, gpu=gpu_id, job=task_name: gpu_resource_manager.release_gpu(  # type: ignore[misc]
                        gpu, job
                    )
                )
                futures.append(future)

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    _logger.error(f"Task failed with error: {e}")

    _logger.info("All jobs completed.")


if __name__ == "__main__":
    main()
