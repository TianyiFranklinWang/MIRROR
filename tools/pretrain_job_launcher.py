import argparse
import logging
import os
import subprocess
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

import timm.utils


_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Bulk launcher for pretrain jobs")
parser.add_argument("--gpu-count", type=int, default=1, help="Number of physical GPUs")
parser.add_argument(
    "--virtual-gpu-factor",
    type=int,
    default=1,
    help="Number of virtual GPUs per physical GPU (to run multiple jobs concurrently on the same GPU)",
)
parser.add_argument("-k", "--k", type=int, default=5, help="The number of folds in CV")
parser.add_argument(
    "--pretrain-launch-script",
    type=str,
    default="./tools/run_train_mirror.py",
    help="Pretraining launch script",
)
parser.add_argument(
    "--pretrain-config",
    type=str,
    required=True,
    default=None,
    help="Pretraining configuration file",
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


def training_job(gpu_id, job_name, command):
    _logger.info(f"Job '{job_name}' starting on GPU {gpu_id}...")
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
        raise RuntimeError(f"Job '{job_name}' failed") from e

    _logger.info(f"Job '{job_name}' finished on GPU {gpu_id} successfully!")


def main(args):
    timm.utils.setup_default_logging()

    jobs = OrderedDict()
    if args.pretrain_config:
        _logger.info(f"Pretraining configuration file: {args.pretrain_config}")
        for i in range(args.k):
            jobs[f"pretrain_fold{i}"] = [
                args.pretrain_launch_script,
                str(1),
                str(1),
                "c10d",
                "localhost:0",
                args.pretrain_config,
                str(i),
            ]
    else:
        raise ValueError("No configuration file provided")

    _logger.info(f"Total jobs to be launched: {len(jobs)}")

    total_virtual_gpus = args.gpu_count * args.virtual_gpu_factor
    gpu_resource_manager = GPUResourceManager(args.gpu_count, args.virtual_gpu_factor)

    if len(jobs) > 0:
        with ThreadPoolExecutor(max_workers=total_virtual_gpus) as executor:
            futures = []
            for job_name, command in jobs.items():
                gpu_id = gpu_resource_manager.acquire_gpu(job_name)
                future = executor.submit(training_job, gpu_id, job_name, command)
                future.add_done_callback(
                    lambda fut, gpu=gpu_id, job=job_name: gpu_resource_manager.release_gpu(  # type: ignore[misc]
                        gpu, job
                    )
                )
                futures.append(future)

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    _logger.error(f"Job failed with error: {e}")

    _logger.info("All jobs completed.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
