"""Convenience launcher for GWHD2021 on single A40.
"""

import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


TRAINING_PRESETS = {
    "baseline": {
        "description": "RT-DETRv2-S GWHD starter profile.",
        "config": "configs/rtdetrv2/rtdetrv2_s_a40_gwhd2021.yml",
        "tuning_candidates": [
            Path("/storage/home/402005/python_project/RT-DETR-main/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
            Path("/storage/home/402005/.cache/torch/hub/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
        ],
    },
    "high_metric": {
        "description": "RT-DETRv2-R50 high-metric GWHD profile with larger input size and more queries.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_a40_high_metric.yml",
        "tuning_candidates": [
            Path("/storage/home/402005/python_project/RT-DETR-main/rtdetrv2_r50vd_6x_coco_ema.pth"),
            Path("/storage/home/402005/.cache/torch/hub/checkpoints/rtdetrv2_r50vd_6x_coco_ema.pth"),
        ],
    },
}


def configure_torch_runtime() -> None:
    # Some cluster CUDA environments expose GPUs but fail on cuDNN engine
    # selection for the first backbone conv. Default to the more conservative
    # native CUDA kernels for this custom training launcher.
    if os.environ.get("RTDETR_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint(candidates) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, Path):
            if candidate.exists():
                return str(candidate)
            continue
        candidate_path = Path(str(candidate))
        if candidate_path.exists():
            return str(candidate_path)
    return None


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), (
        "Only support from_scrach or resume or tuning at one time"
    )

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None}
    )

    cfg = YAMLConfig(args.config, **update_dict)
    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RT-DETRv2 on GWHD2021 with preset hyperparameters."
    )

    parser.add_argument("--preset", choices=sorted(TRAINING_PRESETS), default="high_metric")
    parser.add_argument("-c", "--config", type=str, help="Config path. Defaults to the selected preset.")
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")
    parser.add_argument("--test-only", action="store_true", default=False)

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    preset = TRAINING_PRESETS[args.preset]
    if args.config is None:
        args.config = preset["config"]

    if args.resume is None and args.tuning is None:
        args.tuning = resolve_checkpoint(preset["tuning_candidates"])

    print(f"Selected preset: {args.preset} - {preset['description']}")
    print(f"Using config: {args.config}")
    if args.tuning:
        print(f"Using tuning checkpoint: {args.tuning}")
    else:
        print("No local tuning checkpoint found, training will start without external pretrained weights.")

    wants_cuda = args.device is None or str(args.device).lower() != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible in the current session. "
            "This A40 launcher should be run inside a GPU allocation/job. "
            "If you intentionally want CPU training, pass '-d cpu'."
        )

    configure_torch_runtime()
    main(args)
