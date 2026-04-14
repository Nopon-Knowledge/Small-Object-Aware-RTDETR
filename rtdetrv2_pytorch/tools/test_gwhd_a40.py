"""Convenience evaluator for GWHD2021 on single A40.
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
try:
    from project_paths import PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import PROJECT_ROOT


EVAL_PRESETS = {
    "baseline": {
        "description": "Evaluate the RT-DETRv2-S GWHD starter run.",
        "config": "configs/rtdetrv2/rtdetrv2_s_a40_gwhd2021.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_s_100e_gwhd_a40_bs24" / "last.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_s_100e_gwhd_a40_bs24" / "eval_last"),
    },
    "high_metric": {
        "description": "Evaluate the RT-DETRv2-R50 high-metric GWHD run.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_a40_high_metric.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_a40_high_metric" / "last.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_a40_high_metric" / "eval_last"),
    },
    "wheat_tsecaf_r18": {
        "description": "Evaluate the RT-DETRv2-R18 Wheat-TS-ECAF GWHD run.",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_gwhd_wheat_tsecaf.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r18vd_120e_gwhd_wheat_tsecaf" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_r18vd_120e_gwhd_wheat_tsecaf" / "eval_best"),
    },
    "wheat_tsecaf_r34": {
        "description": "Evaluate the RT-DETRv2-R34 Wheat-TS-ECAF GWHD run.",
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_180e_gwhd_wheat_tsecaf.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r34vd_180e_gwhd_wheat_tsecaf" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_r34vd_180e_gwhd_wheat_tsecaf" / "eval_best"),
    },
    "wheat_tsecaf_r50": {
        "description": "Evaluate the RT-DETRv2-R50 Wheat-TS-ECAF GWHD run.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_wheat_tsecaf.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_wheat_tsecaf" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_wheat_tsecaf" / "eval_best"),
    },
    "wheat_tsecaf_r101": {
        "description": "Evaluate the RT-DETRv2-R101 Wheat-TS-ECAF GWHD run.",
        "config": "configs/rtdetrv2/rtdetrv2_r101vd_180e_gwhd_wheat_tsecaf.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r101vd_180e_gwhd_wheat_tsecaf" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_r101vd_180e_gwhd_wheat_tsecaf" / "eval_best"),
    },
    "wheat_ears_tsecaf_r18": {
        "description": "Evaluate the RT-DETRv2-R18 Wheat-TS-ECAF Wheat Ears run.",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_wheat_ears_wheat_tsecaf.yml",
        "resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r18vd_120e_wheat_ears_wheat_tsecaf" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "output" / "rtdetrv2_r18vd_120e_wheat_ears_wheat_tsecaf" / "eval_best_maxdets300"),
    },
}


def configure_torch_runtime() -> None:
    # Match the training launcher: some cluster CUDA setups expose GPUs but
    # fail cuDNN engine selection on the first conv unless cuDNN is disabled.
    if os.environ.get("RTDETR_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None}
    )

    cfg = YAMLConfig(args.config, **update_dict)
    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.val()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RT-DETRv2 GWHD2021 checkpoints with preset defaults."
    )

    parser.add_argument("--preset", choices=sorted(EVAL_PRESETS), default="wheat_tsecaf_r18")
    parser.add_argument("-c", "--config", type=str, help="Config path. Defaults to the selected preset.")
    parser.add_argument("-r", "--resume", type=str, help="checkpoint for evaluation")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    preset = EVAL_PRESETS[args.preset]
    if args.config is None:
        args.config = preset["config"]
    if args.resume is None:
        args.resume = preset["resume"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]

    print(f"Selected preset: {args.preset} - {preset['description']}")
    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {args.resume}")
    print(f"Evaluation outputs: {args.output_dir}")

    wants_cuda = args.device is None or str(args.device).lower() != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible in the current session. "
            "This A40 evaluator should be run inside a GPU allocation/job. "
            "If you intentionally want CPU evaluation, pass '-d cpu'."
        )

    configure_torch_runtime()
    main(args)
