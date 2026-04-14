"""Resume RT-DETRv2-R18 GWHD training from the saved last checkpoint.
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


DEFAULT_CONFIG = "configs/rtdetrv2/rtdetrv2_r18vd_180e_gwhd_wheat_tsecaf_resume.yml"
DEFAULT_RESUME = str(
    PROJECT_ROOT / "output" / "rtdetrv2_r18vd_120e_gwhd_wheat_tsecaf" / "last.pth"
)


def configure_torch_runtime() -> None:
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
    solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resume RT-DETRv2-R18 GWHD training from last.pth to 180 epochs."
    )

    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("-r", "--resume", type=str, default=DEFAULT_RESUME, help="checkpoint to resume from")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    wants_cuda = args.device is None or str(args.device).lower() != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible in the current session. "
            "This A40 resume launcher should be run inside a GPU allocation/job. "
            "If you intentionally want CPU training, pass '-d cpu'."
        )

    configure_torch_runtime()
    main(args)
