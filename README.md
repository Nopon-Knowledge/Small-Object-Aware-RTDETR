# Small-Object-Aware Adaptation of RT-DETR for Dense Wheat Head Detection in Field Images

This repository contains the PyTorch implementation used for the paper
**Small-Object-Aware Adaptation of RT-DETR for Dense Wheat Head Detection in Field Images**.
It is a reduced research-code release derived from RT-DETRv2 and keeps only the
components required to reproduce the wheat-head detection experiments.

## What Is Included

- RT-DETRv2 training and evaluation code for COCO-style object detection.
- Wheat-head detection configs for GWHD2021, Wheat Ears, and Global Wheat CodaLab.
- The proposed small-object-aware changes:
  - `WheatTSECAFFusion` and low-level detail enhancement in `small_object_aware_rtdetr/src/zoo/rtdetr/hybrid_encoder.py`.
  - `agnostic_small` query selection, `anchor_grid_size`, and `query_size_prior` in `small_object_aware_rtdetr/src/zoo/rtdetr/rtdetrv2_decoder.py`.
- Training, direct-transfer evaluation, ablation, visualization, SVG plotting, ONNX export, and profiling tools.

The original Paddle implementation, RT-DETR v1 code, generic COCO/VOC examples,
classification models, TensorRT benchmark wrappers, IDE metadata, generated outputs,
datasets, and model weights are intentionally excluded.

## Repository Layout

```text
small_object_aware_rtdetr/
  configs/
    dataset/                  # COCO-style wheat dataset definitions
    rtdetrv2/                 # baseline, proposed, ablation, and transfer configs
    runtime.yml
  src/
    core/                     # YAML configuration and object registry
    data/                     # detection datasets, dataloaders, and transforms
    misc/                     # logging, distributed helpers, profiling helpers
    nn/backbone/              # PResNet backbone
    optim/                    # optimizer, EMA, AMP, warmup
    solver/                   # train and validation loops
    zoo/rtdetr/               # RT-DETRv2 model and proposed modules
  tools/
    train_gwhd_a40.py         # paper training presets
    test_gwhd_a40.py          # source-domain and transfer evaluation presets
    prepare_wheat_datasets.py # dataset conversion helpers
    visualize_*.py            # prediction and feature-map visualization
    plot_*.py                 # SVG figure generation from logs/eval files
    export_onnx.py
    run_profile.py
```

## Installation

Create a Python environment and install the project dependencies:

```bash
cd small_object_aware_rtdetr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA training, install the `torch` and `torchvision` wheels that match your
CUDA runtime before installing the remaining dependencies if your platform needs
a custom PyTorch package index.

## Datasets

All dataset configs expect COCO-style directories under `small_object_aware_rtdetr/datasets`
by default. The expected layout is:

```text
datasets/
  gwhd_2021/
    train2017/
    val2017/
    annotations/
      instances_train2017.json
      instances_val2017.json
  Wheat_Ears_Detection_Dataset_rtdetr/
    train2017/
    val2017/
    annotations/
      instances_train2017.json
      instances_val2017.json
  global-wheat-codalab-official_rtdetr/
    train2017/
    val2017/
    annotations/
      instances_train2017.json
      instances_val2017.json
```

To keep data outside the repository, set:

```bash
export WHEAT_DATASETS_ROOT=/path/to/datasets
```

The helper script can convert the Wheat Ears and Global Wheat CodaLab source
folders into the COCO-style layouts used by the configs:

```bash
python tools/prepare_wheat_datasets.py --preset wheat_ears
python tools/prepare_wheat_datasets.py --preset global_wheat_codalab
```

## Pretrained Checkpoints

Training presets look for official RT-DETRv2 COCO checkpoints in these locations:

- the repository root,
- `~/.cache/torch/hub/checkpoints`,
- the directory pointed to by `RTDETR_PRETRAINED_DIR`.

Example:

```bash
export RTDETR_PRETRAINED_DIR=/path/to/checkpoints
```

Model weights and training outputs are not included in this repository.

## Training

Run commands from `small_object_aware_rtdetr/`.

Train the proposed R50 model:

```bash
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r50
```

Train the paired baselines:

```bash
python tools/train_gwhd_a40.py --preset baseline_r18
python tools/train_gwhd_a40.py --preset baseline_r34
python tools/train_gwhd_a40.py --preset baseline_r50
```

Train the R50 ablation variants:

```bash
python tools/train_gwhd_a40.py --preset ablation_wheat_fusion_r50
python tools/train_gwhd_a40.py --preset ablation_detail_enhance_r50
python tools/train_gwhd_a40.py --preset ablation_agnostic_small_r50
python tools/train_gwhd_a40.py --preset ablation_full_small_object_aware_r50
```

Outputs are written under `output/`, which is ignored by Git.

## Evaluation

Evaluate on GWHD2021:

```bash
python tools/test_gwhd_a40.py --preset small_object_aware_rtdetr_r50
python tools/test_gwhd_a40.py --preset baseline_r50
```

Evaluate direct transfer to external wheat datasets:

```bash
python tools/test_gwhd_a40.py --preset wheat_ears_small_object_aware_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_small_object_aware_r50
python tools/test_gwhd_a40.py --preset wheat_ears_baseline_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_baseline_r50
```

## Figures and Analysis

Generate SVG figures from saved logs and evaluation dictionaries:

```bash
python tools/plot_training_curves_svg.py
python tools/plot_ablation_svg.py
python tools/plot_generalization_svg.py
python tools/plot_pr_curves_svg.py
```

Generate qualitative prediction or feature-map visualizations:

```bash
python tools/visualize_gwhd_predictions.py --preset small_object_aware_rtdetr_r50_val_best
python tools/visualize_feature_maps.py --preset gwhd_r50_baseline_vs_small_object_aware
```

Generated figures are written under `vis_results/`, which is ignored by Git.

## Export and Profiling

Export a trained checkpoint to ONNX:

```bash
python tools/export_onnx.py \
  -c configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml \
  -r output/small_object_aware_rtdetr_r50vd_180e_gwhd/best.pth \
  -o small_object_aware_rtdetr_r50.onnx \
  -s 1024 \
  --check
```

Profile model parameters and FLOPs:

```bash
python tools/run_profile.py \
  -c configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml
```

## Git Hygiene

The repository ignores local datasets, checkpoints, exported models, logs,
generated figures, Python caches, and IDE metadata. Before publishing a release,
check the staging area with:

```bash
git status --short
git ls-files -i -c --exclude-standard
```

## License

This code is released under the license provided in `LICENSE`.
