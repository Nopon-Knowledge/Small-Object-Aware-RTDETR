# Small-Object-Aware RT-DETRv2 for Robust Wheat Head Detection in Dense Field Images

This repository contains the PyTorch research code used for the manuscript
**Small-Object-Aware RT-DETRv2 for Robust Wheat Head Detection in Dense Field Images**.
The local draft PDF, when included, is available at [paper.pdf](paper.pdf).

The code is a reduced RT-DETRv2 release focused on wheat-head detection. It keeps
the model, configuration files, data conversion helpers, training/evaluation
launchers, visualization tools, ONNX export, and profiling utilities needed to
reproduce the RT-DETRv2 baseline, proposed method, transfer experiments, and
ablation experiments reported in the paper.

## Scope of This Release

Included:

- RT-DETRv2 training and evaluation code for COCO-style wheat-head detection.
- GWHD2021, Wheat Ears, and Global Wheat CodaLab dataset configuration files.
- The proposed modules:
  - `WheatTSECAFFusion` and `WheatDetailEnhancer` in
    `small_object_aware_rtdetr/src/zoo/rtdetr/hybrid_encoder.py`.
  - `agnostic_small` query selection, `anchor_grid_size`, and
    `query_size_prior` in
    `small_object_aware_rtdetr/src/zoo/rtdetr/rtdetrv2_decoder.py`.
- R18, R34, and R50 RT-DETRv2 baseline and proposed-model configs.
- Direct-transfer configs for Wheat Ears and Global Wheat CodaLab.
- R50 ablation configs for feature fusion, detail enhancement, and query
  selection.
- Tools for training, evaluation, dataset conversion, prediction visualization,
  feature-map visualization, SVG plotting, ONNX export, and profiling.

Not included:

- Datasets, trained checkpoints, generated figures, logs, exported models, or
  local experiment outputs.
- External baseline training code for Faster R-CNN, RetinaNet, YOLOv8, or
  YOLO11. Those baselines were trained with public toolchains and are documented
  below for reproducibility, but their frameworks are not vendored into this
  reduced RT-DETRv2 code release.
- The original Paddle implementation, RT-DETR v1 code, generic COCO/VOC
  examples, classification models, TensorRT benchmark wrappers, and IDE files.

## Paper-to-Code Map

| Paper item | Code location |
|---|---|
| RT-DETRv2 baseline | `configs/rtdetrv2/rtdetrv2_*_gwhd_baseline.yml` |
| Proposed R18/R34/R50 models | `configs/rtdetrv2/small_object_aware_rtdetr_*_gwhd.yml` |
| Wheat-TS-ECAF fusion | `src/zoo/rtdetr/hybrid_encoder.py` |
| Low-level detail enhancement | `src/zoo/rtdetr/hybrid_encoder.py` |
| Small-object-aware query selection | `src/zoo/rtdetr/rtdetrv2_decoder.py` |
| Anchor prior and query size prior | `RTDETRTransformerv2.anchor_grid_size`, `query_size_prior` |
| Source-domain training presets | `tools/train_gwhd_a40.py` |
| Source-domain and transfer evaluation presets | `tools/test_gwhd_a40.py` |
| Dataset conversion | `tools/prepare_wheat_datasets.py` |
| Profiling Params/GFLOPs | `tools/run_profile.py` |
| Qualitative figures | `tools/visualize_gwhd_predictions.py`, `tools/visualize_feature_maps.py` |

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

For CUDA training, install the `torch` and `torchvision` wheels matching your
CUDA runtime before installing the remaining dependencies if your platform needs
a custom PyTorch package index.

The paper experiments were run on a single NVIDIA A40 GPU. The launchers disable
cuDNN by default through `RTDETR_DISABLE_CUDNN=1` because some cluster
environments exposed unstable cuDNN engine selection for the first backbone
convolution. To re-enable cuDNN, set:

```bash
export RTDETR_DISABLE_CUDNN=0
```

## Datasets

All dataset configs expect COCO-style directories under
`small_object_aware_rtdetr/datasets` by default. To keep data outside the
repository, set:

```bash
export WHEAT_DATASETS_ROOT=/path/to/datasets
```

Expected layout:

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

Dataset protocol used in the paper:

- GWHD2021 is the source dataset. The project split contains 5499 training
  images and 1016 validation images.
- Wheat Ears is converted from XML annotations to COCO format and split with an
  80:20 ratio using seed 42.
- Global Wheat CodaLab uses only labeled subsets for quantitative evaluation;
  each labeled subset is split with an 80:20 ratio. Target-domain training
  partitions are created for layout consistency only and are not used for
  fine-tuning in the direct-transfer experiments.

Convert the external datasets:

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

## Training RT-DETRv2 Models

Run commands from `small_object_aware_rtdetr/`.

Train the proposed R50 model:

```bash
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r50
```

Train the paired RT-DETRv2 baselines:

```bash
python tools/train_gwhd_a40.py --preset baseline_r18
python tools/train_gwhd_a40.py --preset baseline_r34
python tools/train_gwhd_a40.py --preset baseline_r50
```

Train all proposed model scales:

```bash
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r18
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r34
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r50
```

Train the R50 ablation variants:

```bash
python tools/train_gwhd_a40.py --preset ablation_wheat_fusion_r50
python tools/train_gwhd_a40.py --preset ablation_detail_enhance_r50
python tools/train_gwhd_a40.py --preset ablation_agnostic_small_r50
python tools/train_gwhd_a40.py --preset ablation_full_small_object_aware_r50
```

Outputs are written under `output/`, which is ignored by Git.

## Three-Seed Protocol

The manuscript reports source-domain accuracy as mean plus/minus standard
deviation over three independent runs. The launcher supports explicit seeds
through `--seed`. A simple reproduction protocol is:

```bash
for seed in 0 1 2; do
  python tools/train_gwhd_a40.py \
    --preset small_object_aware_rtdetr_r50 \
    --seed ${seed} \
    --output-dir output/seed_${seed}/small_object_aware_rtdetr_r50vd_180e_gwhd
done
```

Use the same seed list for the paired baseline and ablation runs when computing
mean and standard deviation. Exact bit-level reproducibility can still vary with
GPU driver, CUDA, PyTorch, and distributed runtime versions.

## Evaluation

Evaluate on GWHD2021:

```bash
python tools/test_gwhd_a40.py --preset small_object_aware_rtdetr_r50
python tools/test_gwhd_a40.py --preset baseline_r50
```

Evaluate direct transfer to external wheat datasets without target-domain
fine-tuning:

```bash
python tools/test_gwhd_a40.py --preset wheat_ears_small_object_aware_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_small_object_aware_r50
python tools/test_gwhd_a40.py --preset wheat_ears_baseline_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_baseline_r50
```

For a seed-specific checkpoint, pass `-r` and `--output-dir` explicitly:

```bash
python tools/test_gwhd_a40.py \
  --preset small_object_aware_rtdetr_r50 \
  -r output/seed_0/small_object_aware_rtdetr_r50vd_180e_gwhd/best.pth \
  --output-dir output/eval_seed_0/small_object_aware_rtdetr_r50
```

The evaluator reports COCO-style AP, AP50, AP75, APs, APm, APl, and recall
statistics. The paper uses AP@[0.50:0.95], AP50, AP75, APs, precision, recall,
F1, Params, GFLOPs, FPS, and image-level counting metrics.

## Image-Level Counting Analysis

No separate counting model is trained. Image-level wheat-head counts are obtained
by counting the final confidence-filtered detection boxes for each image.

Counting protocol:

- Ground-truth count: number of annotated wheat-head boxes in the image.
- Predicted count: number of final predicted boxes after confidence filtering.
- Threshold: the validation confidence threshold that maximizes F1 for each
  detector.
- Metrics: MAE, RMSE, MAPE, and R2.
- Density analysis: validation images are grouped by the number of annotated
  heads per image.

The reduced release provides detector training and evaluation outputs from which
these counts can be derived. If submitting supplementary material, include the
per-image prediction files or the derived counting CSVs together with the paper.

## External Baselines in the Paper

The main paper compares the proposed RT-DETRv2 variants against external
detectors. Their source code is not vendored here, but the comparison protocol is
summarized for reproducibility.

| Baseline | Toolchain | Main setting |
|---|---|---|
| Faster R-CNN-R50-FPN | `torchvision` detection | Same GWHD2021 split, 1024 input scale, COCO evaluator |
| RetinaNet-R50 | `torchvision` detection | Same GWHD2021 split, 1024 input scale, COCO evaluator |
| YOLOv8s, YOLOv8m | Ultralytics YOLO | Same GWHD2021 split, 1024 input scale |
| YOLO11s, YOLO11m | Ultralytics YOLO11 | Same GWHD2021 split, 1024 input scale |

External baseline fairness rules used in the manuscript:

- All source-domain models use the same GWHD2021 train/validation split.
- All source-domain models are evaluated at 1024 x 1024 input resolution.
- COCO-style detection metrics are used for all models.
- Precision, recall, F1, and counting metrics use each model's F1-maximizing
  validation threshold.
- Params, GFLOPs, and FPS are measured on the same NVIDIA A40 GPU with batch
  size 1.

## Figures, Visualization, and Profiling

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

Profile model parameters and FLOPs:

```bash
python tools/run_profile.py \
  -c configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml
```

Export a trained checkpoint to ONNX:

```bash
python tools/export_onnx.py \
  -c configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml \
  -r output/small_object_aware_rtdetr_r50vd_180e_gwhd/best.pth \
  -o small_object_aware_rtdetr_r50.onnx \
  -s 1024 \
  --check
```

## Reproducibility Notes

- This repository is intended to reproduce the RT-DETRv2 part of the manuscript:
  paired baselines, proposed models, transfer evaluation, and ablations.
- Public datasets must be downloaded separately from their original sources.
- Official RT-DETRv2 COCO pretrained checkpoints must be downloaded separately.
- Trained model weights are not stored in Git because of file size.
- Generated outputs are ignored by Git. Keep logs, evaluation JSON files, and
  counting CSVs as supplementary artifacts when preparing a paper release.
- If exact table reproduction is required, archive the trained checkpoints and
  the per-seed evaluation outputs together with the manuscript version.

## Git Hygiene

The repository ignores local datasets, checkpoints, exported models, logs,
generated figures, Python caches, and IDE metadata. Before publishing a release,
check the staging area with:

```bash
git status --short
git ls-files -i -c --exclude-standard
```

## License

This code is released under the license provided in [LICENSE](LICENSE).

## Citation

If you use this repository, please cite the associated paper after publication
and cite RT-DETRv2 and the public wheat datasets used in your experiments.
