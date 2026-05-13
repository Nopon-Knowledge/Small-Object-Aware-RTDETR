# Implementation Directory

This directory contains the retained PyTorch implementation for
Small-Object-Aware RT-DETR wheat-head detection.

Use the repository-level README for the full installation, dataset, training,
evaluation, visualization, export, and profiling instructions.

Key entry points:

- `configs/rtdetrv2/`: baseline, proposed, ablation, and direct-transfer configs.
- `src/zoo/rtdetr/hybrid_encoder.py`: Wheat-TS-ECAF fusion and detail enhancement.
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: small-object-aware query selection.
- `tools/train_gwhd_a40.py`: training presets.
- `tools/test_gwhd_a40.py`: GWHD2021 and cross-dataset evaluation presets.
- `tools/prepare_wheat_datasets.py`: dataset conversion helpers.

Run commands from this directory:

```bash
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r50
python tools/test_gwhd_a40.py --preset wheat_ears_small_object_aware_r50
```
