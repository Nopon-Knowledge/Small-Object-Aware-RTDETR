# Tools

Run these commands from `small_object_aware_rtdetr/`.

## Training

```bash
python tools/train_gwhd_a40.py --preset baseline_r50
python tools/train_gwhd_a40.py --preset small_object_aware_rtdetr_r50
```

## Evaluation

```bash
python tools/test_gwhd_a40.py --preset small_object_aware_rtdetr_r50
python tools/test_gwhd_a40.py --preset wheat_ears_small_object_aware_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_small_object_aware_r50
```

## Ablations

```bash
python tools/train_gwhd_a40.py --preset ablation_wheat_fusion_r50
python tools/train_gwhd_a40.py --preset ablation_detail_enhance_r50
python tools/train_gwhd_a40.py --preset ablation_agnostic_small_r50
python tools/train_gwhd_a40.py --preset ablation_full_small_object_aware_r50
```

## Figures

```bash
python tools/visualize_gwhd_predictions.py --preset small_object_aware_rtdetr_r50_val_best
python tools/visualize_feature_maps.py --preset gwhd_r50_baseline_vs_small_object_aware
python tools/plot_training_curves_svg.py
python tools/plot_ablation_svg.py
python tools/plot_generalization_svg.py
python tools/plot_pr_curves_svg.py
```

## Utilities

```bash
python tools/prepare_wheat_datasets.py --preset wheat_ears
python tools/prepare_wheat_datasets.py --preset global_wheat_codalab
python tools/run_profile.py -c configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml
python tools/export_onnx.py -c configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml -r output/small_object_aware_rtdetr_r50vd_180e_gwhd/best.pth -o small_object_aware_rtdetr_r50.onnx -s 1024 --check
```
