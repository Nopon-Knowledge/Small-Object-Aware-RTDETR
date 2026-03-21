

Train/test script examples
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only` 


Tuning script examples
- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth` 


Export script examples
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


GPU do not release memory
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


Save all logs
- Appending `&> train.log 2>&1 &` or `&> train.log 2>&1`


GWHD 2021 preprocessing (CSV -> COCO)
- `python tools/prepare_gwhd2021.py --source-root /path/to/gwhd_2021 --output-root ./dataset/gwhd_2021_coco --splits train val test --link-mode symlink --normalize-format --keep-empty-images --clean-output`
- Generated layout:
```
dataset/gwhd_2021_coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json
  train2017/
  val2017/
  test2017/
  preprocess_report.json
```
