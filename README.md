# tam_object_detection

## Use [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)

### For Sigverse

```bash
roslaunch tam_object_detection hsr_head_rgbd_lang_sam_service.launch
```

### For physical robot

- [langsam_detection_service.pyのトピック名](https://github.com/tamhome/tam_object_detection/blob/b68f77145cbcfbd0cb0b5cf6cb33c4e2c6857580/tam_object_detection/script/lang_sam_detection_service.py#L73-L74)を変更してから実行してください．
- その他のパラメタは[hsr_head_rgbd_lang_sam_service.launch](./tam_object_detection/launch/hsr_head_rgbd_lang_sam_service.launch)に記載があります．

## Use YOLOv8

- 【Note】SigverseではYOLOを使用しておらず，動作確認をしておりません．

## Trained model
https://drive.google.com/drive/folders/14dEhn8K4RCnkLzO9gL6ehuJ4zIIoEwJ4?usp=share_link

## YOLOv8 のモデル変換

```
hoge.pt -> hoge.onnx -> hoge.engine
```

必ず Singularity 内で行うこと．--nv フラグを忘れないこと．

### 0. ディレクトリ移動

```shell
cd ./third_party/YOLOv8-TensorRT
```

### 1. hoge.pt -> hoge.onnx

- Object Detection の場合

```shell
python export.py --weights <path to checkpoint>.pt --opset 11 --sim --input-shape 1 3 <image_size> <image_size> --device cuda:0
```

- Instance Segmentation の場合

```shell
python export_seg.py --weights <path to checkpoint>.pt --opset 11 --sim --input-shape 1 3 <image_size> <image_size> --device cuda:0
```

### 2. hoge.onnx -> hoge.engine

- Object Detection の場合

```shell
python build.py --weights <path to checkpoint>.onnx --fp16 --device cuda:0
```

- Instance Segmentation の場合

```shell
python build.py --weights <path to checkpoint>.onnx --fp16 --device cuda:0 --seg
```

# LICENCE
This repository is based on YOLOv8 and the license is based on AGPL 3.0 created by [ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
