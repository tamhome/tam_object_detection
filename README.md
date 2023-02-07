# tam_object_detection

## YOLOv8 のモデル変換

```
hoge.pt -> hoge.onnx -> hoge.engine
```

必ず Singularity 内で行うこと

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
