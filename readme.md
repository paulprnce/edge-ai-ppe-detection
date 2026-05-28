# Edge AI PPE Detection using YOLOv8n and ONNX FP16
This project implements a real-time industrial PPE (Personal Protective Equipment) detection system optimized for edge deployment.

The system detects:
- Helmet
- No Helmet
- Vest
- No Vest
- Person

The objective is to simulate a lightweight industrial safety monitoring solution capable of running on edge devices with reduced latency and improved inference speed.

Model: YOLOv8n
Framework: Ultralytics YOLO
Quantization: FP16
Export Format: ONNX

Dataset Source: https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset

Classes:
- helmet
- no_helmet
- vest
- no_vest
- person

epochs=5
imgsz=416
batch=4

## Training

python train.py

## Export

python export_model.py

## Live Inference

python live_inference.py



## Performance Benchmark

| Metric            | FP32 YOLOv8n (.pt) | ONNX FP16 (.onnx) |
| ----------------- | ------------------ | ----------------- |
| Model Size        | 5.92 MB            | 5.83 MB           |
| mAP50-95          | 0.477              | ~0.47             |
| FPS               | ~30 FPS            | ~50 FPS           |
| Inference Latency | ~26 ms             | ~20 ms            |

### Observations

* The ONNX FP16 model achieved improved inference speed and lower latency compared to the FP32 baseline model.
* The YOLOv8n baseline model was already lightweight, so FP16 quantization resulted in only a modest reduction in model size.
* A small reduction in accuracy was acceptable in exchange for significantly improved real-time edge inference performance.


## Model Weights

### FP32 Baseline Model

https://drive.google.com/file/d/1oc3i_tC7ylq2rN97HOXqCFcqfuclrxcW/view?usp=sharing

### ONNX FP16 Edge Model

https://drive.google.com/file/d/19IcQRsge9onw534MvKNk4AeCdno04QkJ/view?usp=sharing
