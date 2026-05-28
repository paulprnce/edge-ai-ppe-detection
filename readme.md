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