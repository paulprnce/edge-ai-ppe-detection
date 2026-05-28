from ultralytics import YOLO

model = YOLO("weights/ppe_yolov8n_fp32.pt")

model.export(
    format="onnx",
    half=True
)

print("ONNX FP16 model exported successfully!")