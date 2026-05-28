from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=416,
    batch=4,
    device="cpu",
    workers=0
)

metrics = model.val()

print(metrics)