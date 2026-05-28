import cv2
import time
from ultralytics import YOLO

model = YOLO("weights/ppe_yolov8n_fp32.onnx")


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.time()

    
    pre_start = time.time()

    resized = cv2.resize(frame, (416, 416))

    pre_end = time.time()

    preprocess_time = (pre_end - pre_start) * 1000


    inf_start = time.time()

    results = model.predict(
        source=resized,
        imgsz=416,
        conf=0.5,
        verbose=False
    )

    inf_end = time.time()

    inference_time = (inf_end - inf_start) * 1000

    post_start = time.time()

    annotated_frame = results[0].plot()

    post_end = time.time()

    postprocess_time = (post_end - post_start) * 1000

    total_time = time.time() - start_time
    fps = 1 / total_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Preprocess: {preprocess_time:.2f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time:.2f} ms",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Postprocess: {postprocess_time:.2f} ms",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.imshow("Edge PPE Detection", annotated_frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()