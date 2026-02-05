import cv2
import os
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

VIDEO_FILENAME = "video1.mp4"
VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
OUTPUT_VIDEO_PATH = os.path.join(OUT_DIR, f"result_{VIDEO_FILENAME}")

CONF_THRESHOLD = 0.3
RESIZE_WIDTH = 1280

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Помилка: Не вдалося відкрити відео {VIDEO_PATH}")
    exit()


fps_input = cap.get(cv2.CAP_PROP_FPS)
video_writer = None

prev_time = time.time()
frame_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        print("Відео завершено")
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))


    results = model(frame, conf=CONF_THRESHOLD, verbose=False)


    current_counts = {name: 0 for name in VEHICLE_CLASSES.values()}

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in VEHICLE_CLASSES:
                class_name = VEHICLE_CLASSES[cls_id]

                current_counts[class_name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label = f'{class_name} {conf:.2f}'

                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = x1 + t_size[0] + 3, y1 + t_size[1] + 4
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)

                cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (230, 160), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps_calc = 1.0 / dt if dt > 0 else 0

    cv2.putText(frame, f'FPS: {fps_calc:.1f}', (20, 35), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

    y_offset = 65
    for name, count in current_counts.items():
        if count > 0:
            text = f"{name}: {count}"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25


    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
