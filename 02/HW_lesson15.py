import os
import cv2
import shutil

PROJECT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "out")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    print("Помилка: Не знайдено файли моделі у папці 'models'.")
    exit()

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = CLASSES.index("person")
CONF_THRESHOLD = 0.5


def detect_people_on_image(image_bgr):
    (h, w) = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    detections = net.forward()

    found_people = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence >= CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            found_people.append((x1, y1, x2, y2, confidence))

    return found_people



allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")

if os.path.exists(IMAGES_DIR):
    files = os.listdir(IMAGES_DIR)
else:
    print(f"Папка {IMAGES_DIR} не існує. Створіть її та додайте фото.")
    files = []

count_images_with_people = 0
count_images_no_people = 0

print(f"Починаємо обробку зображень з {IMAGES_DIR}...\n")

for filename in files:
    if not filename.lower().endswith(allowed_ext):
        continue

    in_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(in_path)

    if img is None:
        print("Не вдалося прочитати:", filename)
        continue

    people_boxes = detect_people_on_image(img)
    people_count = len(people_boxes)

    if people_count > 0:
        out_path = os.path.join(PEOPLE_DIR, filename)
        shutil.copy2(in_path, out_path)
        count_images_with_people += 1

        boxed = img.copy()

        for (x1, y1, x2, y2, conf) in people_boxes:
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{conf:.2f}"
            cv2.putText(boxed, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        info_text = f"People count: {people_count}"
        cv2.putText(boxed, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        boxed_path = os.path.join(PEOPLE_DIR, "boxed_" + filename)
        cv2.imwrite(boxed_path, boxed)

        print(f"[PEOPLE] {filename} -> знайдено {people_count}")

    else:
        # Якщо N = 0 -> в out/no_people/ (Task 3)
        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        shutil.copy2(in_path, out_path)
        count_images_no_people += 1

        print(f"[NO]     {filename} -> людей не знайдено")

print("\nГотово!")
print(f"Фото з людьми: {count_images_with_people}")
print(f"Фото без людей: {count_images_no_people}")
print(f"Результати збережено в: {OUT_DIR}")