import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from CW_lesson10 import mean_color
from SamostiynaLesson11 import contours


def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    if shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "black": (0, 0, 0)
}


X = []
y = []

for color_name, bgr in colors.items():
    for _ in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,(20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1200:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]

            mean_color = cv2.mean(roi)[:3]
            mean_color - np.array(mean_color).reshape(1, -1)

            label = model.predict(mean_color)[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label.upper(), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)

    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()