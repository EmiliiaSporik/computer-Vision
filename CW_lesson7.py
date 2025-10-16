import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])


points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            object_found = True
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                points.append((cx, cy))


    if object_found:
        cv2.putText(frame, "Object detected", (390, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (5, 5), (635, 475), (0, 255, 0), 2)
    else:
        points = []
        cv2.putText(frame, "Object not detected", (360, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (5, 5), (635, 475), (0, 0, 255), 2)
    # for i in range(1, len(points)):
    #     if points[i - 1][1] is None or points[i][1] is None:
    #         continue
    #     cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()