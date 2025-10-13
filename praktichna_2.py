import cv2
import numpy as np

img = cv2.imread("image/figure.jpg")

img_copy = img.copy()

h = [0, 225]

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([101, 69, 0])
upper_blue = np.array([151, 255, 255])

lower_green = np.array([41, 92, 0])
upper_green = np.array([66, 255, 255])

lower_red = np.array([0, 54, 0])
upper_red = np.array([5, 255, 255])

lower_yellow = np.array([17, 57, 85])
upper_yellow = np.array([27, 255, 255])

mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_red = cv2.inRange(img, lower_red, upper_red)
mask_green = cv2.inRange(img, lower_green, upper_green)

mask_total = cv2.bitwise_or(mask_yellow, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_red)
mask_total = cv2.bitwise_or(mask_total, mask_green)
img = cv2.bitwise_and(img, img, mask=mask_total)



contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)

        perimetr = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimetr ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True)
        if len(approx) == 9:
            shape = "Frog"
        elif len(approx) >= 8:
            shape = "Oval"
        else:
            shape = "Car"
        if h == [26, 35]:
            color = "yellow"
        elif h == [96, 130]:
            color = "blue"
        elif h == [0, 10]:
            color = "red"
        elif h == [36, 85]:
            color = "green"
        else:
            color = "another"



        cv2.putText(img_copy, f'S: {int(area)}, P: {int(perimetr)}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img_copy, f' X: {x}, Y: {y} ', (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (0, 0, 0), -1)
        cv2.drawContours(img_copy, [cnt], -1, (0, 0, 0), 2)
        cv2.putText(img_copy, f"AR: {aspect_ratio}, C: {compactness}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img_copy, f"shape: {shape}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img_copy, f'Color: {color}', (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

cv2.imshow("figure", img)
cv2.imshow("mask", img_copy)
cv2.imwrite("image/result.jpg", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()