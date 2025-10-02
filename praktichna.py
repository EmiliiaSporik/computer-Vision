import cv2
import numpy as np

fon = np.zeros((400, 600, 3), np.uint8)
fon[:] = 153, 119, 84
cv2.rectangle(fon, (10, 10), (590, 390), (204, 217, 137), 2)


cv2.putText(fon, "Emiliia Sporik", (170, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
cv2.putText(fon, "Computer Vision Student", (170, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "Email: emily10sporik@gmail.com", (170, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "Phone: +380 93 320 6604", (170, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "10/09/2010", (170, 260), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "OpenCV Business Card", (130, 360), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)


x, y = 30, 50
img = cv2.resize(cv2.imread("image/foto.jpg"), (120, 150))
fon[y:y + 150, x:x + 120] = img

qr_size = 100
x1 = 600 - qr_size - 100
y1 = 400 - qr_size - 70
img1 = cv2.resize(cv2.imread("image/qrcode.jpg"), (qr_size, qr_size))
fon[y1:y1 + qr_size, x1:x1 + qr_size] = img1




cv2.imshow("OpenCV Business Card", fon)
cv2.imwrite("image/business_card.png", fon)

cv2.waitKey(0)
cv2.destroyAllWindows()