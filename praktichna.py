import cv2
import numpy as np

fon = np.zeros((400, 600, 3), np.uint8)
fon[:] = 153, 119, 84
cv2.rectangle(fon, (10, 10), (590, 390), (204, 217, 137), 2)

img = cv2.imread("image/foto.jpg")
img = cv2.resize(img, (img.shape[1] // 20, img.shape[0] // 20))

cv2.putText(fon, "Emiliia Sporik", (180, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
cv2.putText(fon, "Computer Vision Student", (180, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "Email: emily10sporik@gmail.com", (180, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "Phone: +380 93 320 6604", (180, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "10/09/2010", (180, 190), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
cv2.putText(fon, "OpenCV Business Card", (150, 360), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

img1 = cv2.imread("image/qrcode.jpg")
img1 = cv2.resize(img, (img1.shape[1] // 10, img1.shape[0] // 10))



cv2.imshow("Card", fon)
cv2.imshow("Image", img)
cv2.imshow("Image1", img1)

cv2.imwrite("business_card.png", fon)

cv2.waitKey(0)
cv2.destroyAllWindows()
