import cv2
import numpy as np

image = cv2.imread("image/me.jpg")
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 300, 300)
kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations = 1)
image = cv2.erode(image, kernel, iterations = 1)
cv2.imshow("Image", image[0:700, 150:500])

cv2.waitKey(0)
cv2.destroyAllWindows()

image1 = cv2.imread("image/gmail.jpg")
image1 = cv2.resize(image1, (image1.shape[1] // 4, image1.shape[0] // 4))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
print(image1.shape)
image1 = cv2.Canny(image1, 700, 700)
kernel = np.ones((3, 3), np.uint8)
image1 = cv2.dilate(image1, kernel, iterations = 1)
image1 = cv2.erode(image1, kernel, iterations = 1)
cv2.imshow("Image", image1)

cv2.waitKey(0)
cv2.destroyAllWindows()