import cv2
import numpy as np

img = cv2.imread("image/me.jpg")
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
cv2.putText(img, "Emiliia Sporik", (260, 290), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)

cv2.rectangle(img, (270, 180), (350, 270), (0, 0, 0), 2)
cv2.imshow("Image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()