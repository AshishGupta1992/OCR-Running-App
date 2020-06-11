import cv2
import numpy as np

image = cv2.imread(r'D:\Personal\Machine Learning\PAN CARD\pancard2.jpg')
hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

low_black = np.array([0, 0, 200])
high_black = np.array([180, 20, 255])

blue_mask = cv2.inRange(hsv_frame, low_black, high_black)
blue = cv2.bitwise_and(image, image, mask=blue_mask)

cv2.imshow("Orig", image)
cv2.imshow("Orig1", blue_mask)

cv2.imshow("Green", blue)

cv2.waitKey(0)
cv2.destroyAllWindows()
