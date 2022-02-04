# https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/
import numpy as np
import cv2

watch_cascade = cv2.CascadeClassifier("cascade.xml")

img = cv2.imread("airplane_001.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

watches = watch_cascade.detectMultiScale(gray, 4, 50)

for (x, y, w, h) in watches:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
