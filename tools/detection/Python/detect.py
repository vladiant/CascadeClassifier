# https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/

import argparse as ap

import numpy as np
import cv2

parser = ap.ArgumentParser()
parser.add_argument("-d", "--descriptor", help="cascade descriptor XML file", required="True")
parser.add_argument("-i", "--image", help="image filename", required="True")
args = vars(parser.parse_args())

descriptor_path = args["descriptor"]
object_cascade = cv2.CascadeClassifier(descriptor_path)

image_path = args["image"]
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

objects = object_cascade.detectMultiScale(gray, 4, 50)

for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
