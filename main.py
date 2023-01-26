import cv2
import numpy as np
import easyocr
import pytesseract
import imutils
from matplotlib import pyplot as pl


cap = cv2.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 300)

img = cv2.imread('images/card2.jpg')


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray_img, 11, 15, 15)

edges = cv2.Canny(img_filter, 30, 50)

img_contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = imutils.grab_contours(img_contours)
img_contours = sorted(img_contours, key=cv2.contourArea, reverse=True)

position = -1
for i in img_contours:
    variable = cv2.approxPolyDP(i, 5, True)
    if (len(variable) == 4):
        position = variable
        break

width = abs(position[0][0][0] - position[2][0][0])
height = abs(position[0][0][1] - position[1][0][1])
move_side = width // 4
move_height = height // 7


mask1 = np.zeros(gray_img.shape, np.uint8)
new_img = cv2.drawContours(mask1, [position], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask1)


(x, y) = np.where(mask1 == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

part = gray_img[x1:y1, x2:y2]
text_part = gray_img[x1 - 30:x1 + 130, y1 - 60:y1 + 60]
"""
text = easyocr.Reader(['en'])
text = text.readtext(text_part)
print(text)


result = 'J'
finale = cv2.putText(img, result, (y1 + 70, x2 + 80), cv2.FONT_HERSHEY_PLAIN, 5, (120, 100, 10), 5)"""
finale = cv2.rectangle(img, (y1, x1), (y2, x2), (200, 100, 200), 5)


cv2.imshow('Result', finale)
cv2.imshow('YOUR CARD', text_part)


cv2.waitKey(0)
exit(0)
