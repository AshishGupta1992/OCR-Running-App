from PIL import Image, ImageTk
import numpy as np
from pytesseract import Output
from xlwt import Workbook
from os import listdir
from PIL import Image, ImageFile
import pytesseract
import cv2
import ftfy
import json
import os
import io
import re
import pandas as pd

image = cv2.imread('D:\Personal\Machine Learning\PAN CARD\pancard6.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the
    # image
#gray = cv2.threshold(gray, 0, 255,
#                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

#gray = cv2.medianBlur(gray, 3)

#gray = cv2.bilateralFilter(gray, 9, 75, 75)
kernel = np.ones((1, 1), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=2)
gray = cv2.erode(gray, kernel, iterations=2)


#gray = cv2.GaussianBlur(gray, (5, 5), 0)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Ashish.Gupta\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
text = pytesseract.image_to_string(Image.open(filename), lang='eng')
        # add +hin after eng within the same argument to extract hindi specific text - change encoding to utf-8 while writing
print(text)
os.remove(filename)
        # print(text)
