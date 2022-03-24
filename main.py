import cv2 as cv
import numpy as np
import pytesseract
import cvzone
from utils import *

"""PyTesseract is being used in this Program for fast and easy Text Detection"""

colorFinder = cvzone.ColorFinder(False)

image = cv.imread("Test_Images/img.png")

imageHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
hsvValue = {'hmin': 24, 'smin': 226, 'vmin': 255, 'hmax': 140, 'smax': 228,
            'vmax': 255}  # hsvValue of colour used for highlights
# OR  {'hmin': 24, 'smin': 140, 'vmin': 238, 'hmax': 50, 'smax': 255, 'vmax': 255}

imgResult, mask = colorFinder.update(imageHsv, hsvValue)
min_area = 200
# Keeps track of the minimum area(used to reduce noise)
image_with_contours, contours_found = cvzone.findContours(image, mask, min_area, False)

roiList = getRoi(image, contours_found)

highlighted_texts = roi_display_and_write(roiList)  # Displays the each portion containing the highlighted texts

cv.imshow("Original Image", image)
cv.waitKey(0)
