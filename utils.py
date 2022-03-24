import cv2 as cv
import pytesseract
import cvzone

# Location of the Pytesseract executable file in the computer
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"


def getRoi(img, contours):
    """Gets the regions of interest(ROI) from the original image"""
    roiList = []
    for contour in contours:
        x, y, w, h = contour['bbox']
        roiList.append(img[y:y + h, x:x + w])
    return roiList


def roi_display_and_write(roiList, display: bool = False, write: bool = True):
    """Displays and/or writes the text found in a file provided"""
    highlights = []
    try:
        file = open("text.txt", "w")
    except FileNotFoundError:
        print("----------File Not Found----------")
        return None
    invertedRoiList = reversed(roiList)
    for idx, roi in enumerate(invertedRoiList):  # idx will serve for the naming of the created windows
        roi_copy = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
        image_text_data = pytesseract.image_to_string(roi_copy)
        image_text_data = image_text_data.strip("\n")
        # image_text_data.strip("")
        highlights.append(image_text_data)
        roi_resized = cv.resize(roi, (0, 0), None, 2,
                                2)  # Increases the size of the window, to make dragging of it possible
        if display:
            cv.imshow(f"Image {idx}", roi_resized)
        if write:
            file.write(f"{image_text_data}\n")
    file.close()
    return highlights
