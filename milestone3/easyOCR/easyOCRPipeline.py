from datasets import load_dataset
import numpy as np
import cv2
import easyocr

def preprocessingEasyOCR(img):
    testingImageArray = np.array(img)

    # greyscale it
    grey = cv2.cvtColor(testingImageArray, cv2.COLOR_RGB2GRAY)

    # gaussian blur
    grey = cv2.GaussianBlur(grey, (5, 5), 0)

    # binary it
    thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    return thresh

# EasyOCR model
def chooseEasyOCRPerformance(img, reader):

    results = reader.readtext(img)

    # snag words
    testing = []
    for m in range(len(results)):
        if results[m][2] > .25:
            testing.append(results[m][1])

    return testing

ds = load_dataset("lansinuote/ocr_id_card")

currImg = ds['train'][0]['image']

reader = easyocr.Reader(['ch_sim'])

results = chooseEasyOCRPerformance(preprocessingEasyOCR(currImg), reader)

print(results)
