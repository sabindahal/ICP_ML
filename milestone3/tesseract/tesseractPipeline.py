from datasets import load_dataset
import numpy as np
import cv2
import math
import pytesseract
import re

# Preprocessing
def preprocessingTesseract(img):
    testingImageArray = np.array(img)
    edgeDetectionArray = cv2.cvtColor(testingImageArray, cv2.COLOR_RGB2GRAY)

    # apply canny edge detection
    edges = cv2.Canny(edgeDetectionArray, 100, 200)

    # do a hough lines transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 75, None, 50, 10)
    linesTest = np.zeros(edges.shape) + 250
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(linesTest, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # find the angles at which each line lies
    theta = np.zeros(len(lines))
    for i in range(0, len(lines)):
        l = lines[i][0]
        if abs(l[3] - l[1]) == 0:
            # horizontal line
            theta[i] = 0
        else:
            theta[i] = math.atan(abs(l[2] - l[0])/abs(l[3] - l[1]))

    # find smallest angle greater than pi/4
    optTheta = 100
    optIndex = 0
    for k in range(len(theta)):
        if theta[k] > np.pi/4:
            if theta[k] < optTheta:
                optTheta = theta[k]
                optIndex = k
    mostHorz = np.float32(lines[optIndex][0])
    # this actually ends up being the least horizontal of the still pretty horizontal lines, deals with noise from extra lines created from the text

    # find some points to warp the image, use the midpoint of the lines offset a bit, and the new destination of the endpoints of the least horizontal but still horizontal line
    mdpnt = [(mostHorz[0] + mostHorz[2])/2,(mostHorz[1] + mostHorz[3])/2]
    adjustment = 100
    if mdpnt[1] > 250:
        initialPts = np.float32([[mostHorz[0], mostHorz[1]],
                                 [mostHorz[2], mostHorz[3]],
                                 [mdpnt[0], mdpnt[1] - adjustment]])
        finalPts = np.float32([[mostHorz[0], mdpnt[1]],
                               [mostHorz[2], mdpnt[1]],
                               [mdpnt[0], mdpnt[1] - adjustment]])
    else:
        initialPts = np.float32([[mostHorz[0], mostHorz[1]],
                                 [mostHorz[2], mostHorz[3]],
                                 [mdpnt[0], mdpnt[1] + adjustment]])
        finalPts = np.float32([[mostHorz[0], mdpnt[1]],
                               [mostHorz[2], mdpnt[1]],
                               [mdpnt[0], mdpnt[1] + adjustment]])

    # warp it
    M = cv2.getAffineTransform(initialPts, finalPts)
    warpedImg = cv2.warpAffine(testingImageArray, M, (testingImageArray.shape[1], testingImageArray.shape[0]))

    # red it
    redshiftImg = warpedImg
    for m in range(warpedImg.shape[0]):
        for n in range(warpedImg.shape[1]):
            redshiftImg[m][n][1] = 0
            redshiftImg[m][n][2] = 0
    return redshiftImg


def chooseTesseractPerformance(img):
    wordsRaw = pytesseract.image_to_string(img, lang='chi_sim')

    # clean and tokenize the data from OCR
    words = wordsRaw.replace(" ", "").splitlines()

    pattern = r'\d+|[\u4e00-\u9fff]+|[A-Za-z]+|\s|[^\w\s]'
    tokens = []
    for i in range(len(words)):
        tokens.append(re.findall(pattern, words[i]))

    # flatten it out
    wordsData = []
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            wordsData.append(tokens[i][j])

    # run the accuracy checker on the final data
    return wordsData


# in practice, you would upload an image and run the code on that, this is an example image from
# our dataset
ds = load_dataset("lansinuote/ocr_id_card")

currImg = ds['train'][0]['image']

results = chooseTesseractPerformance(preprocessingTesseract(currImg))

print(results)

