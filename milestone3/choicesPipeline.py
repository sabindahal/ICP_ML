from datasets import load_dataset
import numpy as np
import cv2
import math
import pytesseract
import re
import time
import easyocr

# Assertions
def accuracyChecker(words, testData):
    name = testData[0]['word']
    accuracy = 0
    yearFound = False
    monthFound = False
    dayFound = False
    IDFound = False

    for i in range(len(words)):
        if words[i] == name:
            accuracy += 1
        if re.fullmatch(r"\d{4}", words[i]) and not yearFound:
            if 2025 > int(words[i]) >= 1900:
                accuracy += 1
                yearFound = True
        if re.fullmatch(r"\d{1,2}", words[i]) and not monthFound:
            if 12 > int(words[i]) >= 1:
                accuracy += 1
                monthFound = True
        if re.fullmatch(r"\d{1,2}", words[i]) and not dayFound:
            if 31 > int(words[i]) >= 1:
                accuracy += 1
                dayFound = True
        if re.fullmatch(r"\d{18}", words[i]) and not IDFound:
            accuracy += 1
            IDFound = True
    return accuracy / 5

# Preprocessing
def preprocessing(img):
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

# Tesseract Model
def chooseTesseract(img, testingData):
    start = time.time()
    wordsRaw = pytesseract.image_to_string(img, lang='chi_sim')
    stop = time.time()
    latency = stop - start

    # clean and tokenize the data from OCR
    words = wordsRaw.replace(" ","").splitlines()

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
    accuracy = accuracyChecker(wordsData, testingData)
    return accuracy, latency

# EasyOCR model
def chooseEasyOCR(img, testingData):
    reader = easyocr.Reader(['ch_sim'])
    start = time.time()
    results = reader.readtext(img)
    stop = time.time()
    latency = stop - start

    # snag the words
    wordsRaw = []
    for res in results:
        wordsRaw.append(res[1])

    # clean and tokenize the data from OCR
    words = []
    for w in wordsRaw:
        w_clean = w.replace(" ", "")
        words.append(w_clean)

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
    accuracy = accuracyChecker(wordsData, testingData)
    return accuracy, latency

# This is what actually chooses which model to use, choice = 0 for tesseract, choice = 1 for easyOCR
# The img input needs to be a pillow img I believe, and the ground truth for comparison has to be from
# the lansinuote ID card set, as the processing for the ground truth is based on the structure of that dataset
# to avoid all issues, use img = ds['train'][imgNum]['image'] and testingData = ds['train'][imgNum]['ocr'],
# imgNum here is just whichever image of the 20000 you would like to select.
def chooseModel(choice, img, testingData):
    processedImg = preprocessing(img)
    if choice == 0:
        accuracy, latency = chooseTesseract(processedImg, testingData)
        return accuracy, latency
    elif choice == 1:
        accuracy, latency = chooseEasyOCR(processedImg, testingData)
        return accuracy, latency
    else:
        return "invalid choice"

# This is just an example run to show how the functions work. You can simply run the above
# function with an input of a choice, image, and the ground truth for accuracy comparison
ds = load_dataset("lansinuote/ocr_id_card")
accuracyTesseract = []
latencyTesseract = []
accuracyEasyOCR = []
latencyEasyOCR = []
numImages = len(ds['train'])
numImages = 10
for imgNum in range(numImages):
    currImg = ds['train'][imgNum]['image']
    testingData = ds['train'][imgNum]['ocr']
    if imgNum % 2 == 0:
        accuracy, latency = chooseModel(imgNum % 2, currImg, testingData)
        accuracyTesseract.append(accuracy)
        latencyTesseract.append(latency)
    elif imgNum % 2 == 1:
        accuracy, latency = chooseModel(imgNum % 2, currImg, testingData)
        accuracyEasyOCR.append(accuracy)
        latencyEasyOCR.append(latency)

print("Accuracy for tesseract: ", np.mean(accuracyTesseract))
print("Latency for tesseract: ", np.mean(latencyTesseract))
print("Accuracy for EasyOCR: ", np.mean(accuracyEasyOCR))
print("Latency for EasyOCR: ", np.mean(latencyEasyOCR))