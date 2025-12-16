import os, time
from concurrent.futures import ThreadPoolExecutor
import grpc
import inference_pb2 as pb
import inference_pb2_grpc as pbg
import numpy as np
import cv2
import math
import pytesseract
import re
from PIL import Image
import io

def preprocessingTesseract(img):
    img = Image.open(io.BytesIO(img)).convert("RGB")
    testingImageArray = np.array(img, dtype=np.uint8)
    edgeDetectionArray = cv2.cvtColor(testingImageArray, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(edgeDetectionArray, 100, 200)

    # Hough lines transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 75, None, 50, 10)
    if lines is None or len(lines) == 0:
        return testingImageArray  # fallback if no lines detected

    # Find angles of lines
    theta = np.zeros(len(lines))
    for i, l in enumerate(lines):
        l0 = l[0]
        if abs(l0[3] - l0[1]) == 0:
            theta[i] = 0
        else:
            theta[i] = math.atan(abs(l0[2] - l0[0]) / abs(l0[3] - l0[1]))

    # Find least horizontal line still fairly horizontal
    optTheta = 100
    optIndex = 0
    for k in range(len(theta)):
        if theta[k] > np.pi / 4 and theta[k] < optTheta:
            optTheta = theta[k]
            optIndex = k
    mostHorz = np.float32(lines[optIndex][0])

    # Compute points for affine warp
    mdpnt = [(mostHorz[0] + mostHorz[2]) / 2, (mostHorz[1] + mostHorz[3]) / 2]
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

    M = cv2.getAffineTransform(initialPts, finalPts)
    warpedImg = cv2.warpAffine(testingImageArray, M, (testingImageArray.shape[1], testingImageArray.shape[0]))

    # Redshift (keep only red channel)
    redshiftImg = warpedImg.copy()
    redshiftImg[:, :, 1] = 0
    redshiftImg[:, :, 2] = 0

    return redshiftImg


def chooseTesseractPerformance(img):
    wordsRaw = pytesseract.image_to_string(img, lang='chi_sim')
    words = wordsRaw.replace(" ", "").splitlines()
    pattern = r'\d+|[\u4e00-\u9fff]+|[A-Za-z]+|\s|[^\w\s]'
    tokens = [re.findall(pattern, w) for w in words]

    # flatten tokens
    wordsData = [t for token in tokens for t in token]
    return wordsData


DEMO_TEXT = os.getenv("DEMO_TEXT", "演示输出：你好，世界")  # Chinese demo text
MODEL_NAME = os.getenv("MODEL_NAME", "Tesseract")
POD_NAME = os.getenv("POD_NAME", "unknown-pod")
POD_IP = os.getenv("POD_IP", "unknown-ip")
STARTED_AT = int(os.getenv("STARTED_AT_UNIX", str(int(time.time()))))

PORT = int(os.getenv("GRPC_PORT", "50051"))

class Svc(pbg.InferenceServicer):
    def Predict(self, request, context):
        # tesseract
        t0 = time.perf_counter()
        try:
            img_bytes = request.image
            preprocessed_img = preprocessingTesseract(img_bytes)
            wordsData = chooseTesseractPerformance(preprocessed_img)
            text_output = ' '.join(wordsData)
        except Exception as e:
            text_output = f"OCR failed: {str(e)}"

        infer_ms = (time.perf_counter() - t0) * 1000.0

        return pb.PredictResponse(
            text=text_output,
            model=MODEL_NAME,
            infer_ms=infer_ms,
            pod_name=POD_NAME,
            pod_ip=POD_IP,
            started_at_unix=STARTED_AT,
        )

def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=16))
    pbg.add_InferenceServicer_to_server(Svc(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    print(f"[{MODEL_NAME}] gRPC ready on :{PORT} pod={POD_NAME} ip={POD_IP} started_at={STARTED_AT}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
