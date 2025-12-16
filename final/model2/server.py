import os, time
from concurrent.futures import ThreadPoolExecutor

import grpc
import inference_pb2 as pb
import inference_pb2_grpc as pbg

import numpy as np
import cv2
import easyocr
from PIL import Image
import io

def preprocessingEasyOCR(img):
    img = Image.open(io.BytesIO(img)).convert("RGB")
    testingImageArray = np.array(img, dtype=np.uint8)

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

# create reader for easyOCR
reader = easyocr.Reader(['ch_sim'])

DEMO_TEXT = os.getenv("DEMO_TEXT", "演示输出：你好，世界")  # Chinese demo text
MODEL_NAME = os.getenv("MODEL_NAME", "EasyOCR")
POD_NAME = os.getenv("POD_NAME", "unknown-pod")
POD_IP = os.getenv("POD_IP", "unknown-ip")
STARTED_AT = int(os.getenv("STARTED_AT_UNIX", str(int(time.time()))))

PORT = int(os.getenv("GRPC_PORT", "50051"))

class Svc(pbg.InferenceServicer):
    def Predict(self, request, context):
        # Simulate variable work (replace with real OCR later)
        t0 = time.perf_counter()
        try:
            img_bytes = request.image
            preprocessed_img = preprocessingEasyOCR(img_bytes)
            wordsData = chooseEasyOCRPerformance(preprocessed_img, reader)
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
