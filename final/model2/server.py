import os
import time
import io
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
import cv2
import easyocr
from PIL import Image

import inference_pb2 as pb
import inference_pb2_grpc as pbg


# =========================
# EasyOCR setup (load from local filesystem)
# =========================
MODEL_DIR = os.getenv("EASYOCR_MODULE_PATH", "/models/easyocr")
# Create the reader once at startup (Chinese simplified)
reader = easyocr.Reader(
    ["ch_sim"],
    gpu=False,
    model_storage_directory=MODEL_DIR,
)

# =========================
# Metadata / config
# =========================
DEMO_TEXT = os.getenv("DEMO_TEXT", "演示输出：你好，世界")
MODEL_NAME = os.getenv("MODEL_NAME", "EasyOCR")
POD_NAME = os.getenv("POD_NAME", "unknown-pod")
POD_IP = os.getenv("POD_IP", "unknown-ip")
STARTED_AT = int(os.getenv("STARTED_AT_UNIX", str(int(time.time()))))
PORT = int(os.getenv("GRPC_PORT", "50051"))


# =========================
# Preprocessing + OCR
# =========================
def preprocessing_easyocr(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return thresh


def run_easyocr(preprocessed_img: np.ndarray) -> str:
    results = reader.readtext(preprocessed_img)
    words = [txt for (_bbox, txt, conf) in results if conf is not None and conf > 0.25]
    return " ".join(words)


# =========================
# gRPC service
# =========================
class Svc(pbg.InferenceServicer):
    def Predict(self, request, context):
        t0 = time.perf_counter()
        try:
            text_output = run_easyocr(preprocessing_easyocr(request.image))
        except Exception as e:
            text_output = f"OCR failed: {e}"

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

    print(
        f"[{MODEL_NAME}] gRPC ready on :{PORT} "
        f"pod={POD_NAME} ip={POD_IP} started_at={STARTED_AT} "
        f"easyocr_models={MODEL_DIR}",
        flush=True,
    )

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
