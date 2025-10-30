import os, time
from io import BytesIO
import grpc
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
from minio import Minio
import pandas as pd

import cv2
import torch
import torch.nn as nn

import inference_pb2, inference_pb2_grpc

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_USER     = os.getenv("MINIO_USER", "minio")
MINIO_PASS     = os.getenv("MINIO_PASS", "minio123")
BUCKET         = os.getenv("REGISTRY_BUCKET", "model-registry")
VARIANT        = os.getenv("VARIANT") 


def predenoise_red(img_bgr):
    img = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = [theta for rho, theta in lines[:, 0]]
        angle = (float(np.mean(angles)) - np.pi/2) * 180/np.pi
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    b,g,r = cv2.split(img)
    red_only = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
    gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (128, 32))

class OCRNet(nn.Module):
    def __init__(self, nclass=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.lstm = nn.LSTM(64*8,128,num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, nclass)
    def forward(self,x):
        x = self.conv(x)
        b,c,h,w = x.size()
        x = x.permute(0,3,1,2).reshape(b,w,-1)
        x,_ = self.lstm(x)
        return self.fc(x)


def minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)

def read_minio_bytes(bucket, key):
    cli = minio_client()
    resp = cli.get_object(bucket, key)
    data = resp.read()
    resp.close(); resp.release_conn()
    return data

def object_exists(bucket, key):
    try:
        cli = minio_client()
        cli.stat_object(bucket, key)
        return True
    except Exception:
        return False


class ModelWrapper:
    def __init__(self, variant:str):
        self.variant = variant
        self.mode = None  
        if object_exists(BUCKET, f"{variant}/model.pt"):
            self.mode = "ocr"
            self.ocr = OCRNet()
            state = torch.load(BytesIO(read_minio_bytes(BUCKET, f"{variant}/model.pt")), map_location="cpu")
            self.ocr.load_state_dict(state)
            self.ocr.eval()
        else:
            self.mode = "tabular"
            blob = read_minio_bytes(BUCKET, f"{variant}/model.joblib")
            self.sklearn = joblib.load(BytesIO(blob))
            self.columns = ["beds","baths","sqft","age","dist_to_center_km","has_garage","school_score"]

    def predict(self, request):
        if request.WhichOneof("input") == "image" and self.mode == "ocr":
            img_bytes = bytes(request.image)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Invalid image bytes")
            proc = predenoise_red(img_bgr)
            tens = torch.tensor(proc/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.ocr(tens)  # [B, T, C]
            infer_ms = (time.perf_counter() - t0) * 1000.0
            y = float(logits.shape[1])  
            return y, infer_ms

        elif request.WhichOneof("input") == "features" and self.mode == "tabular":
            feats = np.asarray(request.features.values, dtype=float).ravel()
            if feats.size != len(self.columns):
                raise ValueError(f"Expected {len(self.columns)} features, got {feats.size}")
            X = pd.DataFrame([feats.tolist()], columns=self.columns)
            t0 = time.perf_counter()
            yhat = float(self.sklearn.predict(X)[0])
            infer_ms = (time.perf_counter() - t0) * 1000.0
            return yhat, infer_ms

        else:
            if self.mode == "ocr":
                raise ValueError("This variant expects image bytes in PredictRequest.image")
            else:
                raise ValueError("This variant expects float features in PredictRequest.features")

class InferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(self, mw: ModelWrapper):
        self.mw = mw

    def Predict(self, request, context):
        try:
            y, infer_ms = self.mw.predict(request)
            return inference_pb2.PredictResponse(y=y, infer_ms=infer_ms)
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return inference_pb2.PredictResponse(y=0.0, infer_ms=0.0)

def serve():
    assert VARIANT, "Set VARIANT env (e.g., VARIANT=model2)"
    mw = ModelWrapper(VARIANT)
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceService(mw), server)
    server.add_insecure_port("[::]:50051")
    print(f"{VARIANT} loaded in mode={mw.mode}; serving on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()