import os, time
from io import BytesIO

import grpc
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
from minio import Minio
import pandas as pd

import inference_pb2, inference_pb2_grpc

# ---- Env (configure at runtime) ----
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_USER     = os.getenv("MINIO_USER", "minio")
MINIO_PASS     = os.getenv("MINIO_PASS", "minio123")
BUCKET         = os.getenv("REGISTRY_BUCKET", "model-registry")
VARIANT        = os.getenv("VARIANT")  # e.g., "RF_v2", "Ridge_v1"

def load_model_from_minio(variant: str):
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)
    obj = client.get_object(BUCKET, f"{variant}/model.joblib")
    data = obj.read()
    return joblib.load(BytesIO(data))

class InferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(self, model):
        self.model = model

    def Predict(self, request, context):
        import numpy as np
        import pandas as pd
        columns = ["beds", "baths", "sqft", "age", "dist_to_center_km", "has_garage", "school_score"]

        # --- Normalize input to 1-D numeric array ---
        feats = np.asarray(request.features, dtype=float)
        if feats.ndim > 1:
            feats = feats.ravel()

        # --- Validate input length ---
        if feats.size != len(columns):
            context.set_details(f"Expected {len(columns)} features but got {feats.size}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return inference_pb2.PredictResponse(y=0.0, infer_ms=0.0)

        # --- Build DataFrame ---
        X = pd.DataFrame([feats.tolist()], columns=columns)

        # --- Inference ---
        t0 = time.perf_counter()
        y = float(self.model.predict(X)[0])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return inference_pb2.PredictResponse(y=y, infer_ms=infer_ms)

def serve():
    assert VARIANT, "Set VARIANT env (e.g., VARIANT=RF_v2)"
    model = load_model_from_minio(VARIANT)

    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceService(model), server)
    server.add_insecure_port("[::]:50051")
    print(f"✅ Loaded {VARIANT} from MinIO; serving on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
