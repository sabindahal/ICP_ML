import os, time, random
from concurrent.futures import ThreadPoolExecutor

import grpc
import inference_pb2 as pb
import inference_pb2_grpc as pbg

DEMO_TEXT = os.getenv("DEMO_TEXT", "演示输出：你好，世界")  # Chinese demo text
MODEL_NAME = os.getenv("MODEL_NAME", "model")
POD_NAME = os.getenv("POD_NAME", "unknown-pod")
POD_IP = os.getenv("POD_IP", "unknown-ip")
STARTED_AT = int(os.getenv("STARTED_AT_UNIX", str(int(time.time()))))

PORT = int(os.getenv("GRPC_PORT", "50051"))

class Svc(pbg.InferenceServicer):
    def Predict(self, request, context):
        # Simulate variable work (replace with real OCR later)
        t0 = time.perf_counter()
        time.sleep(random.uniform(0.01, 0.08))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        return pb.PredictResponse(
            text=DEMO_TEXT,
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
