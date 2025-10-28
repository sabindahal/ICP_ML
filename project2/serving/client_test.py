import grpc
import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = inference_pb2_grpc.InferenceStub(channel)

# req = inference_pb2.PredictRequest(features=[3, 2, 1800, 10, 5, 1, 8])  # beds,baths,sqft,age,dist,garage,school
req = inference_pb2.PredictRequest(
    features=[3, 2, 1800, 10, 5, 1, 8]
)
resp = stub.Predict(req)
print(f"y={resp.y:.2f} | infer_ms={resp.infer_ms:.2f}")