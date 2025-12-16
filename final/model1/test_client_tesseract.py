import grpc
from final.common import inference_pb2 as pb
from final.common import inference_pb2_grpc as pbg
from datasets import load_dataset
import io

SERVER_ADDRESS = "127.0.0.1:50052"  # match your server port

def main():
    # Connect to gRPC server
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = pbg.InferenceStub(channel)

    dataset = load_dataset("lansinuote/ocr_id_card")
    example_img = dataset['train'][0]['image']

    img_byte_arr = io.BytesIO()
    example_img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Make a Predict request
    request = pb.PredictRequest(image=img_bytes)
    response = stub.Predict(request)

    print("OCR Result:")
    print(response.text)
    print(f"Model: {response.model}")
    print(f"Inference time: {response.infer_ms:.2f} ms")

if __name__ == "__main__":
    main()