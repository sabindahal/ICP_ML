FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchvision==0.17.2

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY common/inference.proto /app/inference.proto
RUN python -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app /app/inference.proto

COPY server.py /app/server.py

ENV PYTHONUNBUFFERED=1

ENV EASYOCR_MODULE_PATH=/models/easyocr
RUN python - <<'PY'
import os
import easyocr

path = os.getenv("EASYOCR_MODULE_PATH", "/models/easyocr")
os.makedirs(path, exist_ok=True)

# Pre-download models at build time
easyocr.Reader(['en','ch_sim'], gpu=False, model_storage_directory=path)

print("EasyOCR models ready in:", path)
PY
ENV GRPC_PORT=50051
EXPOSE 50051

CMD ["python", "server.py"]
