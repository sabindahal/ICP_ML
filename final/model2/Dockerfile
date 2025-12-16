FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libxext6 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchvision==0.17.2

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY common/inference.proto /app/inference.proto
RUN python -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app /app/inference.proto

COPY server.py /app/server.py

ENV GRPC_PORT=50051
EXPOSE 50051

CMD ["python", "server.py"]
