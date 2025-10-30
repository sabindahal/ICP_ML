FROM python:3.11-slim
WORKDIR /app
COPY inference.proto /app/
RUN pip install --no-cache-dir grpcio grpcio-tools
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py /app/
EXPOSE 50051

CMD ["python", "server.py"]
