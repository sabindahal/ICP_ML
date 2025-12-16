FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY common/inference.proto /app/inference.proto
RUN python -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app /app/inference.proto

COPY router.py /app/router.py
COPY web /app/web

ENV ROUTER_HTTP_PORT=8080
EXPOSE 8080

CMD ["python", "router.py"]