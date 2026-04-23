FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flwr==1.8.0
COPY server.py .
COPY mist_proxy.py .
CMD ["python", "server.py"]
