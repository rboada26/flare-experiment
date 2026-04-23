FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cnn_client/client.py .
COPY mist_proxy.py .
# Pre-download dataset at build time so no external traffic during experiments
RUN python -c "import torchvision; torchvision.datasets.CIFAR10('./data', download=True)"
CMD ["python", "client.py"]
