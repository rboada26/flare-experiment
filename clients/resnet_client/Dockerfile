FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download dataset at build time so no external traffic during experiments
RUN python -c "import torchvision; torchvision.datasets.CIFAR10('./data', download=True)"
COPY resnet_client/client.py .
CMD ["python", "client.py"]
