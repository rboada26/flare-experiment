FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY uci_har.py .
RUN python -c "from uci_har import load_uci_har; load_uci_har()"
COPY lstm_client/client.py .
CMD ["python", "client.py"]
