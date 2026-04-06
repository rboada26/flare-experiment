#!/usr/bin/env bash
set -euo pipefail

echo "Creating new client directories..."
mkdir -p clients/resnet_client clients/mobilenet_client clients/lstm_client clients/bilstm_client

echo "Writing resnet client..."
cat >clients/resnet_client/client.py <<'EOF'
import flwr as fl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
full_train = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
full_test  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
trainset   = Subset(full_train, range(500))
testset    = Subset(full_test,  range(100))
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader  = DataLoader(testset,  batch_size=128)

device = torch.device("cpu")
model  = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model  = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class ResNetClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]
    def set_parameters(self, params):
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)})
        model.load_state_dict(state, strict=True)
    def fit(self, params, config):
        self.set_parameters(params)
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        return self.get_parameters(config), len(trainset), {}
    def evaluate(self, params, config):
        self.set_parameters(params)
        model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return loss / len(testloader), total, {"accuracy": correct / total}

if __name__ == "__main__":
    import os, time
    time.sleep(8)
    fl.client.start_client(
        server_address=f"{os.environ.get('FL_SERVER_ADDR','resnet_server')}:8080",
        client=ResNetClient().to_client(),
    )
EOF

echo "Writing resnet Dockerfile..."
cat >clients/resnet_client/Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY resnet_client/client.py .
CMD ["python", "client.py"]
EOF

echo "Writing mobilenet client..."
cat >clients/mobilenet_client/client.py <<'EOF'
import flwr as fl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
full_train = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
full_test  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
trainset   = Subset(full_train, range(500))
testset    = Subset(full_test,  range(100))
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset,  batch_size=64)

device = torch.device("cpu")
model  = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 10)
model  = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class MobileNetClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]
    def set_parameters(self, params):
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)})
        model.load_state_dict(state, strict=True)
    def fit(self, params, config):
        self.set_parameters(params)
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        return self.get_parameters(config), len(trainset), {}
    def evaluate(self, params, config):
        self.set_parameters(params)
        model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return loss / len(testloader), total, {"accuracy": correct / total}

if __name__ == "__main__":
    import os, time
    time.sleep(8)
    fl.client.start_client(
        server_address=f"{os.environ.get('FL_SERVER_ADDR','mobilenet_server')}:8080",
        client=MobileNetClient().to_client(),
    )
EOF

cat >clients/mobilenet_client/Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mobilenet_client/client.py .
CMD ["python", "client.py"]
EOF

echo "Writing lstm client..."
cat >clients/lstm_client/client.py <<'EOF'
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

def make_dataset(n, seq_len=50, features=9, classes=6):
    return TensorDataset(torch.randn(n, seq_len, features), torch.randint(0, classes, (n,)))

trainset    = make_dataset(500)
testset     = make_dataset(100)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader  = DataLoader(testset,  batch_size=128)

class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden=64, layers=1, classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device    = torch.device("cpu")
model     = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class LSTMClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]
    def set_parameters(self, params):
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)})
        model.load_state_dict(state, strict=True)
    def fit(self, params, config):
        self.set_parameters(params)
        model.train()
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
        return self.get_parameters(config), len(trainset), {}
    def evaluate(self, params, config):
        self.set_parameters(params)
        model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return loss / len(testloader), total, {"accuracy": correct / total}

if __name__ == "__main__":
    import os, time
    time.sleep(8)
    fl.client.start_client(
        server_address=f"{os.environ.get('FL_SERVER_ADDR','lstm_server')}:8080",
        client=LSTMClient().to_client(),
    )
EOF

cat >clients/lstm_client/Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lstm_client/client.py .
CMD ["python", "client.py"]
EOF

echo "Writing bilstm client..."
cat >clients/bilstm_client/client.py <<'EOF'
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

def make_dataset(n, seq_len=50, features=9, classes=6):
    return TensorDataset(torch.randn(n, seq_len, features), torch.randint(0, classes, (n,)))

trainset    = make_dataset(500)
testset     = make_dataset(100)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader  = DataLoader(testset,  batch_size=128)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden=64, layers=1, classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden * 2, classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device    = torch.device("cpu")
model     = BiLSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class BiLSTMClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]
    def set_parameters(self, params):
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)})
        model.load_state_dict(state, strict=True)
    def fit(self, params, config):
        self.set_parameters(params)
        model.train()
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
        return self.get_parameters(config), len(trainset), {}
    def evaluate(self, params, config):
        self.set_parameters(params)
        model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return loss / len(testloader), total, {"accuracy": correct / total}

if __name__ == "__main__":
    import os, time
    time.sleep(8)
    fl.client.start_client(
        server_address=f"{os.environ.get('FL_SERVER_ADDR','bilstm_server')}:8080",
        client=BiLSTMClient().to_client(),
    )
EOF

cat >clients/bilstm_client/Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bilstm_client/client.py .
CMD ["python", "client.py"]
EOF

echo "Writing new docker-compose.yml..."
cat >docker-compose.yml <<'EOF'
networks:
  simplecnn_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  resnet_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.1.0/24
  mobilenet_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.2.0/24
  gru_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/24
  lstm_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.1.0/24
  bilstm_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.2.0/24

services:
  simplecnn_server:
    build: ./server
    container_name: fl_simplecnn_server
    networks:
      simplecnn_network:
        ipv4_address: 172.20.0.10
    ports:
      - "8080:8080"

  simplecnn_client:
    build:
      context: ./clients
      dockerfile: cnn_client/Dockerfile
    container_name: fl_simplecnn_client
    networks:
      simplecnn_network:
        ipv4_address: 172.20.0.11
    environment:
      - FL_SERVER_ADDR=172.20.0.10
    depends_on:
      - simplecnn_server

  simplecnn_client2:
    build:
      context: ./clients
      dockerfile: cnn_client/Dockerfile
    container_name: fl_simplecnn_client2
    networks:
      simplecnn_network:
        ipv4_address: 172.20.0.12
    environment:
      - FL_SERVER_ADDR=172.20.0.10
    depends_on:
      - simplecnn_server

  resnet_server:
    build: ./server
    container_name: fl_resnet_server
    networks:
      resnet_network:
        ipv4_address: 172.20.1.10
    ports:
      - "8081:8080"

  resnet_client:
    build:
      context: ./clients
      dockerfile: resnet_client/Dockerfile
    container_name: fl_resnet_client
    networks:
      resnet_network:
        ipv4_address: 172.20.1.11
    environment:
      - FL_SERVER_ADDR=172.20.1.10
    depends_on:
      - resnet_server

  resnet_client2:
    build:
      context: ./clients
      dockerfile: resnet_client/Dockerfile
    container_name: fl_resnet_client2
    networks:
      resnet_network:
        ipv4_address: 172.20.1.12
    environment:
      - FL_SERVER_ADDR=172.20.1.10
    depends_on:
      - resnet_server

  mobilenet_server:
    build: ./server
    container_name: fl_mobilenet_server
    networks:
      mobilenet_network:
        ipv4_address: 172.20.2.10
    ports:
      - "8082:8080"

  mobilenet_client:
    build:
      context: ./clients
      dockerfile: mobilenet_client/Dockerfile
    container_name: fl_mobilenet_client
    networks:
      mobilenet_network:
        ipv4_address: 172.20.2.11
    environment:
      - FL_SERVER_ADDR=172.20.2.10
    depends_on:
      - mobilenet_server

  mobilenet_client2:
    build:
      context: ./clients
      dockerfile: mobilenet_client/Dockerfile
    container_name: fl_mobilenet_client2
    networks:
      mobilenet_network:
        ipv4_address: 172.20.2.12
    environment:
      - FL_SERVER_ADDR=172.20.2.10
    depends_on:
      - mobilenet_server

  gru_server:
    build: ./server
    container_name: fl_gru_server
    networks:
      gru_network:
        ipv4_address: 172.21.0.10
    ports:
      - "8083:8080"

  gru_client:
    build:
      context: ./clients
      dockerfile: rnn_client/Dockerfile
    container_name: fl_gru_client
    networks:
      gru_network:
        ipv4_address: 172.21.0.11
    environment:
      - FL_SERVER_ADDR=172.21.0.10
    depends_on:
      - gru_server

  gru_client2:
    build:
      context: ./clients
      dockerfile: rnn_client/Dockerfile
    container_name: fl_gru_client2
    networks:
      gru_network:
        ipv4_address: 172.21.0.12
    environment:
      - FL_SERVER_ADDR=172.21.0.10
    depends_on:
      - gru_server

  lstm_server:
    build: ./server
    container_name: fl_lstm_server
    networks:
      lstm_network:
        ipv4_address: 172.21.1.10
    ports:
      - "8084:8080"

  lstm_client:
    build:
      context: ./clients
      dockerfile: lstm_client/Dockerfile
    container_name: fl_lstm_client
    networks:
      lstm_network:
        ipv4_address: 172.21.1.11
    environment:
      - FL_SERVER_ADDR=172.21.1.10
    depends_on:
      - lstm_server

  lstm_client2:
    build:
      context: ./clients
      dockerfile: lstm_client/Dockerfile
    container_name: fl_lstm_client2
    networks:
      lstm_network:
        ipv4_address: 172.21.1.12
    environment:
      - FL_SERVER_ADDR=172.21.1.10
    depends_on:
      - lstm_server

  bilstm_server:
    build: ./server
    container_name: fl_bilstm_server
    networks:
      bilstm_network:
        ipv4_address: 172.21.2.10
    ports:
      - "8085:8080"

  bilstm_client:
    build:
      context: ./clients
      dockerfile: bilstm_client/Dockerfile
    container_name: fl_bilstm_client
    networks:
      bilstm_network:
        ipv4_address: 172.21.2.11
    environment:
      - FL_SERVER_ADDR=172.21.2.10
    depends_on:
      - bilstm_server

  bilstm_client2:
    build:
      context: ./clients
      dockerfile: bilstm_client/Dockerfile
    container_name: fl_bilstm_client2
    networks:
      bilstm_network:
        ipv4_address: 172.21.2.12
    environment:
      - FL_SERVER_ADDR=172.21.2.10
    depends_on:
      - bilstm_server
EOF

echo "Writing new collect_data.sh..."
cat >collect_data.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

NUM_SESSIONS=8
ARCHITECTURES=(simplecnn resnet mobilenet gru lstm bilstm)
SUBNETS=(172.20.0 172.20.1 172.20.2 172.21.0 172.21.1 172.21.2)

wait_for_all_clients() {
    echo "Polling for training completion..."
    while true; do
        all_done=true
        status_line=""
        for arch in "${ARCHITECTURES[@]}"; do
            s1=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client"  2>/dev/null || echo "gone")
            s2=$(docker inspect -f '{{.State.Status}}' "fl_${arch}_client2" 2>/dev/null || echo "gone")
            status_line+="${arch}:${s1}/${s2} "
            if [[ "$s1" != "exited" || "$s2" != "exited" ]]; then
                all_done=false
            fi
        done
        echo "  $status_line"
        $all_done && { echo "All clients finished."; break; }
        sleep 5
    done
}

for i in $(seq 1 $NUM_SESSIONS); do
    echo ""
    echo "============================================"
    echo " Starting session $i / $NUM_SESSIONS"
    echo "============================================"

    docker compose down --remove-orphans 2>/dev/null || true
    sleep 2

    docker compose up -d \
      simplecnn_server resnet_server mobilenet_server \
      gru_server lstm_server bilstm_server
    sleep 10

    mkdir -p captures
    PIDS=()
    for idx in "${!ARCHITECTURES[@]}"; do
        arch="${ARCHITECTURES[$idx]}"
        subnet="${SUBNETS[$idx]}"
        net_name="flare-experiment_${arch}_network"
        net_id=$(docker network inspect "$net_name" --format '{{.Id}}' 2>/dev/null | cut -c1-12)
        iface="br-${net_id}"
        pcap="captures/session${i}_${arch}_$(date +%Y%m%d_%H%M%S).pcap"
        filter="(src net ${subnet}.0/24 and dst net ${subnet}.0/24) and not multicast"
        sudo tcpdump -i "$iface" -w "$pcap" -s0 "$filter" &
        PIDS+=($!)
        sleep 0.5
    done

    sleep 2

    docker compose up -d \
      simplecnn_client simplecnn_client2 \
      resnet_client resnet_client2 \
      mobilenet_client mobilenet_client2 \
      gru_client gru_client2 \
      lstm_client lstm_client2 \
      bilstm_client bilstm_client2

    wait_for_all_clients

    sleep 3

    for pid in "${PIDS[@]}"; do
        sudo kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    docker compose down --remove-orphans
    sleep 3

    echo "Session $i done:"
    ls -lh captures/session${i}_*.pcap 2>/dev/null | awk '{print $5, $9}'
done

echo ""
echo "All sessions complete."
ls -lh captures/
EOF

chmod +x collect_data.sh

echo ""
echo "Setup complete. Verifying structure..."
echo ""
ls clients/
echo ""
wc -l docker-compose.yml
echo ""
echo "Ready to build. Run:"
echo "  docker compose down --remove-orphans"
echo "  docker system prune -f"
echo "  docker compose build --no-cache"
