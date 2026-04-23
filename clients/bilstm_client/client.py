import os, time
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from uci_har import load_uci_har, apply_jitter

apply_jitter()

trainset, testset = load_uci_har()
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset,  batch_size=64)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden=64, layers=2, classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
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
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)}
        )
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
    _profile = os.environ.get('CLIENT_PROFILE', '')
    if _profile in ('orin', 'laptop', 'macbook'):
        import subprocess
        _tc = {'orin': ('50mbit', '20ms', '8ms'), 'laptop': ('150mbit', '10ms', '3ms'), 'macbook': ('300mbit', '5ms', '1ms')}
        _r, _d, _j = _tc[_profile]
        subprocess.run(['tc', 'qdisc', 'replace', 'dev', 'eth0', 'root', 'netem', 'rate', _r, 'delay', _d, _j], check=False)
    time.sleep(8)
    _mist = os.environ.get('MIST_ENABLED', '').lower() == 'true'
    _host = os.environ.get('FL_SERVER_ADDR', 'bilstm_server')
    if _mist:
        import threading, asyncio
        from mist_proxy import MISTConfig, run_client_proxy
        _cfg = MISTConfig(
            int(os.environ.get('MIST_P_FIXED', '262144')),
            int(os.environ.get('MIST_RATE', '10')),
            float(os.environ.get('MIST_SESSION_DURATION', '0')),
        )
        threading.Thread(target=lambda: asyncio.run(run_client_proxy(9091, _host, 9090, _cfg)), daemon=True).start()
        time.sleep(1)
        _addr = 'localhost:9091'
    else:
        _addr = f'{_host}:8080'
    fl.client.start_client(server_address=_addr, client=BiLSTMClient().to_client())
