import os, time
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
trainset   = Subset(full_train, range(50))
testset    = Subset(full_test,  range(20))
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader  = DataLoader(testset,  batch_size=32)

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
    _profile = os.environ.get('CLIENT_PROFILE', '')
    if _profile in ('orin', 'laptop', 'macbook'):
        import subprocess
        _tc = {'orin': ('50mbit', '20ms', '8ms'), 'laptop': ('150mbit', '10ms', '3ms'), 'macbook': ('300mbit', '5ms', '1ms')}
        _r, _d, _j = _tc[_profile]
        subprocess.run(['tc', 'qdisc', 'replace', 'dev', 'eth0', 'root', 'netem', 'rate', _r, 'delay', _d, _j], check=False)
    time.sleep(8)
    _mist = os.environ.get('MIST_ENABLED', '').lower() == 'true'
    _host = os.environ.get('FL_SERVER_ADDR', 'resnet_server')
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
    fl.client.start_client(server_address=_addr, client=ResNetClient().to_client())
