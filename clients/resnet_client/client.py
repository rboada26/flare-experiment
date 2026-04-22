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
    import os, time
    time.sleep(8)
    fl.client.start_client(
        server_address=f"{os.environ.get('FL_SERVER_ADDR','resnet_server')}:8080",
        client=ResNetClient().to_client(),
    )
