import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

def make_dataset(n, seq_len=50, features=9, classes=6):
    return TensorDataset(
        torch.randn(n, seq_len, features),
        torch.randint(0, classes, (n,))
    )

trainset    = make_dataset(500)   # was 2000
testset     = make_dataset(100)   # was 400
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader  = DataLoader(testset,  batch_size=128)

class GRUModel(nn.Module):
    def __init__(self, input_size=9, hidden=64, layers=1, classes=6):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers,
                          batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(hidden * 2, classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

device    = torch.device("cpu")
model     = GRUModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class RNNClient(fl.client.NumPyClient):
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
    import os, time
    time.sleep(8)
    fl.client.start_client(
        server_address=f"{os.environ.get('FL_SERVER_ADDR', 'rnn_server')}:8080",
        client=RNNClient().to_client(),
    )
