import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(in_channels, hidden_channels),
                          nn.BatchNorm1d(hidden_channels), nn.ReLU(),
                          nn.Linear(hidden_channels, hidden_channels), nn.ReLU()))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                          nn.BatchNorm1d(hidden_channels), nn.ReLU(),
                          nn.Linear(hidden_channels, hidden_channels), nn.ReLU()))
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                          nn.BatchNorm1d(hidden_channels), nn.ReLU(),
                          nn.Linear(hidden_channels, hidden_channels), nn.ReLU()))
        self.lin1 = nn.Linear(hidden_channels*3, hidden_channels*3)
        self.lin2 = nn.Linear(hidden_channels*3, out_channels)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        h = torch.cat((h1, h2, h3), dim=1)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h

device = torch.device('cpu')
model = GIN(dataset.num_features, dataset.num_classes, hidden_channels=64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_test_acc = 0

for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    best_test_acc = max(best_test_acc, test_acc)
print(f"Best test acc: {best_test_acc}")
