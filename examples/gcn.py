


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# edge_index = torch.tensor(
#     [[0, 1, 1, 2],
#     [1, 0, 2, 1]], dtype=torch.long
# )
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

edge_index = torch.tensor(
    [[3, 1, 1, 2],
    [1, 3, 2, 1]], dtype=torch.long
)
x = torch.tensor([[-1], [0], [1], [-1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
model = GCN()

for epoch in range(20):
    out = model(data)
    print(out)
    break