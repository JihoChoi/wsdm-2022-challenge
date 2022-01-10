
"""
2022-01-00
author: Jiho Choi

"""


import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
# from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from dataset import LargeGraphDataset
from parse_args import params
from utils import correct_count, load_pickle_file, multi_acc, save_pickle_file

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero


class TemporalGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):

        # self.embedding_lookup = nn.Embedding(
        #     num_embeddings=self.corpus_size,
        #     embedding_dim=64,
        # )


        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

# TODO: sort_edge_index

# model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
# model = to_hetero(model, data.metadata(), aggr='sum')


if __name__ == '__main__':
    """
    USAGE: (env) python3 ./scripts/models.py
    """
    print("--------------------")
    print("    MODELS (DEV)    ")
    print("--------------------")
    dataset = LargeGraphDataset()
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for data in dataloader:
        print(data)
        print("data.edge_index:", data.edge_index[0])
        print("data.edge_attrs:", data.edge_attrs[0])
        break



    exit()

    # 3) MODEL
    device = params['device']
    model = TemporalGNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    data = next(iter(dataloader))
    out = model(data)

    # Overfit on a Single Batch
    for epoch in range(100):
        y_pred = model(data)
        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

    model.eval()
    y_pred = model(x_seq)

    print("-------------------------------------------")
    print(f"accuracy : {multi_acc(y_pred, y_true)}")
    print(f"correct  : {correct_count(y_pred, y_true)} / {32}")
    print("-------------------------------------------")
    print(y_true.tolist())
    print(torch.max(y_pred, dim=1).indices.tolist())
    print("-------------------------------------------")
    print(torch.max(y_pred, dim=1))
    print(y_info)
    # print(y_pred)
    # print(F.softmax(y_pred, dim=1))
