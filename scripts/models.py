
"""
2022-01-00
author: Jiho Choi

Reference
* https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py

"""


import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from dataset import LargeGraphDataset
from parse_args import params
from utils import correct_count, load_pickle_file, multi_acc, save_pickle_file

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import batched_negative_sampling


# HEATConv
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn.py


class TemporalGNN(torch.nn.Module):

    def __init__(self, num_nodes, num_relations):
        super().__init__()

        # -----------------
        #     Dataset B
        # -----------------
        num_classes = 2
        # self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))

        # self.entity_embedding = nn.Embedding(num_entities, 100)
        # self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        self.conv1 = RGCNConv(
            num_nodes, 64, num_relations, num_bases=30
        )
        self.conv2 = RGCNConv(
            64, 32, num_relations, num_bases=30
        )
        self.rel_emb = Parameter(torch.Tensor(num_relations, 32))
        print("self.rel_emb:", self.rel_emb.shape)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        # edge_attrs = data.edge_attrs
        edge_types = data.edge_types
        edge_timestamps = data.edge_timestamps

        edge_types = edge_types[0:edge_index.size(1)]

        x = F.relu(self.conv1(None, edge_index, edge_types))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_types)

        # x = F.log_softmax(x, dim=1)
        return x



    def emb_concat(self, g, etype):
        pass
        # source, target, edge_type, timestamp


    def link_prediction(self, node_embeddings, edge_index, edge_type):
        z = node_embeddings
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)  # Element-wise Product


    def calc_loss(self, node_embeddings, samples, target):
        pass
        # source, target, edge_type, timestamp -> y



# model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
# model = to_hetero(model, data.metadata(), aggr='sum')

if __name__ == '__main__':
    """
    USAGE: (env) python3 ./scripts/models.py
    """
    print("--------------------")
    print("    MODELS (DEV)    ")
    print("--------------------")
    dataset = LargeGraphDataset(dataset_name='B')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    data = next(iter(dataloader))

    model = TemporalGNN(
        # num_nodes: 869068, num_relations: 14
        num_nodes=869068 + 1,  # B) sample_dataset.num_nodes
        num_relations=14 + 1,  # B) sample_dataset.num_relations
    )
    # model = model  #.to(params['device'])

    y_pred = model(data)

    exit()
    for data in dataloader:
        print("-------------------------------")
        print("data:", data)
        print("-------------------------------\n")
        y_pred = model(data)
        # y_pred = model(data)
        # loss = criterion(y_pred, y_true)
        # print(data)
        # print("data.edge_index:", data.edge_index.shape)
        # print("data.edge_attrs:", data.edge_attrs.shape)
        break

        loss.backward()
        optimizer.step()
        pass

    exit()

    # 3) MODEL
    device = params['device']
    model = TemporalGNN(

    ).to(device)
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
