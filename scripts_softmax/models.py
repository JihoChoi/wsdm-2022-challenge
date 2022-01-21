
"""
2022-01-00
author: Jiho Choi

Reference
* https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py

"""


import os
import sys
import datetime
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
# from torch_geometric.nn import BatchNorm, BatchNorm1d
from torch.nn import BatchNorm1d

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
        # num_classes = 2
        # self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))

        # self.entity_embedding = nn.Embedding(num_entities, 100)
        # self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        self.node_embedding = nn.Embedding(num_nodes, 32)  # Lookup
        self.bn32 = BatchNorm1d(32)
        self.conv1 = RGCNConv(
            32, 64, num_relations, # num_bases=30
        )
        self.bn64 = BatchNorm1d(64)
        self.bn64_2 = BatchNorm1d(64)
        self.conv2 = RGCNConv(
            64, 64, num_relations, # num_bases=30
        )
        self.emb_rel = Parameter(torch.Tensor(num_relations, 64))
        # self.emb_rel = nn.Linear(num_relations, 64)
        # self.bn1 = nn.BatchNorm1d(1)  # continuous time
        # self.timestamp_emb = Parameter(torch.Tensor())
        # self. emb_ts = nn.Embedding(1, 100)
        self.emb_ts = nn.Linear(1, 2)
        self.bn2 = BatchNorm1d(2)
        self.bn1 = BatchNorm1d(1)
        self.fc_1 = nn.Linear(1, 12)


        self.emb_triplets = nn.Linear(64 * 3, 64)  # src, link, tgt -> tri
        # self.emb_link = nn.Linear(64 + 2, 16)  # tri + ts -> prob
        self.emb_link = nn.Linear(64 + 12, 16)  # tri + ts -> prob
        self.emb_link2 = nn.Linear(16, 2)  # tri + ts -> prob  # SOFTMAX

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.xavier_uniform_(self.emb_rel.weight)
        nn.init.xavier_uniform_(self.emb_ts.weight)
        nn.init.xavier_uniform_(self.emb_triplets.weight)
        nn.init.xavier_uniform_(self.emb_link.weight)
        nn.init.xavier_uniform_(self.emb_link2.weight)


        # print("self.emb_rel:", self.emb_rel.shape)

    def forward(self, data):

        x = np.unique(data.edge_index)

        # print("data.node_idx:",data.node_idx)
        # print("data.x:", x)
        # print("data.x:", x.shape)
        x = self.node_embedding(data.n_id)
        x = self.bn32(x)

        edge_index = data.edge_index
        # edge_attrs = data.edge_attrs
        edge_types = data.edge_types
        # edge_timestamps = data.edge_timestamps

        edge_types = edge_types[0:edge_index.size(1)]


        x = F.relu(self.conv1(x, edge_index, edge_types))  # [2, 41] -> [869069, 32]
        x = self.bn64(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_types)  # [869069, 32] -> [869069, 64]
        x = self.bn64_2(x)

        # x = F.log_softmax(x, dim=1)
        return x

    def link_embedding(self, node_embeddings, edge_index, edge_type):

        z = node_embeddings
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.emb_rel[edge_type]

        # print("z_src   :", z_src.shape)
        # print("z_dst   :", z_dst.shape)
        # print("rel     :", rel.shape)
        # print("torch.sum(z_src * rel * z_dst, dim=1):", torch.sum(z_src * rel * z_dst, dim=1).shape)
        # print(torch.sum(z_src * rel * z_dst, dim=1)[0:4])

        z_tri = self.emb_triplets(torch.cat((z_src, rel, z_dst), 1))

        return z_tri
        # source, target, edge_type, timestamp

    def temporal_link_prediction(self, z_tri, edge_timestamps):

        # edge_timestamps.apply_(
        #     lambda x: int(datetime.datetime.fromtimestamp(x).strftime("%Y%m%d%H%M"))
        # )


        # print("edge_timestamps   :", edge_timestamps.shape)
        # edge_timestamps = self.bn1(edge_timestamps)
        # print(edge_timestamps)
        edge_timestamps = edge_timestamps.float().unsqueeze(1)  # [41] -> [41, 1]

        edge_timestamps = self.bn1(edge_timestamps)  # TODO: TODO:
        edge_timestamps = self.fc_1(edge_timestamps)
        edge_timestamps = F.relu(edge_timestamps)

        

        """
        edge_timestamps = self.emb_ts(edge_timestamps)
        edge_timestamps = self.bn2(edge_timestamps)
        edge_timestamps = F.relu(edge_timestamps)  # TODO:
        # print("edge_ts :", edge_timestamps.shape)
        # edge_timestamps = edge_timestamps.unsqueeze(1)
        """

        # print("z_tri:", z_tri.shape)
        link_prob = self.emb_link(torch.cat((z_tri, edge_timestamps), 1))
        link_prob = F.relu(link_prob)
        link_prob = F.dropout(link_prob, p=0.5, training=self.training)
        link_prob = self.emb_link2(link_prob)

        # link_likelihood = F.relu(link_likelihood)  # TODO:
        # print("z_tri:", link_likelihood.shape)


        # torch.sum(z_src * rel * z_dst, dim=1)
        # torch.sum(z_src * rel * z_dst, dim=1)  # element-wise product

        # link_prob = torch.softmax(link_prob, dim=1)  # crossentropyloss include
        return link_prob  # [n, 2]


    def calc_loss(self, node_embeddings, samples, target):
        pass
        # source, target, edge_type, timestamp -> y



# model = GNN(hidden_channels=32, out_channels=dataset.num_classes)
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

