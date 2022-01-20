
"""
2022-01-00
author: Jiho Choi

Reference
- https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.GraphSAINTSampler
- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py

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
from models import TemporalGNN
from parse_args import params
from utils import correct_count, load_pickle_file, multi_acc, save_pickle_file

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import batched_negative_sampling


from torch_geometric.loader import NeighborLoader


if __name__ == '__main__':
    """
    USAGE: (env) python3 ./scripts/loader.py
    """
    print("--------------------")
    print("    loader (DEV)    ")
    print("--------------------")

    data = LargeGraphDataset(dataset_name='B')
    data = data[0]
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    loader = NeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=4,
        # input_nodes=data.train_mask,
    )

    model = TemporalGNN(
        # num_nodes: 869068, num_relations: 14
        num_nodes=869068,  # B) sample_dataset.num_nodes
        num_relations=14,  # B) sample_dataset.num_relations
    )

    print("len(loader):", len(loader))  # 224813

    for index, sampled_data in enumerate(loader):
        print(sampled_data)
        print("sampled_data.edge_index:", \
            sampled_data.edge_index.shape, sampled_data.edge_index)
        print("sampled_data.edge_types:", \
            sampled_data.edge_types.shape, sampled_data.edge_types)
        print("sampled_data.edge_timestamps:", \
            sampled_data.edge_timestamps.shape, sampled_data.edge_timestamps)

        y_pred = model(data)

        if index >= 1:
            break
