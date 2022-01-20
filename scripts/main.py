
"""
2022-01-00
author: Jiho Choi
Reference
- https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.GraphSAINTSampler
- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py
- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py

"""


import os
import sys
import platform
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import Adam

from torch.nn import Parameter
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import batched_negative_sampling
from torch_geometric.loader import NeighborLoader

from torch.utils.tensorboard import SummaryWriter


from dataset import LargeGraphDataset
from models import TemporalGNN
from parse_args import params
from utils import correct_count, load_pickle_file, multi_acc, save_pickle_file


writer = SummaryWriter()

print(params, end='\n\n')
device = params['device']

dataset_name = 'B'


if __name__ == '__main__':
    """
    USAGE: (env) python3 ./scripts/main.py
    """
    start_datetime = datetime.datetime.now()

    dataset = LargeGraphDataset(dataset_name=dataset_name)
    dataset = dataset[0]  # PyG Data()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TemporalGNN(
        # num_nodes: 869068, num_relations: 14
        num_nodes=869068 + 1,  # B) sample_dataset.num_nodes
        num_relations=14 + 1,  # B) sample_dataset.num_relations
    )

    # criterion = nn.CrossEntropyLoss()
    criterion, sigmoid = nn.BCELoss(), nn.Sigmoid()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # --------------------------------
    # Train & Validation
    # --------------------------------
    num_epoch = 5
    for epoch in range(num_epoch):
        loader = NeighborLoader(
            dataset,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            batch_size=4,
            # input_nodes=data.train_mask,
        )
        for index, data in enumerate(loader):
            print("data:", data)
            node_embeddings = model(data)
            print("node_embeddings:", node_embeddings.shape)

            pos_out = model.link_prediction(
                node_embeddings,
                data.edge_index,
                data.edge_types,
            )

            # TODO: negative sampling

            if index >= 1:
                break

            # loss.backward()
            # optimizer.step()

        print(
            f"Epoch [{epoch}/{num_epoch}] : " \
            # f"Loss: {loss.item():.4f}, " \
            f"\n"
        )

        # TODO: SAVE
        torch.save({
            'state_dict': model.state_dict(), 'epoch': epoch
        }, f'./checkpoints/model_{dataset_name}_{epoch:03d}.pth')

        break

    # --------------------------------
    # Test (Prediction)
    # --------------------------------
    # model.eval()
    with torch.no_grad():
        pass


        # checkpoint = torch.load(
        #     f'./checkpoints/model_{dataset_name}_{epoch:03d}.pth'
        # )
        # model.load_state_dict(checkpoint['state_dict'])


    end_datetime = datetime.datetime.now()
    total_time = round((end_datetime - start_datetime).total_seconds(), 3)

    print("\n")
    print(f"start_datetime  : {start_datetime}")
    print(f"end_datetime    : {end_datetime}")
    print(f"Elapsed Time    : {total_time} seconds")
