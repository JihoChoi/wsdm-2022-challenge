
"""
2021-12-00
author: Jiho Choi

References
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.data import download_url, InMemoryDataset
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from parse_args import params
from utils import label_stats
from utils import load_pickle_file
from utils import save_pickle_file

import csv


# Reference
# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
# https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8


# class KarateClub(InMemoryDataset):

class LargeGraphDataset(Data):
    """
    data.x          : 노드 특징 행렬 [|N|, |F_node|]
    data.edge_index : 그래프의 연결성 [2, |E|]
    data.edge_attr  : 엣지 특징 행렬 [|E|, |F_edge|]
    data.y          : 학습하고 싶은 대상 (타겟) 노드 레벨 → [|N|, *], 그래프 레벨 → [1, *]
    data.pos        : 노드 위치 행렬 [|N|, num_dimensions]
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

if __name__ == '__main__':

    print("---------------------")
    print("    DATASET (DEV)    ")
    print("---------------------")

    DATA_DIR = "./data/wsdm-2022"
    """
    edge_list = pd.read_csv(
        f"{DATA_DIR}/train/edges_train_A.csv",
        header=None,
        names=['src_id', 'dst_id', 'edge_type', 'timestamp'],
        dtype={'src_id': int, 'dst_id': int, 'edge_type': int, 'timestamp': int},
    ).sort_values('timestamp')

    edge_list['datetime'] = edge_list['timestamp'].copy().apply(
        lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d_%H%M%S")
    )
    # edge_list['month'] = edge_list['timestamp'].apply(
    #     lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m")
    # )
    edge_list['date'] = edge_list['timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d")
    )

    save_pickle_file(f"{DATA_DIR}/train/edges_train_A_cache.pickle", edge_list)
    """
    edge_list_df = load_pickle_file(f"{DATA_DIR}/train/edges_train_A_cache.pickle")

    grouped_df = edge_list_df.groupby('date')

    data_list = []
    for date, group in grouped_df:
        group = group.reset_index(drop=True)
        print(date)
        print(group)

        source_nodes = group['src_id']
        target_nodes = group['dst_id']
        edge_type
        timestamp

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        print(edge_index)
        print(edge_index.shape)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        break


    """
    with open(f"{DATA_DIR}/train/edges_train_A.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for index, row in enumerate(reader):
            src_id, dst_id, edge_type, timestamp = row
            print(row)

            if index > 100:
                break
    """


    # HeteroData
    # collect(key: str) → Dict[Union[str, Tuple[str, str, str]], Any][source]
    # Collects the attribute key from all node and edge types.

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)

    # loader = GraphSAINTRandomWalkSampler(
    #                     data, batch_size=6000,
    #                     walk_length=2, num_steps=5,
    #                     sample_coverage=100,
    #                     save_dir=dataset.processed_dir,
    #                     num_workers=4
    # )


# Reference
# https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.HeteroData

def csv2graph(args):
    if args.dataset == 'A':
        src_type = 'Node'
        dst_type = 'Node'
    elif args.dataset == 'B':
        src_type = 'User'
        dst_type = 'Item'
    else:
        print(' Input parameters error')

    edge_csv = pd.read_csv(
        f'./data/wsdm-2022/raw/train/edges_train_{args.dataset}.csv', header=None)

    heterogenous_group = edge_csv.groupby(2)
    graph_dict = {}
    ts_dict = {}

    for event_type, records in heterogenous_group:
        event_type = str(event_type)
        graph_dict[(src_type, event_type, dst_type)] = (
            records[0].to_numpy(), records[1].to_numpy())
        ts_dict[(src_type, event_type, dst_type)] = (
            torch.FloatTensor(records[3].to_numpy()))
    g = dgl.heterograph(graph_dict)

    g.edata['ts'] = ts_dict

    if args.dataset == 'A':
        # Assign Node feature in to graph
        node_feat_csv = pd.read_csv(
            './data/wsdm-2022/raw/train/node_features.csv', header=None)
        node_feat = node_feat_csv.values[:, 1:]
        node_idx = node_feat_csv.values[:, 0]
        g.nodes[src_type].data['feat'] = torch.zeros(
            (g.number_of_nodes(src_type), 8))
        g.nodes[src_type].data['feat'][node_idx] = torch.FloatTensor(node_feat)

        # Assign Edge Type Feature as the graph`s label, which can be saved along with dgl.heterograph
        etype_feat_csv = pd.read_csv(
            './data/wsdm-2022/raw/train/edge_type_features.csv', header=None)
        etype_feat_tensor = torch.FloatTensor(etype_feat_csv.values[:, 1:])
        etype_feat = {}
        for i, etype in enumerate(g.etypes):
            etype_feat[etype] = etype_feat_tensor[i]

        dgl.save_graphs(
            f"./data/wsdm-2022/DGLgraphs/Dataset_{args.dataset}.bin", g, etype_feat)

    if args.dataset == 'B':
        etype_feat = None
        # Assign Edge Feature
        for event_type, records in heterogenous_group:
            event_type = str(event_type)
            etype = (src_type, event_type, dst_type)
            if len(str(records[4].iloc[0])) > 3:
                g.edges[etype].data['feat'] = (
                    extract_edge_feature(records[4]))
        dgl.save_graphs(f"./data/wsdm-2022/DGLgraphs/Dataset_{args.dataset}.bin", g)
