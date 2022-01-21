import os
import os.path as osp
import datetime
import csv
from tqdm import tqdm

import pandas as pd

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url


class LargeGraphDataset(Dataset):
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    def __init__(self, root="./data/wsdm-2022"):
        print("__init__ (Dataset), root:", root)
        self.root = root  # DATA_DIR
        # self.processed_paths = f"{self.root}/processed"

        super().__init__(root)

        # DATA_DIR = "../data/wsdm-2022"

    @property
    def processed_file_names(self):
        # print("--------------------")
        # print("processed_file_names")
        # print("--------------------")

        date_start = datetime.datetime.strptime("20141019", "%Y%m%d")
        date_end = datetime.datetime.strptime("20170514", "%Y%m%d")
        file_name_list = []
        for index, date in enumerate(pd.date_range(date_start, date_end)):
            file_name_list.append(f'graph_{index}')

        # print(f"len: {len(file_name_list)}")
        # print(f"file_name_list[0]: {file_name_list[0]}")
        return file_name_list

    def process(self):
        print("---------------")
        print("    process    ")
        print("---------------")
        DATA_DIR = self.root

        # ---------------- edge_type_features.csv ----------------
        edge_type_feature_dict = {}
        with open(f"{DATA_DIR}/raw/train/edge_type_features.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = list(map(lambda x: int(x), row))
                edge_type, features = row[0], row[1:]
                edge_type_feature_dict[edge_type] = features

        # ---------------- node_features.csv ----------------

        node_feature_dict = {}
        with open(f"{DATA_DIR}/raw/train/node_features.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = list(map(lambda x: int(x), row))
                node_id, features = row[0], row[1:]
                features = list(map(lambda x: x if x != -1 else 0, features))
                node_feature_dict[node_id] = features

        # ---------------- edges_train_A.csv ----------------

        edge_list_df = pd.read_csv(
            f"{DATA_DIR}/raw/train/edges_train_A.csv", header=None,
            names=['src_id', 'dst_id', 'edge_type', 'timestamp'],
            dtype={'src_id': int, 'dst_id': int,
                   'edge_type': int, 'timestamp': int},
        ).sort_values('timestamp')
        edge_list_df['date'] = edge_list_df['timestamp'].apply(
            lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d")
        )
        grouped_df = edge_list_df.groupby('date')

        file_name_list = []
        for index, (date, group) in enumerate(grouped_df):
            group = group.reset_index(drop=True)

            source_nodes = group['src_id']
            target_nodes = group['dst_id']
            timestamp = group['timestamp']
            edge_type = group['edge_type']

            edge_index = torch.tensor(
                [source_nodes, target_nodes], dtype=torch.long
            )

            # TODO: node features
            # TODO: edge features
            # edge_type_features[edge_type_features['edge_type']==0].values.tolist()[0]

            edge_attrs = torch.tensor([timestamp, edge_type], dtype=torch.long)
            edge_attrs = edge_attrs.transpose(0, 1)

            data = Data(edge_index=edge_index, edge_attrs=edge_attrs)
            torch.save(data, f"{self.processed_paths[index]}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        # data = torch.load(f"{self.root}/processed/train/graph_{index}.pt")
        # data = torch.load(osp.join(self.processed_dir, f'graph_{index}.pt'))
        data = torch.load(self.processed_paths[index])
        return data


if __name__ == '__main__':

    print("---------------------")
    print("    DATASET (DEV)    ")
    print("---------------------")

    edge_index = torch.tensor(
        [[0, 1, 1, 2],
        [1, 0, 2, 1]], dtype=torch.long
    )
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)

    dataset = LargeGraphDataset()
    print(dataset[0])
    print(dataset[0].edge_attrs[0])
