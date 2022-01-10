import os
import os.path as osp
import datetime
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
# from torch_geometric.utils import sort_edge_index

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url


class LargeGraphDataset(Dataset):
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    def __init__(self, root="./data/wsdm-2022"):
        print("(__init__) root:", root)
        self.root = root  # DATA_DIR
        # self.processed_paths = f"{self.root}/processed"

        super().__init__(root)

        # DATA_DIR = "../data/wsdm-2022"

    @property
    def processed_file_names(self):
        # print("--------------------")
        # print("processed_file_names")
        # print("--------------------")
        # self.processed_paths, self.processed_file_names

        # TODO: handle empty dates -> list of existing dates
        date_start = datetime.datetime.strptime("20141019", "%Y%m%d")
        date_end = datetime.datetime.strptime("20170514", "%Y%m%d")
        file_name_list = []
        for index, date in enumerate(pd.date_range(date_start, date_end)):
            file_name_list.append(f'graph_{index}')

        """
        TODO: if not set
        date_set = set()
        with open(f"{self.root}/raw/train/edges_train_A.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                date = datetime.datetime.fromtimestamp(int(row[3]))
                date = date.strftime("%Y%m%d")  # %H:%M:%S
                date_set.add(date)
        file_name_list = []
        for index, date in enumerate(sorted(list(date_set))):
            file_name_list.append(f'graph_{index}')
        """

        # print("file_name_list:", len(file_name_list), file_name_list[0])
        return file_name_list

    def process(self):
        print("---------------")
        print("    process    ")
        print("---------------")
        # DATA_DIR = self.root

        # ---------------- edge_type_features.csv ----------------

        edge_type_feature_dict = {}
        with open(f"{self.root}/raw/train/edge_type_features.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = list(map(lambda x: int(x), row))
                edge_type, features = row[0], row[1:]
                edge_type_feature_dict[edge_type] = features

        # ---------------- node_features.csv ----------------

        node_feature_dict = {}
        with open(f"{self.root}/raw/train/node_features.csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = list(map(lambda x: int(x), row))
                node_id, features = row[0], row[1:]
                features = list(map(lambda x: x if x != -1 else 0, features))
                node_feature_dict[node_id] = features

        # ---------------- edges_train_A.csv ----------------

        edge_list_df = pd.read_csv(
            f"{self.root}/raw/train/edges_train_A.csv", header=None,
            names=['src_id', 'dst_id', 'edge_type', 'timestamp'],
            dtype={
                'src_id': int, 'dst_id': int, 'edge_type': int, 'timestamp': int
            },
        ).sort_values('timestamp')
        edge_list_df['date'] = edge_list_df['timestamp'].apply(
            lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d")
        )
        grouped_df = edge_list_df.groupby('date')

        # ---------------- Preprocess ----------------

        for index, (date, group) in enumerate(grouped_df):
            group = group.reset_index(drop=True)

            source_nodes = group['src_id']
            target_nodes = group['dst_id']
            timestamp_series = group['timestamp']
            edge_type_series = group['edge_type']

            node_index_mapper = LabelEncoder()  # sorter: compress edge index
            node_index_mapper.fit(
                pd.concat([source_nodes, target_nodes], axis=0)
            )

            source_nodes = node_index_mapper.transform(source_nodes)
            target_nodes = node_index_mapper.transform(target_nodes)

            # ------------ edge_index ------------
            edge_index = torch.tensor(
                np.array([source_nodes, target_nodes]), dtype=torch.long
            )

            # ------------ node_features ------------
            max_index = np.amax((source_nodes, target_nodes))
            # print("max_index:", max_index)
            node_indexes = node_index_mapper.inverse_transform(
                list(range(0, max_index + 1))
            )
            node_features = [
                node_feature_dict[node_index] for node_index in node_indexes
            ]
            # x = torch.LongTensor(node_features)
            x = torch.tensor(node_features, dtype=torch.long)

            # ------------ edge_attrs / edge_labels ------------
            edge_attrs = [
                edge_type_feature_dict[edge_type] for edge_type in edge_type_series
            ]
            edge_attrs = torch.tensor(edge_attrs, dtype=torch.long)

            edge_labels = torch.tensor(
                [timestamp_series, edge_type_series], dtype=torch.long
            ).transpose(0, 1)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attrs=edge_attrs,
                edge_labels=edge_labels
            )
            print("dataset:", data) if index == 0 else None

            torch.save(data, f"{self.processed_paths[index]}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, index):
        # data = torch.load(f"{self.root}/processed/train/graph_{index}.pt")
        # data = torch.load(osp.join(self.processed_dir, f'data_{index}.pt'))
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
