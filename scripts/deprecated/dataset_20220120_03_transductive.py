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

from torch_geometric.data import HeteroData
from utils import ensure_directory


# Transductive ()
# "The test set will not involve any nodes that do not already appear in the given edge set"


class LargeGraphDataset():
    def __new__(cls, dataset_name='B', root="./data/wsdm-2022"):
        if dataset_name in ['a', 'A']:
            dataset = LargeGraphDatasetA(root="./data/wsdm-2022")
        elif dataset_name in ['b', 'B']:
            dataset = LargeGraphDatasetB(root="./data/wsdm-2022")
        else:
            print("ERROR: @LargeGraphDataset")
        return dataset

# -----------------------------------------------------------------------------
# B
# -----------------------------------------------------------------------------


class LargeGraphDatasetB(Dataset):
    def __init__(self, root="./data/wsdm-2022"):
        print("(__init__) root:", root)
        self.root = root  # DATA_DIR
        # ensure_directory(f"{self.root}/processed/dataset_a/")
        ensure_directory(f"{self.root}/processed/dataset_b/")
        super().__init__(root)

    @property
    def processed_file_names(self):
        # $ head -n 200 ./data/wsdm-2022/raw/train/edges_train_B.csv
        date_start = datetime.datetime.strptime("20150101", "%Y%m%d")
        date_end = datetime.datetime.strptime("20151001", "%Y%m%d")
        file_name_list = []
        for index, date in enumerate(pd.date_range(date_start, date_end)):
            file_name_list.append(f'dataset_b/graph_{index}.pt')
        return file_name_list

    def process(self):
        def extract_edge_feature(records):
            feat_list = []
            # edge_feature_exist: 552215, edge_feature_not_exist: 7726216
            for record in records:
                if record != 'nan':
                    feat_list.append(record.strip().split(','))
                    # print(len(record.strip().split(',')), end=' ')  # 768
                    # print(record.strip().split(',')[0:5])
                else:
                    # TODO: rand (?)
                    # feat_list.append("NaN")
                    pass
            return torch.FloatTensor(np.array(feat_list).astype('float32'))


        print("------------------")
        print("    process (B)   ")
        print("------------------")

        # ---------------- edges_train_B.csv ----------------
        edge_list_df = pd.read_csv(
            f"{self.root}/raw/train/edges_train_B.csv", header=None,
            names=['src_id', 'dst_id', 'edge_type',
                   'timestamp', 'edge_features'],
            dtype={
                'src_id': int, 'dst_id': int,
                'edge_type': int, 'timestamp': int,
                'edge_features': str,
            },
            # nrows=2200,  # DEV MODE
        ).sort_values('timestamp')
        edge_list_df['edge_features'] = edge_list_df[
            'edge_features'
        ].astype(str)
        edge_list_df['date'] = edge_list_df['timestamp'].apply(
            lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d")
        )
        grouped_df = edge_list_df.groupby('date')

        # Graph Properties
        num_nodes = int(
            max(edge_list_df['src_id'].max(), edge_list_df['dst_id'].max())
        )
        num_relations = int(edge_list_df['edge_type'].max())

        # print(edge_list_df['edge_features'])
        # print(extract_edge_feature(edge_list_df['edge_features']).shape)
        # exit()

        print("num_nodes     :", num_nodes)
        print("num_relations :", num_relations)

        # ---------------- Preprocess ----------------

        # Transductive Setting
        edge_type_series = edge_list_df['edge_type']
        timestamp_series = edge_list_df['timestamp']
        edge_types = torch.tensor(edge_type_series, dtype=torch.long)
        edge_timestamps = torch.tensor(timestamp_series, dtype=torch.long)

        torch.save(
            edge_types,
            f"{self.root}/processed/dataset_b/edge_types.pt"
        )
        torch.save(
            edge_timestamps,
            f"{self.root}/processed/dataset_b/edge_timestamps.pt"
        )

        for index, (date, group) in enumerate(grouped_df):
            group = group.reset_index(drop=True)

            source_nodes = group['src_id']
            target_nodes = group['dst_id']
            # Inductive -> Transductive
            # edge_type_series = group['edge_type']
            # timestamp_series = group['timestamp']
            # edge_features = group['edge_features']

            # ------------ edge_index ------------
            edge_index = torch.tensor(
                np.array([source_nodes, target_nodes]), dtype=torch.long
            )

            # data = HeteroData()  # TODO: Heterogeneous Graph
            data = Data(
                # x=x,  # node attr
                edge_index=edge_index,
                # edge_attrs=edge_attrs,
                # edge_types=edge_types,
                # edge_timestamps=edge_timestamps,
                name="DatasetB",
                num_nodes=num_nodes,
                num_relations=num_relations,
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


# -----------------------------------------------------------------------------
# A
# -----------------------------------------------------------------------------

"""
class LargeGraphDatasetA(Dataset):
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
        # date_start = datetime.datetime.strptime("20150101", "%Y%m%d")
        # date_end = datetime.datetime.strptime("20151001", "%Y%m%d")
        file_name_list = []
        for index, date in enumerate(pd.date_range(date_start, date_end)):
            file_name_list.append(f'dataset_a/graph_{index}')

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

            # ------------ edge_index ------------
            edge_index = torch.tensor(
                np.array([source_nodes, target_nodes]), dtype=torch.long
            )

            # ------------ node_features ------------
            max_index = np.amax((source_nodes, target_nodes))

            # x = torch.LongTensor(node_features)
            x = torch.tensor(node_features, dtype=torch.float)

            # ------------ edge_attrs / edge_labels ------------
            edge_attrs = [
                edge_type_feature_dict[edge_type] for edge_type in edge_type_series
            ]
            edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

            # edge_labels = torch.tensor(
            #     [timestamp_series, edge_type_series], dtype=torch.long
            # ).transpose(0, 1)

            edge_timestamps = torch.tensor(timestamp_series, dtype=torch.long)
            edge_types = torch.tensor(edge_type_series, dtype=torch.long)

            data = Data(
                x=x,  # node attr
                edge_index=edge_index,
                edge_types=edge_types,
                edge_attrs=edge_attrs,
                edge_timestamps=edge_timestamps,
                name="DatasetA",
                num_nodes=num_nodes,
                num_relations=num_relations,
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

    # dataset = LargeGraphDataset(dataset_name=dataset_name)
    # dataset = load_dataset(dataset_name='B')
    dataset = LargeGraphDataset(dataset_name='B')

    print(dataset[0])
    # print(dataset[0].edge_attrs[0])
"""
