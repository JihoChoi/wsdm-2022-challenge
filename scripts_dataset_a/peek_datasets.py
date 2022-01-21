



import sys
import os
import datetime

import numpy as np
import pandas as pd
import networkx as nx

"""
A
27045268
len(edge_list['edge_type'].unique()): 248
len(edge_list['date'].unique()): 939
20141019_050000 ~ 20170514_050000

B
8278431
len(edge_list['edge_type'].unique()): 14
len(edge_list['date'].unique()): 274
20150101_100012 ~ 20151001_005959
"""


DATA_DIR = "./data/wsdm-2022/raw"

print("FILES")
print(sorted(os.listdir(f"{DATA_DIR}/train")))
print(sorted(os.listdir(f"{DATA_DIR}/test")))

# dataset_name = 'A'
dataset_name = 'B'

print(f"-----------------------")
print(f"Dataset: {dataset_name}")
print(f"-----------------------")

edge_list = pd.read_csv(
    f"{DATA_DIR}/train/edges_train_{dataset_name}.csv",
    header=None,
    names=['src_id', 'dst_id', 'edge_type', 'timestamp', 'etc'],
    dtype={'src_id': str, 'dst_id': str, 'edge_type': str, 'timestamp': int, 'etc': str},
).sort_values('timestamp')
print(f"edge_list: {sys.getsizeof(edge_list)} bytes")

print(f"edge_count: {len(edge_list)}")
print(edge_list.head())

timestamps = edge_list['timestamp'].copy()
datetimes = timestamps.apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d_%H%M%S")
)
dates = timestamps.apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y%m%d")
)
edge_list['datetime'] = datetimes
edge_list['date'] = dates

print("edge_list['edge_type'].unique():", edge_list['edge_type'].unique())
print("len(edge_list['edge_type'].unique()):", len(edge_list['edge_type'].unique()))
print("len(edge_list['date'].unique()):", len(edge_list['date'].unique()))
ts_min = edge_list['timestamp'].min()
ts_max = edge_list['timestamp'].max()
print(
    datetime.datetime.fromtimestamp(ts_min).strftime("%Y%m%d_%H%M%S"),
    '~', datetime.datetime.fromtimestamp(ts_max).strftime("%Y%m%d_%H%M%S")
)

print(f"source_id: {edge_list['src_id'].min()}, {edge_list['src_id'].max()}")
print(f"destination_id: {edge_list['dst_id'].min()}, {edge_list['dst_id'].max()}")

print(f"edge_list: {sys.getsizeof(edge_list)} bytes")
print(f"edge_list: {sys.getsizeof(edge_list) // (10**6)} MB")
print(f"edge_list: {sys.getsizeof(edge_list) // (10**9)} GB")
