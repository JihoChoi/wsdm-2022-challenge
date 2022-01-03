
"""
2021-12-00
author: Jiho Choi
"""

import sys
import os
import platform
import datetime
import argparse
import yaml  # pip3 install PyYaml

import pyarrow as pa
import pyarrow.parquet as pq

from google.cloud import bigquery
from pytz import timezone

# from utils import load_yaml_file


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models import BiGRU
from dataset import GraphDataset

from parse_args import params

from utils import ensure_directory
from utils import save_json_file
from utils import load_pickle_file
from utils import save_pickle_file
from utils import set_gcp_credentials
from utils import multi_acc
from utils import correct_count


print(params, end='\n\n')
device = params['device']


# https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/ctdne-link-prediction.html


if __name__ == '__main__':
    """
    USAGE:
        (env) python3 ./scripts/main.py
    """

    # 1) Prepare Dataset & SAVE
    dataset = GraphDataset(max_len=200)
    save_pickle_file("./temp/dataset_cache.pickle", dataset)

    # 2) LOAD
    dataset = load_pickle_file("./temp/dataset_cache.pickle")

    # TEST
    # train_dataset = [dataset[i] for i in range(0, 1000000)]
    # val_dataset = [dataset[i] for i in range(1000000, 1037414)]
    # train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)

    # DEV
    train_dataset = [dataset[i] for i in range(0, 100)]
    val_dataset = [dataset[i] for i in range(100, 200)]
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # model = BiGRU(input_size, hidden_size, num_layers, num_classes).to(device)
    model = BiGRU().to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion, sigmoid = nn.BCELoss(), nn.Sigmoid()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # --------------------------------
    # Train & Validation
    # --------------------------------

    # TODO: move params
    num_epoch = 5

    for epoch in range(num_epoch):

        # model.train()

        # ------------ Train ------------

        for idx, batch in enumerate(train_dataloader):
            x_feat, x_seq, y_true, y_info = batch
            x_seq = x_seq.to(device)
            y_true = y_true.to(device)

            # Forward
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)  # CEL 는 softmax 포함

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient Descent (adam step)
            optimizer.step()

            if idx % 100 == 0:
                print(
                    f"Train Batch [{idx}/{len(train_dataloader)}], Acc: {multi_acc(y_pred, y_true)}")

        print(f"Epoch [{epoch}/{num_epoch}], Loss: {loss.item():.4f}")

        # ------------ Validation ------------
        correct = 0

        for batch in val_dataloader:
            x_feat, x_seq, y_true, y_info = batch
            x_seq = x_seq.to(device)
            y_true = y_true.to(device)

            # Forward
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)  # CEL 는 softmax 포함

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += correct_count(y_pred, y_true)

            # TODO: Early Stop

            if idx % 10 == 0:
                print(
                    f"Validation Batch [{idx}/{len(val_dataloader)}], Acc: {multi_acc(y_pred, y_true)}")

        # accuracy = 100 * correct / len(val_dataloader)
        result_str = "\n" \
            + f"Epoch [{epoch}/{num_epoch}]\n" \
            + f"Loss: {loss.item():.4f}, " \
            + f"Acc: {correct / len(val_dataset)} ({correct}/{len(val_dataset)}), " \
            + f"\n"

        print(result_str)

    # --------------------------------
    # Test (Prediction)
    # --------------------------------
    # model.eval()
    with torch.no_grad():
        pass

    # TODO: SAVE

    end_datetime = datetime.datetime.now(timezone("Asia/Seoul"))
    total_time = round((end_datetime - start_datetime).total_seconds(), 3)

    print("\n")
    print(f"start_datetime  : {start_datetime}")
    print(f"end_datetime    : {end_datetime}")
    print(f"Elapsed Time    : {total_time} seconds")
