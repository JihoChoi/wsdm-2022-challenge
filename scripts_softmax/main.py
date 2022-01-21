
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
import pandas as pd
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
from torch_geometric.loader import NeighborSampler, NeighborLoader, HGTLoader

from sklearn.metrics import roc_auc_score


from torch.utils.tensorboard import SummaryWriter


from dataset import LargeGraphDataset
from models import TemporalGNN
from parse_args import params
from utils import correct_count, load_pickle_file, multi_acc, save_pickle_file
from utils import ensure_directory


writer = SummaryWriter()

print(params, end='\n\n')
device = params['device']
device = 'cpu'
# device = torch.device('cuda:1')


dataset_name = 'B'


def save_prob(exist_prob, start_time, epoch=None):
    results_dir = f'results/wsdm-2022/B/{start_time}/'
    ensure_directory(results_dir)
    with open(f"{results_dir}output_B_{epoch:03d}.csv", "a") as file_object:
        for prob in exist_prob.tolist():
            # file_object.write(str(prob) + ',\n')
            if prob >= 0.5:
                file_object.write(str(1) + ',\n')
            else:
                file_object.write(str(0) + ',\n')

if __name__ == '__main__':
    """
    USAGE: (env) python3 ./scripts/main.py
    """
    start_datetime = datetime.datetime.now()

    dataset = LargeGraphDataset(dataset_name=dataset_name)
    dataset = dataset[0]  # PyG Data()

    dataset.n_id = torch.arange(dataset.num_nodes)

    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    num_nodes = dataset.num_nodes
    num_relations = dataset.num_relations
    model = TemporalGNN(
        # num_nodes: 869068, num_relations: 14
        num_nodes=num_nodes,  # B) sample_dataset.num_nodes
        num_relations=num_relations,  # B) sample_dataset.num_relations
    ).to(device)

    # criterion = nn.CrossEntropyLoss()
    # criterion, sigmoid = nn.BCELoss(), nn.Sigmoid()
    # loss_bce = nn.BCEWithLogitsLoss()  # include sigmoid
    # loss_bce = nn.BCELoss()  # no sigmoid
    loss_ce = nn.CrossEntropyLoss(reduce=None, reduction='mean')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001,
        # weight_decay=5e-4,
    )

    # --------------------------------
    # Train & Validation
    # --------------------------------
    model.train()
    optimizer.zero_grad()

    num_epoch = 20
    # print(np.unique(dataset.edge_index)[0:4])
    # print(len(np.unique(dataset.edge_index)))
    for epoch in range(num_epoch):

        # train_loader = NeighborSampler(
        #     dataset,
        #     sizes=[-1],  # all TODO:
        #     # shuffle=True,
        #     input_nodes=None,  # all
        #     batch_size=4,
        #     return_e_id=True,
        # )
        train_loader = NeighborLoader(
            dataset,
            # num_neighbors=[-1],
            # num_neighbors=[30] * 2,
            # num_neighbors=[30] * 1,
            num_neighbors=[-1] * 1,
            shuffle=True,
            input_nodes=None,
            batch_size=1024*16
            # batch_size=1024*32
            # batch_size=1024*64
        )
        # batch = next(iter(train_loader))
        # print(dir(batch))
        # print("x", batch.x)
        # print("y", batch.y)
        # print("node_stores", batch.node_stores)

        # print(batch.n_id)
        # print("len(train_loader):", len(train_loader))

        for index, data in enumerate(train_loader):
            # data = dataset
            # print(data)
            # print("data.edge_index:", data.edge_index)
            # print("data.n_id:", data.n_id)
            optimizer.zero_grad()

            # print("data:", data)
            data = data.to(device)
            node_embeddings = model(data)
            # print("node_embeddings:", node_embeddings.shape)

            z_tri = model.link_embedding(
                node_embeddings,
                data.edge_index,
                data.edge_types,
            )

            # print("---- positive ------------------------------")

            pos_exist_prob = model.temporal_link_prediction(
                z_tri,
                data.edge_timestamps,
            )
            # print("z_tri:", z_tri.shape)
            # print("pos_exist_prob:", pos_exist_prob.shape)
            pos_label = torch.ones_like(data.edge_timestamps)

            # print("---- negative ------------------------------")

            # Neg 1
            ts = data.edge_timestamps
            ts_shuffle = ts[torch.randperm(ts.shape[0])]
            neg_label1 = torch.zeros_like(ts)
            neg_label1[ts_shuffle >= ts] = 1
            # print("ts         :" ts.shape, ts)
            # print("ts_shuffle :", ts_shuffle.shape, ts_shuffle)
            # print("neg_label  :", neg_label.shape, neg_label)
            # print("neg_label  :", neg_label.shape, neg_label)

            neg_exist_prob = model.temporal_link_prediction(
                z_tri,
                ts_shuffle,
            )
            # print("neg_exist_prob:", neg_exist_prob.shape)
            # print("neg_exist_prob:", neg_exist_prob)

            # 2) Neg 2

            ts_before = data.edge_timestamps - 24 * 3600 * 4 # (-4 day)
            neg_label2 = torch.zeros_like(ts_before)
            neg_exist_prob2 = model.temporal_link_prediction(
                z_tri,
                ts_before,
            )
            # 3) Neg 3
            ts_after = data.edge_timestamps + 24 * 3600 * 4 # (+4 day)
            neg_label3 = torch.zeros_like(ts_after)
            neg_exist_prob3 = model.temporal_link_prediction(
                z_tri,
                ts_after,
            )

            # print("pos_label  :", pos_label[0:8])
            # print("neg_label1 :", neg_label1[0:8])
            # print("neg_label2 :", neg_label2[0:8])
            # print("neg_label3 :", neg_label3[0:8])

            # print("ts         :", ts[0:8])
            # print("ts_shuffle :", ts_shuffle[0:8])
            # print("ts_before  :", ts_before[0:8])
            # print("ts_after   :", ts_after[0:8])

            # print("pos_exist_prob  :", pos_exist_prob[0:8])  # 1
            # print("neg_exist_prob  :", neg_exist_prob[0:8])  # 0
            # print("neg_exist_prob2 :", neg_exist_prob2[0:8])
            # print("neg_exist_prob3 :", neg_exist_prob3[0:8])



            # print("---- loss ----------------------------------")

            # pos_exist_prob = pos_exist_prob.squeeze()
            # neg_exist_prob = neg_exist_prob.squeeze()
            # print(pos_exist_prob)
            # print(neg_exist_prob)
            prob = torch.cat([pos_exist_prob, neg_exist_prob, neg_exist_prob2, neg_exist_prob3], 0).squeeze()

            # print("prob:", prob.shape, prob)
            label = torch.cat([pos_label, neg_label1, neg_label2, neg_label3], 0)

            # print("prob:", prob)
            # print("prob:", prob)
            # print("label:", label)
            one_hot_label = nn.functional.one_hot(label).float()
            # print("one_hot:", one_hot)


            # print("prob", prob.shape, prob)
            # print("label", label.shape, label)

            # print(data.edge_types)
            # print(np.unique(data.edge_types))
            # print(len(np.unique(data.edge_types)))
            # loss = loss_bce(prob, label) / len(np.unique(data.edge_types))
            # loss = loss_bce(prob, label)
            loss = loss_ce(prob, one_hot_label)


            loss.backward()
            optimizer.step()

            # pred_label = torch.sigmoid(prob)
            # pred_label = (prob > 0.5).float()
            pred_label = torch.argmax(prob, dim=1)
            # print("pred_label :", pred_label)
            # print("label      :", label)
            correct = (pred_label == label).float().sum()

            # exit()
            print(
                f"Batch[{index}/{len(train_loader)}] loss: {loss.item():.5f} "
                f"Acc: {int(correct.item())}/{len(prob)} = {correct.item() / len(prob)}"
            )
            writer.add_scalar(
                'loss (training)',
                loss.item(),
                epoch * len(train_loader) + index
            )
            # break  # TODO: TODO: TODO: TODO:

        print(f"Epoch[{epoch}]: loss: {loss.item():.5f}")

        # if index >= 1:
        #     break

        # link_likelihood = F.relu(link_likelihood)  # TODO:

        print(
            f"Epoch [{epoch}/{num_epoch}] : "
            f"Loss: {loss.item():.4f}, "
            f"\n"
        )

        # TODO: SAVE
        # torch.save({
        #     'state_dict': model.state_dict(), 'epoch': epoch
        # }, f'./checkpoints/model_{dataset_name}_{epoch:03d}.pth')

        # --------------------------------
        # validate
        # --------------------------------
        # model.eval()
        model.eval()
        with torch.no_grad():
            test_csv = pd.read_csv(
                f"data/wsdm-2022/raw/test/input_B_initial.csv",
                names=['src', 'dst', 'type', 'start_at', 'end_at', 'exist']
            )
            # Load test_csv
            label = test_csv.exist.values
            # start_at = torch.tensor(test_csv.start_at.values)
            # end_at = torch.tensor(test_csv.end_at.values)
            edge_types = torch.tensor(test_csv.type.values)
            start_ts = torch.tensor(test_csv.start_at.values)
            end_ts = torch.tensor(test_csv.end_at.values)
            edge_index = torch.tensor(
                np.array([test_csv.src.values, test_csv.dst.values]), dtype=torch.long
            )
            # dataset.n_id = torch.arange(dataset.num_nodes)
            # print("node_embeddings:", node_embeddings.shape)
            data = Data(
                # x=x,  # node attr
                edge_index=edge_index,
                # edge_attrs=edge_attrs,
                edge_types=edge_types,
                # edge_timestamps=edge_timestamps,
                # name="DatasetB",
                num_nodes=num_nodes,
                # num_relations=num_relations,
            )
            data.n_id = torch.arange(num_nodes)

            node_embeddings = model(data)
            z_tri = model.link_embedding(
                node_embeddings,  # load
                edge_index,
                edge_types,
            )
            pred_prob_end = model.temporal_link_prediction(
                z_tri, end_ts).squeeze()
            pred_prob_start = model.temporal_link_prediction(
                z_tri, start_ts).squeeze()
            # print("pred_prob_end:", pred_prob_end[0:10])
            # print("pred_prob_start:", pred_prob_start[0:10])
            # pred_prob_end = torch.sigmoid(pred_prob_end)  # TODO: HERE
            # pred_prob_start = torch.sigmoid(pred_prob_start)  # TODO: HERE
            exist_prob = pred_prob_end - pred_prob_start

            # exist_prob = torch.sigmoid(exist_prob)  # TODO: HERE
            # pred_label = (exist_prob > 0.5).float()


            # pred_label = (exist_prob > 0.0  ).float()
            # print("pred_label          :", pred_label.shape)
            # print("torch.tensor(label) :", torch.tensor(label).shape)
            # correct = (pred_label == label).sum()

            exist_prob = F.softmax(exist_prob, dim=1)
            pred_label = torch.argmax(exist_prob, dim=1)

            # print(pred_label[0:10])
            # print(torch.tensor(label)[0:10])
            # print((pred_label == torch.tensor(label))[0:10])
            correct = (pred_label == torch.tensor(label)).sum()

            # correct = (pred_label == label).float().sum()
            # correct = np.array((pred_label == torch.tensor(label))).sum()

            # print("val:", pred_label)
            # print("np.unique(label, return_counts=True):", np.unique(label, return_counts=True))
            # AUC = roc_auc_score(label, exist_prob)
            AUC = roc_auc_score(label, pred_label)
            print(f'\nAUC is {round(AUC, 5)} '
                  f"Acc: {int(correct.item())}/{len(exist_prob)} = {correct.item() / len(exist_prob)}\n"
                  )

            # TODO: Write results for test set

            # checkpoint = torch.load(
            #     f'./checkpoints/model_{dataset_name}_{epoch:03d}.pth'
            # )
            # model.load_state_dict(checkpoint['state_dict'])

        # --------------------------------
        # TEST
        # --------------------------------
        # model.eval()
        model.eval()
        with torch.no_grad():
            test_csv = pd.read_csv(
                f"data/wsdm-2022/raw/test/input_B.csv",
                names=['src', 'dst', 'type', 'start_at', 'end_at']
            )
            # Load test_csv
            # start_at = torch.tensor(test_csv.start_at.values)
            # end_at = torch.tensor(test_csv.end_at.values)
            edge_types = torch.tensor(test_csv.type.values)
            start_ts = torch.tensor(test_csv.start_at.values)
            end_ts = torch.tensor(test_csv.end_at.values)
            edge_index = torch.tensor(
                np.array([test_csv.src.values, test_csv.dst.values]), dtype=torch.long
            )
            # dataset.n_id = torch.arange(dataset.num_nodes)
            # print("node_embeddings:", node_embeddings.shape)
            data = Data(
                # x=x,  # node attr
                edge_index=edge_index,
                # edge_attrs=edge_attrs,
                edge_types=edge_types,
                # edge_timestamps=edge_timestamps,
                # name="DatasetB",
                num_nodes=num_nodes,
                # num_relations=num_relations,
            )
            data.n_id = torch.arange(num_nodes)

            node_embeddings = model(data)
            z_tri = model.link_embedding(
                node_embeddings,  # load
                edge_index,
                edge_types,
            )
            pred_prob_end = model.temporal_link_prediction(
                z_tri, end_ts).squeeze()
            pred_prob_start = model.temporal_link_prediction(
                z_tri, start_ts).squeeze()
            # pred_prob_end = torch.sigmoid(pred_prob_end)  # TODO: HERE
            # pred_prob_start = torch.sigmoid(pred_prob_start)  # TODO: HERE
            exist_prob = pred_prob_end - pred_prob_start
            exist_prob = torch.softmax(exist_prob, dim=1)  # TODO: HERE
            # exist_prob = torch.clamp(exist_prob, min=0, max=1)

            # print("test:", exist_prob)
            exist_prob = torch.argmax(exist_prob, dim=1)
            # print("test:", exist_prob)


            save_prob(exist_prob, start_datetime.strftime(
                '%Y%m%d_%H%M%S'), epoch)

            # TODO: Write results for test set

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
