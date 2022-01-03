
"""
2022-01-00
author: Jiho Choi

"""


import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from dataset import GraphDataset
from parse_args import params
from utils import correct_count, load_pickle_file, multi_acc, save_pickle_file

# self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)


class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()

        self.num_classes = 2
        self.num_layers = 2
        self.hidden_size = 64
        self.corpus_size = 90  # TODO: dynamic
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        # Embedding Lookup:
        # Batch x Sequence -> Batch x Sequence x Embedding
        self.embedding_lookup = nn.Embedding(
            num_embeddings=self.corpus_size,
            embedding_dim=64,
        )
        self.gru = nn.GRU(
            64, 64, 2,  # input_size, hidden_size, num_layers
            batch_first=True, bidirectional=self.bidirectional
        )
        # x -> batch_size, sequence_length, feature_dimension

        self.fc = nn.Linear(64 * 2, self.num_classes)  # 2 for bidirection
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding_lookup.weight)
        # if isinstance(self.gru, nn.GRU):
        #     # https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
        #     print("GRU INIT")
        #     for param in self.gru.parameters():
        #         if len(param.shape) >= 2:
        #             init.orthogonal_(param.data)
        #         else:
        #             init.normal_(param.data)
        init.xavier_normal_(self.fc.weight)

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_size
        ))
        return hidden.to(params['device'])

        # For LSTM
        # cell = Variable(torch.zeros(
        #     self.num_layers * self.num_directions, batch_size, self.hidden_size
        # ))
        # return hidden, cell

    def forward(self, x):
        h0 = self.init_hidden(batch_size=x.size(0))

        # [Batch, Seq] -> [Batch, Seq, Emb]
        x = F.relu(self.embedding_lookup(x))
        x, _ = self.gru(x, h0)                # [2, Seq, 64] -> [2, Seq, 128]
        # [2, 50, 128] -> [2, 128]  # 마지막 hidden
        x = x[:, -1, :]
        out = self.fc(x)                      # [2, 50, 128] -> [2, 2]
        return out


if __name__ == '__main__':
    """
    USAGE: (env) python3 ./scripts/models.py
    """
    print("--------------------")
    print("    MODELS (DEV)    ")
    print("--------------------")

    # # 1) SAVE
    # dataset = GraphDataset(max_len=50)
    # save_pickle_file("./temp/dataset_cache.pickle", dataset)

    # 2) LOAD
    dataset = load_pickle_file("./temp/dataset_cache.pickle")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3) MODEL
    device = params['device']
    model = BiGRU().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    x_feat, x_seq, y_true, y_info = next(iter(dataloader))
    x_seq = x_seq.to(device)
    y_true = y_true.to(device)

    # Overfit on a Single Batch
    for epoch in range(100):
        y_pred = model(x_seq)
        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

    model.eval()
    y_pred = model(x_seq)

    print("-------------------------------------------")
    print(f"accuracy : {multi_acc(y_pred, y_true)}")
    print(f"correct  : {correct_count(y_pred, y_true)} / {32}")
    print("-------------------------------------------")
    print(y_true.tolist())
    print(torch.max(y_pred, dim=1).indices.tolist())
    print("-------------------------------------------")
    print(torch.max(y_pred, dim=1))
    print(y_info)
    # print(y_pred)
    # print(F.softmax(y_pred, dim=1))
