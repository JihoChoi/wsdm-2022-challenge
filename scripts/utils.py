
# 2021-12-00
# author: Jiho Choi


import sys
import os
import platform
import time
import datetime
import argparse
import json
import pickle
import yaml  # pip3 install PyYaml

import numpy as np
import torch

# from pytz import timezone


# -----------------------------
#         Project Setup
# -----------------------------


import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models


def random_seed(random_seed=random.randint(1, 1000)):
    # Python, NumPy
    random.seed(random_seed)
    np.random.seed(random_seed)
    # PyTorch, CuDNN
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # tf.random.set_seed(seed)  # IF TF
    print("RANDOM SEED:", random_seed)



# -----------------------
#         PyTorch
# -----------------------

def label_stats(dataset, label_count=2):
    labels = [0] * label_count

    if isinstance(dataset[0], tuple):  # TODO: DEPRECATED
        for data in dataset:
            labels[data[0]] += 1
    if isinstance(dataset[0], dict):
        for data in dataset:
            labels[data['label']] += 1

    print(labels)
    percents = [x / sum(labels) for x in labels]
    return labels, percents


def multi_acc(y_pred, y_test):
    # https://discuss.pytorch.org/t/crossentropyloss-for-sequences-loss-and-accuracy-calculation/122965/2
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    # acc = torch.round(acc * 100)
    acc = round(acc.item() * 100, 2)
    return acc


def correct_count(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    return correct_pred.sum().item()


# -----------------------
#         FILE IO
# -----------------------

def ensure_directory(path):
    # path: file path (./dir/file.ext) or dir path (./dir/)
    path = os.path.split(path)
    if not os.path.exists(path[0]):
        os.makedirs(path[0])


def path_join(*elem):
    return os.path.join(*elem)


def load_yaml_file(path):
    with open(path, "r", encoding='UTF8') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return data


def save_json_file(path, dict_data):
    ensure_directory(path)
    with open(path, "w", encoding="UTF-8") as json_file:
        json.dump(dict_data, json_file, ensure_ascii=False)


def load_json_file(path):
    with open(path, "r") as json_file:
        data = json.loads(json_file.read())
    return data


def save_pickle_file(path, data):
    ensure_directory(path)
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle_file(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def read_sql_file(path):
    with open(path, "r", encoding='UTF8') as sql_file:
        data = sql_file.read()
    return data


def write_data(path, data):
    ensure_directory(path)
    with open(path, "w") as sql_file:
        sql_file.write(data)
    return data


# ---------------------------
#         DATE / TIME
# ---------------------------

def datetime_to_date_str(regdatetime):
    assert(type(regdatetime) == datetime.datetime)
    if not regdatetime:
        return None
    date_str = regdatetime.strftime("%Y%m%d")
    return date_str


def previous_date_str(date_str):
    datetime_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
    datetime_obj = datetime_obj - datetime.timedelta(days=1)
    date_str = datetime_obj.strftime("%Y%m%d")
    return date_str


# ---------------------------
#         LOG / PRINT
# ---------------------------

def print_dict(dict_data, name=None, indent=0):
    print(name) if name else None
    print(dict_to_str(dict_data, indent=indent))
    print("")


def dict_to_str(dict_data, indent=0, prefix="", end='\n'):
    dict_string = ''
    for key in dict_data:
        dict_string += '\t' * indent + f"{prefix}{key}: {dict_data[key]}{end}"
    return dict_string


if __name__ == '__main__':
    print("TEST: utils.py")
    random_seed(2022)
    print(random.randint(1, 100), random.randint(1, 100))

    pass
