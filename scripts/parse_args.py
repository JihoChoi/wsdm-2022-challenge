import sys
import os
import platform
import time
import datetime
import argparse

import torch

def parse_arguments():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--train-flag", required=True, type=str2bool)
    args = vars(parser.parse_args())
    return args

# -------------------------------------------------------
# CONFIG / ARGS -> PARAMS
# -------------------------------------------------------

args = {
    # 'RETRIEVE_INPUT_FLAG': True,
}
configs = {
    'file_name': {
        'corpus': f"./data/raw/corpus_data/log_corpus.json",
    },
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
}
# args = parse_arguments()


params = {**configs, **args}