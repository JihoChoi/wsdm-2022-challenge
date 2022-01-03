

"""

Author: Jiho Choi
References:
    * https://www.dgl.ai/WSDM2022-Challenge/
    * https://github.com/dglai/WSDM2022-Challenge


Datasets
* Train
    * Dataset A
        * edges_train_A.csv
            * src_id -> dst_id, edge_type, timestamp
        * node_features.csv
            * node_id, anonymized categorical features
        * edge_type_features.csv
            * edge_id, anonymized categorical features
    * Dataset B
        * edges_train_B.csv
            * src_id -> dst_id, edge_type, timestamp, feat (anonymized edge features)
* Test
    * Dataset A
        * src_id -> dst_id, edge_type, start_time, end_time
    * Dataset B
        * src_id -> dst_id, edge_type, start_time, end_time

"""


import os
import sys
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser('csv2graph')
    parser.add_argument(
        '--dataset', type=str, choices=['A', 'B'], default='A', help='dataset'
    )
    args = parser.parse_args()
    # parser.print_help()

    # -----------------
    # Download Datasets
    # -----------------

    data_dir = 'data/wsdm-2022'
    host_url = 'https://data.dgl.ai/dataset/WSDMCup2022'

    if not os.path.exists(f'{data_dir}/train'):
        os.system(f'wget -P {data_dir}/train {host_url}/edges_train_A.csv.gz')
        os.system(f'wget -P {data_dir}/train {host_url}/node_features.csv.gz')
        os.system(f'wget -P {data_dir}/train {host_url}/edge_type_features.csv.gz')
        os.system(f'wget -P {data_dir}/train {host_url}/edges_train_B.csv.gz')
        os.system(f'gzip -d {data_dir}/train/*.gz')

    if not os.path.exists('f{data_dir}/test'):
        os.system(f'wget -P {data_dir}/test {host_url}/input_A_initial.csv.gz')
        os.system(f'wget -P {data_dir}/test {host_url}/input_B_initial.csv.gz')
        os.system(f'wget -P {data_dir}/test {host_url}/intermediate/input_A.csv.gz')
        os.system(f'wget -P {data_dir}/test {host_url}/intermediate/input_B.csv.gz')
        os.system(f'gzip -d {data_dir}/test/*.gz')
