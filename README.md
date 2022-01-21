# WSDM Cup 2022

### Keywords

- Link Prediction
- Temporal Graph Neural Networks
- Continuous-Time
- Variational Time Embedding

### Overview

.

### Idea

- Continuous Time Embedding
    - Temporal metapath2vec
    - variational time embedding
- Continuous Time Heterogeneous Link Prediction with Relational Graph Attention Networks

### Reference

WSDM 2021 CUP

- https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/
- https://www.dgl.ai/WSDM2022-Challenge/
- https://github.com/dglai/WSDM2022-Challenge

PyTorch Geometric

- https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py

Link Prediction

- https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/gnn.py
- https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/ctdne-link-prediction.html
- https://paperswithcode.com/task/link-prediction

Heterogeneous / MetaPath2Vec

- https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
- https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.MetaPath2Vec
- Heterogeneous Graph Attention Network (WWW 2019)
- HGCN: A Heterogeneous Graph Convolutional Network-Based Deep Learning Model Toward Collective Classification (KDD 2020)
- R-GAT
    - Relational Graph Attention Networks
    - r-GAT: Relational Graph Attention Network for Multi-Relational Graphs
- R-GCN
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py
    - https://github.com/JinheonBaek/RGCN/blob/master/models.py
- Negative Sampling
    - Robust Negative Sampling for Network Embedding

### Setup

```bash

$ sudo apt-get install python3-venv
$ cd ./[repo]
$ python3 -m venv env
$ source ./env/bin/activate

# (env)
pip install --upgrade pip

# -----------------------------------~~~~
#     PyTorch / PyTorch Geometric
# -----------------------------------
nvidia-smi
nvcc --version
# TORCH 1.10.0 + CUDA 113

# https://pytorch.org/get-started/locally/
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# -----------------------------------
#     Machine Learning
# -----------------------------------
pip install matplotlib
pip install seaborn
pip install numpy  # for torch
pip install scipy  # for torch-sparse
pip install sklearn
pip install jupyterlab
pip install networkx
pip install pyvis

# pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

```

### Usage

```bash
# (env)
python ./scripts/prepare_dataset.py


```

### References

- [negative edges](https://discuss.pytorch.org/t/imbalanced-positive-negative-edges-graph-link-prediction/84032/3)
- [continuous time embedding](https://dl.acm.org/doi/fullHtml/10.1145/3184558.3191526)
-


### Note
* The test set will not involve any nodes that do not already appear in the given edge set. -> Transductive