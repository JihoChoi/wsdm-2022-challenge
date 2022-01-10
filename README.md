

# WSDM Cup 2022


### Keywords
* Link Prediction
* Temporal Graph Neural Networks
* Continuous-Time
* Variational Time Embedding

### Overview
.




### Reference

WSDM 2021 CUP
* https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/
* https://www.dgl.ai/WSDM2022-Challenge/
* https://github.com/dglai/WSDM2022-Challenge

PyTorch Geometric
* https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
* https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py

Link Prediction
* https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/gnn.py
* https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/ctdne-link-prediction.html
* https://paperswithcode.com/task/link-prediction







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

```


### Usage
```bash
# (env)
python ./scripts/prepare_dataset.py


```
















