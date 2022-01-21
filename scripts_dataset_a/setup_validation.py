

# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import torch; print(torch.cuda.is_available())"


from subprocess import call
import sys
import torch
import torch_sparse
import torch_scatter
import torch_geometric
from torch_geometric.data import Data

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

x = torch.rand(5, 3)
print(x)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
matrix = [[0, 1, 1, 2], [1, 0, 2, 1]]
edge_index = torch.tensor(matrix, dtype=torch.long)
data = Data(x=x, edge_index=edge_index)
print(data)


print("-------------------------------------------------------")
print('Python version           :', sys.version)
print("Torch version            :", torch.__version__)
print("CUDA version             :", torch.version.cuda)
print('CUDNN version            :', torch.backends.cudnn.version())
print("torch_scatter version    :", torch_scatter.__version__)
print("torch_sparse version     :", torch_sparse.__version__)
print("torch_geometric version  :", torch_geometric.__version__)
print("-------------------------------------------------------")
print('is CUDA available        :', torch.cuda.is_available())
print('available CUDA devices   :', torch.cuda.device_count())
print('active CUDA Device       :', torch.cuda.current_device())
print("-------------------------------------------------------")

call(["nvcc", "--version"])
# call(["nvidia-smi"])

# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# ! nvcc --version
