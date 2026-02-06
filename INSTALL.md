# Installation Instructions

Follow these steps to set up the project environment. These instructions are optimized for a broader environment support including **CUDA 12.1** support.

### 1. Clone the Repository
```bash
git clone https://github.com/ayush1298/DeepGLSTM.git
cd DeepGLSTM
```

### 2. Install PyTorch (CUDA 12.1)
Install PyTorch, torchvision, and torchaudio compatible with CUDA 12.1.
```bash
python3 -m pip install --force-reinstall --no-cache-dir \
  torch==2.3.1+cu121 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric Dependencies
Install optional dependencies for PyG (scatter, sparse, cluster, spline_conv) from the PyG wheel index.
```bash
python3 -m pip install --no-cache-dir \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### 4. Install PyTorch Geometric
```bash
python3 -m pip install torch_geometric
```

### 5. Install Remaining Requirements
Install other dependencies (NumPy, Pandas, RDKit, etc.) from the requirements file.
```bash
python3 -m pip install -r requirements.txt
```

> [!NOTE]
> **RDKit Compatibility**: We are using the official `rdkit` package instead of `rdkit-pypi`, as the latter does not have a compatible version for this setup.

### 6. Verify Installation
Run the following Python script to verify that the correct versions are installed and CUDA is available.

```python
import torch
import torch_geometric

print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("PyG  :", torch_geometric.__version__)

# Expected Output:
# Torch: 2.3.1+cu121
# CUDA : 12.1
# PyG  : 2.7.0
```
