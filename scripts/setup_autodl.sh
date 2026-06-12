#!/bin/bash
# ============================================================
# AutoDL 一键环境配置 (在云实例上运行)
# ============================================================
# 用法:
#   1. git clone <repo> /root/autodl-tmp/Point_transformer_learning
#   2. 上传 autodl_data_package.tar.gz 到 /root/autodl-tmp/
#   3. cd /root/autodl-tmp/Point_transformer_learning
#   4. bash scripts/setup_autodl.sh
#   5. 选择: V1 训练 或 V2 训练
# ============================================================

set -euo pipefail

REPO_ROOT="/root/autodl-tmp/Point_transformer_learning"
DATA_PKG="/root/autodl-tmp/autodl_data_package.tar.gz"

echo "============================================"
echo "  AutoDL Environment Setup"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ---- 1. Conda environment ----
echo "[1/5] Setting up conda environment..."
if conda env list | grep -q pt_partnet; then
    echo "  ⏭ conda env 'pt_partnet' already exists, skipping"
else
    conda create -n pt_partnet python=3.10 -y
    echo "  ✅ conda env 'pt_partnet' created"
fi

source activate pt_partnet

# ---- 2. Install PyTorch (CUDA 12.x) ----
echo ""
echo "[2/5] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
echo "  ✅ PyTorch installed"

# ---- 3. Install project dependencies ----
echo ""
echo "[3/5] Installing project dependencies..."
pip install numpy pyyaml h5py tensorboard open3d -q
echo "  ✅ Dependencies installed"

# ---- 4. Extract data ----
echo ""
echo "[4/5] Extracting data..."
if [ -f "$DATA_PKG" ]; then
    cd /root/autodl-tmp
    tar -xzf "$DATA_PKG"
    echo "  ✅ Data extracted to /root/autodl-tmp/datasets/"
else
    echo "  ⚠ $DATA_PKG not found!"
    echo "  Upload it first: AutoDL 控制台 → 文件上传 → /root/autodl-tmp/"
    exit 1
fi

# ---- 5. Verify ----
echo ""
echo "[5/5] Quick verification..."
cd "$REPO_ROOT"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Start V1 training:"
echo "    cd $REPO_ROOT"
echo "    python train.py --config configs/partnet_pt_autodl.yaml"
echo ""
echo "  Start V2 training (GVA + PE Mul + Grid Pooling):"
echo "    cd $REPO_ROOT"
echo "    python train.py --config configs/partnet_pt_v2_autodl.yaml"
echo ""
echo "  TensorBoard:"
echo "    tensorboard --logdir /root/autodl-tmp/experiments/ --bind_all"
echo "============================================"
