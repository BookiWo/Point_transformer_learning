# CLAUDE.md — Point Transformer PartNet Segmentation

## 项目概述

基于 Point Transformer 的点云部件分割项目，使用 PartNet 数据集。
严格按 "paper → code → experiment → analysis" 流程推进。

## 环境

- **WSL Ubuntu 24.04** (项目在 `/mnt/d/` 下，跨文件系统访问)
- **Conda 环境**: `pt_partnet` (`/home/laplace37/miniconda3/envs/pt_partnet`)
- **激活方式**: `conda activate pt_partnet`
- **Python**: 3.x (conda 环境内)
- **Node.js**: v20.20.2 (via nvm, 用于 Claude Code)
- **关键依赖**: PyTorch >= 2.2, Open3D >= 0.18, numpy, PyYAML, h5py, tensorboard, pytest

## 目录结构

| 目录/文件 | 用途 |
|---|---|
| `configs/` | 实验 YAML 配置 (baseline, preprocess) |
| `datasets/raw/` | PartNet 原始数据 (不可改) |
| `datasets/processed/` | 预处理后的数据 |
| `datasets/splits/` | train/val/test 划分 |
| `datasets/partnet_dataset.py` | DataLoader 实现 |
| `models/blocks/` | Point Transformer 基础模块 (KNN, attention, 位置编码) |
| `models/backbones/` | 编码器/解码器主干 |
| `models/heads/` | 分割头 (每点分类) |
| `models/point_transformer_seg.py` | 完整网络入口 |
| `losses/` | 损失函数 (CrossEntropy 等) |
| `utils/metrics/` | mIoU, Acc, per-class IoU |
| `utils/visualization/` | 点云预测可视化 |
| `tools/preprocess_partnet.py` | 数据预处理脚本 |
| `train.py` | 训练入口 |
| `eval.py` | 评估入口 |
| `experiments/` | 实验输出 (checkpoints, logs, tensorboard) |
| `outputs/preds/` | 预测结果 .npz |
| `outputs/viz/` | 可视化 .ply |
| `scripts/` | 辅助运行脚本 |
| `tests/` | pytest 单元测试 |
| `notebooks/` | Jupyter 分析笔记本 |

## 常用命令

### 环境激活
```bash
conda activate pt_partnet
```

### 数据预处理
```bash
python tools/preprocess_partnet.py
```

### 训练
```bash
python train.py --config configs/partnet_pt_baseline.yaml
```

### TensorBoard
```bash
tensorboard --logdir experiments/exp_pt_partnet_baseline/tensorboard
```

### 评估
```bash
python eval.py --config configs/partnet_pt_baseline.yaml --checkpoint experiments/exp_pt_partnet_baseline/checkpoints/best.pth --split test --save-viz
```

### 运行测试
```bash
pytest -q
```

## 配置系统

所有超参在 `configs/partnet_pt_baseline.yaml` 中管理：
- `dataset.num_points`: 2048
- `dataset.num_parts`: 50
- `model.hidden_dim`: 128
- `model.num_layers`: 4
- `model.num_heads`: 4
- `training.batch_size`: 8
- `training.lr`: 0.001
- `training.epochs`: 30
- `training.seed`: 42

## 编码规范

1. **先写配置，再写代码** — 所有参数放 configs/，避免硬编码
2. **先单测模块** — 每个新模块用随机张量测试前向
3. **先 1 batch overfit** — 确认 loss 下降再长训
4. **每次只改一个变量** — 便于定位问题
5. **保存实验记录** — 参数/结果/现象写入对应 experiment 目录

## 关键设计要点

- Point Transformer 中 attention 的 K/Q/V 来自点特征
- 点云任务需相对位置编码 (KNN 邻域内)
- mIoU 比 OA 更关键 (类别不平衡场景)
- 训练震荡时优先调: lr, weight_decay, batch_size
- PartNet 标签体系映射到 `num_parts=50`

## Git 提交规范

- feat: 新功能
- fix: 修复
- refactor: 重构
- docs: 文档
- exp: 实验相关
