# PyTorch学习笔记

## 快速概览

- 项目主要模块：数据预处理（`tools/`）、数据集加载（`datasets/`）、模型实现（`models/`）、损失/指标（`losses/`, `utils/metrics/`）、训练/评估脚本（`scripts/`）和可视化（`utils/visualization/`）。
- 推荐流程：预处理 → 验证 Dataset → 模型前向测试 → 训练小规模实验 → 完整训练与评估。

---

## 一键运行（建议按顺序）

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 预处理（冒烟测试，先处理少量样本）：
```bash
python tools/preprocess_partnet.py
```

3. 训练基线：
```bash
python train.py --config configs/partnet_pt_baseline.yaml
```

4. 评估并保存可视化：
```bash
python eval.py --config configs/partnet_pt_baseline.yaml --checkpoint experiments/exp_pt_partnet_baseline/checkpoints/best.pth --split test --save-viz
```

5. 运行单元测试（验证基本模块）：
```bash
PYTHONPATH=. python -m pytest tests/test_metrics.py tests/test_dataset.py -q
```

---

## 项目关键文件与常用函数速查

- `tools/preprocess_partnet.py` — 预处理入口
  - `_load_label_hook()`：动态加载可选钩子文件 `preprocess_partnet_label_hooks.py` 中的 `build_label_payload`，并用 `lru_cache` 缓存。用于为每个样本生成额外标签或元数据。
  - `build_label_payload(file_path, config)`：统一调用 hook 的接口，返回额外字段字典。
  - `load_config()` / `resolve_paths(config, repo_root)`：读取 `configs/Partnet_preprocess.yaml` 并解析路径。
  - `normalize_points(points, method)`：支持 `unit_sphere`、`bounding_box` 两种归一化方法，返回 `(points, center, scale)`。
  - `maybe_sample(...)`：按配置随机/重复采样到固定点数。
  - `save_sample(...)`：保存为 `.npz`（包含 `points`, `seg_labels`, `category_id` 等）或 `.ply`。

- `configs/Partnet_preprocess.yaml`、`configs/partnet_pt_baseline.yaml` — 配置文件，包含数据、采样、归一化、训练超参等。

- `datasets/partnet_dataset.py` — 数据加载器
  - `PartNetDataset(processed_root, split_file, num_points, augment)`：读取 split 列表，返回样本路径；`__getitem__` 返回 `{ 'points', 'labels', 'category', 'sample_id' }`（Tensor）。
  - `_resample(points, labels)`：重采样到指定 `num_points`（允许重复）。
  - `_augment(points)`：演示旋转 + jitter 的数据增强。

- `models/` — 模型骨架（轻量 Point Transformer 风格）
  - `models/blocks/simple_point_transformer_block.py`：`SimplePointTransformerBlock(dim, num_heads, dropout)`，包含位置编码 MLP、`MultiheadAttention`、FFN 与残差。
  - `models/backbones/point_transformer_backbone.py`：`PointTransformerBackbone`，由 stem + 多层 block 组成。
  - `models/heads/segmentation_head.py`：`SegmentationHead(hidden_dim, num_parts)`，点级分类头。
  - `models/point_transformer_seg.py`：`PointTransformerSeg`，整网封装，`forward(points)` 返回 `[B,N,num_parts]` logits。

- `losses/segmentation_loss.py` — `SegmentationLoss(ignore_index)`：CrossEntropyLoss 包装，接收 `logits [B,N,C]` 与 `labels [B,N]`。

- `utils/metrics/segmentation_metrics.py` — `compute_segmentation_metrics(logits, labels, num_parts, ignore_index)`：输出 `overall_acc` 与 `miou`。

- `utils/visualization/pointcloud_viz.py` — `label_to_color(labels)`、`save_xyzrgb_ply(path, points, colors)`：用于将预测结果保存为带颜色的 PLY，方便可视化。

- `scripts/train.py` / `scripts/eval.py` — 训练与评估实现（加载配置、构建 dataloader、模型训练/验证、保存 checkpoint、评估并保存预测/可视化）。

---

## 推荐学习路径（按天/周计）

1. Python 与 NumPy 基础（1–3 天）
   - 学习重点：数组操作、广播、随机采样、文件 IO。练习：实现点云下采样/上采样函数。

2. PyTorch 基础（2–4 天）
   - 学习重点：`Tensor`、`autograd`、`nn.Module`、`Dataset`/`DataLoader`、训练循环、优化器。练习：用随机数据训练一个小 MLP，使 loss 下降。

3. 点云基础与 Open3D（1–2 天）
   - 学习重点：读写 PLY、可视化、法线、简单变换。练习：读取 PLY 并保存带颜色的 PLY。

4. 预处理与 Dataset（1–2 天）
   - 学习重点：采样、归一化、标签映射、split 管理。练习：运行 `tools/preprocess_partnet.py`（`max_samples` 模式），并用 `PartNetDataset` 检查 batch 输出。

5. 模型构建（2–4 天）
   - 学习重点：`MultiheadAttention` 的用法、位置编码、LayerNorm、残差结构。练习：单测 `SimplePointTransformerBlock` 的 forward。

6. 训练与调参（2–4 天）
   - 学习重点：调节学习率、权重衰减、batch size、scheduler。练习：在小数据集上训练 10 个 epoch，记录曲线并改参数观察差别。

7. 进阶（连续学习）
   - KNN 邻域、FPS（furthest point sampling）、局部注意力、encoder-decoder 分层结构、跨类别 IoU 报表与消融实验。

---

## 练习任务（项目导向）

- 任务 A：把 `maybe_sample` 换成 FPS（实现或使用已有实现），比较训练效果。保留 `max_samples` 做冒烟验证。
- 任务 B：在 `SimplePointTransformerBlock` 中实现基于 KNN 的局部注意力（只计算邻域点的 attention），观察速度与精度变化。
- 任务 C：在 `datasets/partnet_dataset.py` 中加入 normals 支持并在模型输入中使用（把 normals 拼接到点特征）。
- 任务 D：扩展 `scripts/eval.py` 以输出 per-class IoU CSV 报表，并保存到 `outputs/reports/`。
- 任务 E：做消融实验（`num_points`=2048 vs 4096），把结果写成短报告放入 `outputs/reports/`。

---

## 调试与常见问题速查

- 标签错位/编码不一致：保证预处理输出的 `seg_labels` 与训练时 `num_parts` 对齐；若使用类内局部 id，需要统一映射到全局 id。 
- 归一化不一致：训练与推理使用相同归一化；可在 `extra_payload` 中保存 `norm_center`/`norm_scale` 以便反归一化可视化。 
- 内存/速度问题：先用小 `max_samples` 或减小 `batch_size`；检查 `num_workers` 设置；避免一次性加载所有数据到内存。 
- CUDA/设备问题：确保所有张量在同一 device（`to(device)`）；在调用 `.numpy()` 前 `.cpu()`。

---

## 参考资料与进阶阅读

- PyTorch 官方教程：https://pytorch.org/tutorials/ 
- Point Transformer 相关论文与实现（推荐阅读原始论文并对比实现细节）
- Open3D 文档：https://www.open3d.org/docs/ 

---

## 文件位置

学习笔记已保存：`docs/py_tutorial_for_project.md`（就是本文件）。

祝你学习顺利！如果需要，我可以把练习任务分解成可执行的 Issue/PR 并逐一实现。