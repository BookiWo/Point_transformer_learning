# ShapeNet Part 分割实验完整报告

## 一、项目概述与初始设想

**目标**：复现 Point Transformer V1 → V2 → V3 在物体部件分割上的演进，量化每代提升。

**初始设想**：三个模型在统一基准上建立可对照的演进链，预期 V2 > V1、V3 > V2。

**模型来源**：
- V1：自实现（FPS + KNN + Vector Attention）
- V2：自实现——后来发现与官方 Gofinge 不一致，重写为官方版本
- V3：Pointcept 官方 `PointTransformerV3/model.py`
- PTX：论文信息实现（CVPR 2026 Workshop）
- SP2T：探索后放弃（场景模型 + 需要编译 pointops）

---

## 二、数据处理历程

### 2.1 数据集探索

| 阶段 | 数据集 | 类别/Part | 问题 |
|------|--------|----------|------|
| 初始 | ShapeNet Part（本地 HDF5） | 16 类 / 50 part | 只有 3 通道，本地标签 remap |
| 扩展 | PartNet-Fine（HuggingFace） | 50 类 / 311 part | 标签矛盾，offset 累加无约束 |
| 最终 | ShapeNet Part（官方 _normal） | 16 类 / 50 part | 6 通道，全局 pid，centering |

### 2.2 关键数据问题

**问题 1：PartNet 层级标注误解（最严重）**

PartNet 的 50 个目录（Bag-1, Bed-1/2/3, Chair-1/2/3…）是 24 种物体 × 2-3 标注粒度。同一批椅子在不同 level 下有不同 label，训练中出现矛盾标注导致 mIoU 天花板 0.40。

**证据**：
```
Chair-2 vs Chair-3: max coord diff = 0.000000 (SAME shapes!)
Labels disagree on 100.0% of points
```

**修复**：预处理脚本添加 `--all-levels` 选项，默认只保留最细粒度（24 类，168 part）。

**问题 2：ShapeNet Part 预处理不标准**

| | 原始预处理 | 标准化后 |
|---|---|---|
| 通道 | **3ch xyz** | **6ch xyz+normal** |
| 标签 | 本地 remap 0..K-1 | **全局 pid 0-49** |
| 归一化 | unit-sphere | **centering only** |
| cls_token | 无 | **16 类 one-hot** |

**修复**：重写 `tools/preprocess_shapenet_part.py`，读取官方 `_normal.zip` 的 `.txt` 文件（x y z nx ny nz label）。

**问题 3：全局标签 offset 累加无 category2part 约束**

PartNet-Fine 的 311 类中每个样本只有 2-21 个有效通道，其余全是负样本。梯度被稀释。官方 Pointcept 用 `category2part` 字典约束评估范围。

### 2.3 最终数据配置

```
数据集：ShapeNet Part（官方 _normal 版本）
训练样本：12,137 | 验证样本：1,870 | 测试样本：2,874
类别：16 种物体 | Part：50 个全局 ID
输入：6 通道（coord 3 + normal 3）
预处理：centering only，采样到 2048 点
存储：.npz（coord, feat, seg_labels, category_idx）
```

---

## 三、模型实现历程

### 3.1 V1 — 自实现

| 组件 | 实现 | 与官方差异 |
|------|------|-----------|
| Vector Attention | ✅ Per-channel softmax | KNN 用 PyTorch 非 pointops |
| FPS 下采样 | ✅ PyTorch 实现 | 等价 |
| cls_token | ✅ 后加入 | 对齐官方 partseg |
| 隐藏维度 | 128 | 自选 |

**最终表现**：0.744 cat_mIoU（论文 0.866，差距主要来自 pointops C++ + voting test）

### 3.2 V2 — 从自实现到官方

| 版本 | 问题 |
|------|------|
| **自实现 V2** | 架构与官方 Gofinge 完全不同（dot-product GVA vs MLP weight-encoding GVA, LayerNorm vs PointBatchNorm, 双残差 vs pre-activation） |
| **官方 V2 (ptv2_official.py)** | 完整移植 Gofinge/PointTransformerV2，替换 pointops 为 PyTorch 原生实现 |

官方 V2 关键组件：
- **GroupedVectorAttention**：MLP `weight_encoding(relation_qk)` 回归注意力权重（非点积）
- **Block**：pre-activation + PointBatchNorm + DropPath + 单层 Linear FFN
- **GridPool**：voxel_grid + segment_csr
- **UnpoolWithSkip**：map 后端 unpooling

**最终表现**：**0.799 cat_mIoU，4.9M 参数，162s/epoch**

### 3.3 V3 — 官方 Pointcept

直接使用官方 `PointTransformerV3/model.py`（982 行，0 diff）。修复 `point_transformer_v3_seg.py` 中 coord（3D）与 feat（6ch）分离。

**patch_size 调整**：官方 1024 为 ScanNet 100K 点设计。2K 点下仅 2 patches → 调整为 256 cascade（=8 patches）。

**最终表现**：0.585 cat_mIoU，46.2M，366s/epoch。

### 3.4 PTX — PointTransformerX

V3 框架基础上替换三个组件：
- **3D-GS-RoPE**：每头 6 参数学习旋转坐标基
- **LinearEmbedding**：替换 spconv.SubMConv3d
- **ReLU² FFN + r=2**

**最终表现**：0.625 cat_mIoU，~10M，350s/epoch。

### 3.5 SP2T

探索后放弃：
- 需要编译 pointops C++ 扩展
- 基于 V3 serialization 框架 → 小物体瓶颈相同
- 论文只验证了场景分割（ScanNet/S3DIS/Waymo）

---

## 四、失误分类整理

### 4.1 代码实现问题

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 1 | V1 使用 Scalar Attention `.sum(dim=-1)` | 非论文 Vector Attention | 去除 sum，保留 [B,N,K,H,D] |
| 2 | Grid Pooling Python for-loop | 49× 减速 + 18GB OOM | 向量化 scatter_reduce |
| 3 | `_batch_gather` expand+gather 反向 | 2GB 中间张量 | Fancy indexing |
| 4 | V3 适配用 MLP 替换 spconv CPE | 丢失 26 个空间邻居信息 | 使用原版 + 安装 spconv |
| 5 | V2 自实现非官方架构 | dot-product vs MLP GVA 差异 | 完整移植 Gofinge 代码 |

### 4.2 OOM 与资源配置

| # | 问题 | 原因 | 修复 |
|---|------|------|------|
| 6 | V2 hidden_dim=256 OOM (32GB) | batch=16, 256ch, 62M param | 降为 192/6，batch=6+GA=3 |
| 7 | V2 容量被削后 plateau | 144ch/4layers 对 311 类不足 | 恢复 192/6 |
| 8 | V3 batch=4+GA=4 才能跑 | spconv + serialization 显存大 | 接受 |

### 4.3 参数不匹配

| # | 问题 | 表现 | 修复 |
|---|------|------|------|
| 9 | `global_num_parts` 检测错误 | 只读首样本 → CUDA assert 越界 | 统一为 `ds_ref.global_num_parts` |
| 10 | hasattr/sed 破坏缩进 | IndentationError | 避免 sed 改 Python |
| 11 | V3 patch_size=1024 不匹配 2K 点 | 2 patches，退化全局 attention | 256 cascade |
| 12 | V2 old checkpoint hidden_dim=144→192 | size mismatch | 从零重训 |

### 4.4 数据集处理不当

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 13 | ShapeNet Part 3ch + unit-sphere + 本地 label | 非标准配置 | 重写预处理为 6ch + centering + 全局 pid |
| 14 | PartNet 50 类层级冲突 | 0.40 mIoU 天花板 | finest-level-only 过滤 |
| 15 | 311 类 offset 累加无 category2part | 负样本梯度稀释 | ShapeNet Part 自带 50 全局 part |
| 16 | 预处理没有 normal | V1/V2/V3 只有 3ch 输入 | 用官方 _normal.zip |

### 4.5 工程操作失误

| # | 问题 | 后果 |
|---|------|------|
| 17 | sed 改 train.py 加 grad_accum | optimizer.step() 从不调用 |
| 18 | `run_nohup.sh` 日志路径猜测错误 | 用户每次手动找日志 |
| 19 | 误杀 DataLoader worker 进程 | 主进程崩溃，丢 epoch 21-28 |
| 20 | 清理时误删 V1/V2 日志 | 已从远程补下 |
| 21 | 多次建议修改官方超参 | 用户多次纠正 |
| 22 | `git push` 反复网络超时 | 用 token 重试解决 |

### 4.6 训练策略失误

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 23 | Per-category 独立训练 50 个 V2 | 小类 600 步停 | Unified dataset |
| 24 | 联合训练不加 cls_token | epoch 60+ plateau | Bottleneck 注入 cls_embed |
| 25 | V2/V3 first on PartNet-Fine | 0.33/0.28 无法对照论文 | 回归 ShapeNet Part |

---

## 五、训练实验总结

### 5.1 完整实验矩阵（三阶段，按时间顺序）

**阶段一（2026-06-07 ~ 06-14）：旧 ShapeNet Part**

数据：本地 HDF5（`datasets/raw/shapeNet/hdf5_data`）。预处理错误——`3ch xyz`（无 normal）、`unit-sphere` 归一化（非 centering）、`本地 label remap 0..K-1`（非全局 pid 0-49）、**无 cls_token**。

| 实验 | 模型 | hidden/layers | 参数 | Epochs | val_mIoU | 配置 |
|------|------|--------------|------|--------|----------|------|
| 1 | **V1** | 256/8 | ~110M | 80 | **0.490** | partnet_pt_baseline.yaml |
| 2 | **V2（自实现旧版）** | 128/4 | 26.1M | 80 | **0.468** | partnet_pt_v2_baseline.yaml |
| 3 | **V2（自实现旧版）** | 256/8 | 208.6M | 80 | **0.476** | partnet_pt_v2_autodl.yaml |

**V2 低于 V1 的原因**：V2 自实现架构有误——dot-product GVA（非官方 MLP weight-encoding GVA）、LayerNorm（非 PointBatchNorm）、双残差（非 pre-activation）。在错误数据上，V1 的 Vector Attention 反而更稳定。

对应文件：[`training_dashboard.png`](outputs/viz/autodl_rtx5090/training_dashboard.png)、[`training_comparison.png`](outputs/viz/v2_report/training_comparison.png)、[`V2_TRAINING_REPORT.md`](outputs/viz/v2_report/V2_TRAINING_REPORT.md)

**阶段二（2026-06-20 ~ 06-22）：PartNet-Fine**

数据：24 类、168 parts（offset 累加全局映射，无 category2part 约束）。标签矛盾已通过 finest-level-only 过滤消除。

| 实验 | 模型 | 参数 | Epochs | avg_cat_mIoU | 配置 |
|------|------|------|--------|-------------|------|
| 4 | V2 old + cls_token | 62.7M | 169 | 0.334 | partnet_pt_v2_unified.yaml |
| 5 | V3 official（patch_size=1024） | 46.2M | 140 | 0.283 | partnet_pt_v3_unified.yaml |

**结果偏低原因**：168 parts 用 offset 累加 → 每样本只用到 2-21 个 part，其余 150+ 个通道全是负样本。V3 的 serialization 在 2K 点上退化（2 patches）。回归 ShapeNet Part 后验证。

**阶段三（2026-06-23 ~ 06-25）：ShapeNet Part 标准基准**

数据：官方 `shapenetcore_partanno_segmentation_benchmark_v0_normal`，6ch（coord+normal）、centering、全局 pid 0-49、cls_token=16。

| 实验 | 模型 | 来源 | 参数 | Epochs | cat_mIoU | 训练时间 | 每 epoch | 配置 |
|------|------|------|------|--------|----------|---------|---------|------|
| 6 | **V1** | 自实现 | 26.1M | 200 | **0.744** | 31.5h | 567s | shapenet_v1_baseline.yaml |
| 7 | **V2 Official** | Gofinge | 4.9M | 200 | **0.799** | 9h | 162s | shapenet_v2_official.yaml |
| 8 | **V3 Official** | Pointcept | 46.2M | 200 | **0.585** | ~20h | 366s | shapenet_v3_official.yaml |
| 9 | **PTX** | 论文实现 | ~10M | 200 | **0.625** | ~19h | 350s | shapenet_ptx.yaml |

**阶段三为最终结果。阶段一和阶段二是探索过程——其数据不能作为正式结论，但其教训（预处理标准、架构验证、数据理解）是本次实验最有价值的部分。**

### 5.2 最终对比

![四模型对比](outputs/viz/four_model_comparison.png)

| 模型 | cat_mIoU | 参数 | 每 epoch | 依赖 | 来源 |
|------|:---:|------|------|------|------|
| **V2 Official** | **0.799** | **4.9M** | **162s** | torch_scatter | Gofinge |
| V1 | 0.744 | 26.1M | 567s | 纯 PyTorch | 自实现 |
| PTX | 0.625 | ~10M | 350s | torch_scatter | 论文实现 |
| V3 | 0.585 | 46.2M | 366s | spconv | Pointcept |

### 5.3 与论文对照

| 模型 | 论文 mIoU | 实验值 | Δ | 差距来源 |
|------|:---:|:---:|:---:|------|
| PTv1 | 0.866 | 0.744 | -12% | pointops C++、voting test、架构细节（planes vs hidden_dim） |
| PTv2 | 无论文结果 | **0.799** | — | 论文未测试 ShapeNet Part |
| PTv3 | 无论文结果 | 0.585 | — | 架构不适配物体分割 |

---

## 六、实验结论

### 6.1 核心结论

1. **V2 是 ShapeNet Part 上的最佳模型**：最少参数（4.9M）、最快速度（162s/epoch）、最高精度（0.799）。GVA + GridPool + pre-act Block + DropPath 的体系性优势在物体分割上最大化。

2. **V3/PTX 的 serialization 框架在小物体上有根本性瓶颈**：ScanNet 100K 点 → 100 patches，局部性有意义。2K 点 → 2-8 patches，退化全局 attention。V3 论文的 "Simpler, Faster, Stronger" 只适用于场景级点云。

3. **参数规模 ≠ 性能**：V3 46M < V2 5M，PTX 10M < V2 5M。架构适配性比参数量重要。

4. **cls_token 在部件分割中必不可少**：让模型明确知道处理的是哪种物体，消除隐式形状分类负担。

5. **PTX 验证了"无稀疏算法"的可行性**，但 serialization 框架本身仍是瓶颈。

### 6.2 ShapeNet Part 基准参考

| 模型 | cat_mIoU | 参数 | 速度 | 架构类型 |
|------|----------|------|------|---------|
| PTv1 (Pointcept, voting) | 0.866 | 26M | — | KNN + FPS |
| **PTv2 (Gofinge)** | **0.799** | **4.9M** | **162s** | **GVA + GridPool** |
| PTv1 (our) | 0.744 | 26M | 567s | KNN + FPS |
| PTv3 (Pointcept) | 0.585 | 46M | 366s | Serialized + spconv |
| PTX | 0.625 | 10M | 350s | Serialized (无 spconv) |

**V2 可作为 ShapeNet Part 上的轻量 baseline。**

### 6.3 后续方向

1. **V2 加 voting test**：8 视角随机偏移 soft voting，预期 +2-3% mIoU
2. **pointops 编译**：C++ KNN/FPS 可缩小 V1 与论文的差距
3. **Per-category heads**：用 category2part 约束分类头，替代 50 类全局 head
4. **场景分割验证**：在 ScanNet/S3DIS 上跑 V2 对照论文结果

### 6.4 方法论教训

| 原则 | 说明 |
|------|------|
| **数据先行** | 理解数据集结构优先于任何代码改动 |
| **忠实复现** | 官方实现 > 自己写的"简化版" |
| **正确基准** | 先在有论文数值的基准上达标，再拓展 |
| **不自行改参** | 原文配置是作者调试的结果，先排查数据差异 |
| **谨慎操作** | 杀进程前 100% 确认；sed 不适用于 Python |
