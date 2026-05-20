# PyTorch 常用函数与类参考

本参考整理 PyTorch 在模型开发训练中常见且实用的函数、类与标准用法。每项包含简短定义、标准签名与示例代码片段。

---

## 1. Tensors（张量）

- `torch.tensor(data, dtype=None, device=None)`
  - 构造张量。例：`t = torch.tensor([[1.,2.]], dtype=torch.float32)`
- `torch.from_numpy(ndarray)` / `tensor.numpy()`
  - 在 CPU 上共享内存转换。例：`t = torch.from_numpy(arr)`，`arr = t.numpy()`（需 `.cpu()`）。
- `.to(device)`
  - 移动到设备或改变 dtype：`x = x.to('cuda')` 或 `x = x.to(torch.float16)`。
- `.requires_grad_(True/False)`
  - 控制是否计算梯度（autograd）。

示例：
```python
x = torch.randn(4,3, requires_grad=True)
y = x.mean()
y.backward()
print(x.grad)
```

---

## 2. Autograd（自动微分）

- `tensor.backward(retain_graph=False)`
  - 计算梯度（标量输出或显式传入梯度）。
- `with torch.no_grad():` / `torch.set_grad_enabled(mode)`
  - 在推理或保存内存时关闭梯度跟踪。
- `torch.autograd.grad(outputs, inputs, grad_outputs=None)`
  - 更细粒度求导。

示例：
```python
with torch.no_grad():
    out = model(x)
```

---

## 3. 神经网络模块（nn.Module）

- `class MyModel(nn.Module):`
  - 标准结构：在 `__init__` 定义子模块，在 `forward(self, x)` 中实现前向。
- `model.parameters()` / `model.named_parameters()`
  - 获取可学习参数，用于优化器。
- `model.train()` / `model.eval()`
  - 切换训练/评估模式（影响 dropout、batchnorm）。
- `model.state_dict()` / `model.load_state_dict(state)`
  - 保存/加载模型权重和缓冲。

示例：
```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10,2)
    def forward(self,x):
        return self.fc(x)

model = M(); model.train()
```

---

## 4. 常用层（nn）

- `nn.Linear(in_features, out_features, bias=True)`
- `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
- `nn.BatchNorm1d(num_features)` / `nn.LayerNorm(normalized_shape)`
- `nn.Dropout(p=0.5)`
- `nn.GRU/LSTM/Transformer/Embedding`（按需使用）
- `nn.MultiheadAttention(embed_dim, num_heads, ...)`（Transformer 注意力）

示例：
```python
layer = nn.Linear(128, 64)
x = layer(x)
```

---

## 5. 损失函数（losses）

- `nn.CrossEntropyLoss(ignore_index=-100)`
  - 常用于分类/分割（logits 输入，labels 为 class idx）。
- `nn.MSELoss()` / `nn.L1Loss()`
- `nn.BCEWithLogitsLoss()`（二分类，稳定数值）

示例（分割）：
```python
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
loss = loss_fn(logits.view(-1,C), labels.view(-1))
```

---

## 6. 优化器与调度（optim & lr_scheduler）

- `torch.optim.SGD(params, lr, momentum=0.9, weight_decay=0)`
- `torch.optim.Adam(params, lr, betas=(0.9,0.999), weight_decay=0)`
- `torch.optim.AdamW(...)`（推荐用于 Transformer）

- 学习率调度：`torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)`、`CosineAnnealingLR`、`ReduceLROnPlateau`。

示例：
```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()
```

---

## 7. DataLoader 与 Dataset

- `class MyDataset(Dataset): __len__ / __getitem__`
- `DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)`
- collate_fn：自定义批次拼接逻辑

示例：
```python
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
for batch in loader:
    points = batch['points']  # shape [B,N,3]
```

---

## 8. Mixed precision（混合精度）

- `torch.cuda.amp.autocast()` 与 `torch.cuda.amp.GradScaler()`

示例：
```python
scaler = torch.cuda.amp.GradScaler()
for x,y in loader:
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = loss_fn(logits,y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 9. Checkpoint（保存/加载）

- `torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': e}, path)`
- `ckpt = torch.load(path, map_location=device)`
- `model.load_state_dict(ckpt['model'])`

建议保存 `config` 与 `scaler`（若使用 amp）。

---

## 10. 训练循环标准模板

```python
for epoch in range(1, epochs+1):
    model.train()
    for batch in train_loader:
        x = batch['points'].to(device)
        y = batch['labels'].to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        evaluate(...)
    scheduler.step()
```

---

## 11. 常见工具函数/技巧

- `torch.save` / `torch.load`：保存与加载对象
- `torch.no_grad()`：评估时禁用 grad
- `model.eval()`：评估模式（影响 dropout/batchnorm）
- `optimizer.zero_grad(set_to_none=True)`：更高效地置零梯度
- `torch.nn.utils.clip_grad_norm_`：梯度裁剪，避免爆炸
- `torch.cuda.empty_cache()`：释放未使用 GPU 内存（仅调试）

---

## 12. 分布式训练与多卡（速览）

- 单机多卡：`torch.nn.DataParallel(model)`（简单）
- 多机/单机多卡：`torch.nn.parallel.DistributedDataParallel`（推荐）
- 关键点：使用 `DistributedSampler` 做按进程划分的数据加载，按 rank 保存 checkpoint。

---

## 13. 调试建议

- 先用 `batch_size=1, num_workers=0, max_samples=20` 做冒烟测试。
- 使用 `torch.autograd.set_detect_anomaly(True)` 定位反向传播异常。
- 在出现 NaN 时打印 `loss`、检查 labels 范围与 `ignore_index`。

---

文件已保存：`docs/pytorch_reference.md`。如需补充示例、画图或把该文档园艺化为教学幻灯片，我可以继续扩展。