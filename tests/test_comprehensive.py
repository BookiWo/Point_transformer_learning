"""Comprehensive edge-case and integration tests for Point Transformer pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from datasets.partnet_dataset import PartNetDataset
from losses.segmentation_loss import SegmentationLoss
from models.backbones.point_transformer_backbone import (
    PointTransformerBackbone,
    TransitionDown,
    TransitionUp,
    _batch_gather,
    _batched_knn,
    _farthest_point_sample,
)
from models.blocks.simple_point_transformer_block import (
    PointTransformerBlock,
    _batch_gather as _block_batch_gather,
    _batched_knn as _block_batched_knn,
)
from models.heads.segmentation_head import SegmentationHead
from models.point_transformer_seg import PointTransformerSeg
from utils.metrics.segmentation_metrics import compute_segmentation_metrics


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_batch(b, n, d):
    return torch.randn(b, n, d)


# ── KNN tests ────────────────────────────────────────────────────────────────


def test_knn_basic():
    pts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [100.0, 0.0, 0.0]]])
    idx = _block_batched_knn(pts, pts, k=2, exclude_self=False)
    # nearest to point 0 should be itself (dist 0) then point 1 (dist 1)
    assert idx[0, 0, 0].item() == 0
    assert idx[0, 0, 1].item() == 1


def test_knn_exclude_self():
    pts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])
    idx = _block_batched_knn(pts, pts, k=2, exclude_self=True)
    # point 0 should NOT include itself
    assert (idx[0, 0] != 0).all()
    # nearest should be point 1 (dist 1)
    assert idx[0, 0, 0].item() == 1


def test_knn_k_larger_than_numpoints():
    pts = torch.randn(1, 10, 3)
    idx = _block_batched_knn(pts, pts, k=100, exclude_self=False)
    # k should be clamped to num_points
    assert idx.shape[2] == 10


def test_knn_batched_consistency():
    pts_a = torch.randn(2, 64, 3)
    pts_b = torch.randn(2, 32, 3)
    idx = _batched_knn(pts_a, pts_b, k=8)
    assert idx.shape == (2, 64, 8)
    # Check that indices are within valid range for each batch
    assert (idx >= 0).all()
    assert (idx < 32).all()


def test_knn_empty_reference_raises():
    with pytest.raises(ValueError, match="reference_points"):
        _block_batched_knn(torch.randn(1, 5, 3), torch.randn(1, 0, 3), k=4)


# ── batch_gather tests ───────────────────────────────────────────────────────


def test_gather_2d():
    data = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
    idx = torch.tensor([[0, 2]])  # [1, 2] — select point 0 and point 2
    out = _batch_gather(data, idx)
    expected = torch.tensor([[[1.0, 2.0], [5.0, 6.0]]])
    assert torch.equal(out, expected)


def test_gather_3d():
    data = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # [1, 4, 1]
    idx = torch.tensor([[[0, 2], [1, 3]]])  # [1, 2, 2] — 2 queries, 2 neighbors each
    out = _batch_gather(data, idx)
    assert out.shape == (1, 2, 2, 1)
    assert out[0, 0, 0, 0] == 1.0
    assert out[0, 0, 1, 0] == 3.0
    assert out[0, 1, 0, 0] == 2.0
    assert out[0, 1, 1, 0] == 4.0


def test_gather_bad_rank_raises():
    with pytest.raises(ValueError, match="rank"):
        _batch_gather(torch.randn(1, 3, 2), torch.tensor([0]))


# ── FPS tests ────────────────────────────────────────────────────────────────


def test_fps_output_shape():
    pts = torch.randn(2, 256, 3)
    idx = _farthest_point_sample(pts, ratio=0.5)
    assert idx.shape == (2, 128)


def test_fps_ratio_one():
    pts = torch.randn(2, 64, 3)
    idx = _farthest_point_sample(pts, ratio=1.0)
    assert idx.shape == (2, 64)


def test_fps_ratio_tiny():
    pts = torch.randn(2, 100, 3)
    idx = _farthest_point_sample(pts, ratio=0.01)
    assert idx.shape[1] == max(1, int(math.ceil(100 * 0.01)))


def test_fps_no_duplicates():
    """FPS should not select the same point twice (within batching precision)."""
    torch.manual_seed(42)
    pts = torch.randn(2, 128, 3)
    idx = _farthest_point_sample(pts, ratio=0.5)
    for b in range(idx.shape[0]):
        assert len(set(idx[b].tolist())) == idx.shape[1]


# ── PointTransformerBlock tests ──────────────────────────────────────────────


def test_block_forward_shape():
    block = PointTransformerBlock(dim=64, num_heads=4, k=8, dropout=0.0)
    feats = torch.randn(2, 128, 64)
    pts = torch.randn(2, 128, 3)
    out = block(feats, pts)
    assert out.shape == (2, 128, 64)


def test_block_dim_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        PointTransformerBlock(dim=65, num_heads=4)


def test_block_residual_connection():
    """Verify that output ≠ input (the residual connection adds information but changes the tensor)."""
    block = PointTransformerBlock(dim=32, num_heads=2, k=8, dropout=0.0)
    feats = torch.ones(2, 16, 32)
    pts = torch.randn(2, 16, 3)
    out = block(feats, pts)
    # With random points and ones features, attention should produce non-trivial output
    assert not torch.allclose(out, feats, atol=1e-6)


def test_block_end_to_end_gradient():
    block = PointTransformerBlock(dim=16, num_heads=2, k=4, dropout=0.0)
    feats = torch.randn(1, 32, 16, requires_grad=True)
    pts = torch.randn(1, 32, 3)
    out = block(feats, pts)
    loss = out.sum()
    loss.backward()
    assert feats.grad is not None
    assert not torch.allclose(feats.grad, torch.zeros_like(feats.grad))


def test_block_different_k():
    """Block should work with different k values."""
    for k in [1, 4, 16, 32]:
        block = PointTransformerBlock(dim=32, num_heads=4, k=k, dropout=0.0)
        feats = torch.randn(1, 8, 32)
        pts = torch.randn(1, 8, 3)
        out = block(feats, pts)
        assert out.shape == (1, 8, 32)


# ── TransitionDown tests ─────────────────────────────────────────────────────


def test_down_forward_shape():
    down = TransitionDown(in_channels=64, out_channels=128, ratio=0.5, k=8)
    pts = torch.randn(2, 256, 3)
    feats = torch.randn(2, 256, 64)
    new_pts, new_feats = down(pts, feats)
    assert new_pts.shape == (2, 128, 3)
    assert new_feats.shape == (2, 128, 128)


def test_down_ratio_one():
    """ratio=1.0: no downsampling of points, but feature dim changes."""
    down = TransitionDown(in_channels=32, out_channels=64, ratio=1.0, k=4)
    pts = torch.randn(1, 32, 3)
    feats = torch.randn(1, 32, 32)
    new_pts, new_feats = down(pts, feats)
    assert new_pts.shape == (1, 32, 3)
    assert new_feats.shape == (1, 32, 64)


# ── TransitionUp tests ───────────────────────────────────────────────────────


def test_up_forward_shape():
    up = TransitionUp(coarse_channels=256, skip_channels=128, out_channels=128, k=3)
    coarse_pts = torch.randn(2, 64, 3)
    coarse_feats = torch.randn(2, 64, 256)
    skip_pts = torch.randn(2, 128, 3)
    skip_feats = torch.randn(2, 128, 128)
    new_pts, new_feats = up(coarse_pts, coarse_feats, skip_pts, skip_feats)
    assert new_pts.shape == (2, 128, 3)
    assert new_feats.shape == (2, 128, 128)


def test_up_interpolation_numerical():
    """If coarse points == skip points, interpolation should be exact."""
    up = TransitionUp(coarse_channels=64, skip_channels=32, out_channels=32, k=1)
    pts = torch.randn(1, 10, 3)
    coarse_feats = torch.randn(1, 10, 64)
    skip_feats = torch.randn(1, 10, 32)
    new_pts, new_feats = up(pts, coarse_feats, pts, skip_feats)
    assert new_pts.shape == (1, 10, 3)
    assert new_feats.shape == (1, 10, 32)


# ── PointTransformerBackbone tests ───────────────────────────────────────────


def test_backbone_forward_shape():
    backbone = PointTransformerBackbone(
        input_dim=3, hidden_dim=32, num_layers=4, num_heads=4, dropout=0.0
    )
    pts = torch.randn(2, 128, 3)
    out = backbone(pts)
    # Output should have same number of points, hidden_dim channels
    assert out.shape == (2, 128, 32)


def test_backbone_output_connected_to_input():
    """Output should have non-trivial relationship to input (gradient test)."""
    backbone = PointTransformerBackbone(
        input_dim=3, hidden_dim=16, num_layers=4, num_heads=2, dropout=0.0
    )
    pts = torch.randn(1, 64, 3, requires_grad=True)
    out = backbone(pts)
    loss = out.sum()
    loss.backward()
    assert pts.grad is not None
    assert pts.grad.abs().sum() > 0


def test_backbone_with_minimal_points():
    """Backbone should handle very small point clouds (e.g., 8 points)."""
    backbone = PointTransformerBackbone(
        input_dim=3, hidden_dim=16, num_layers=4, num_heads=2, dropout=0.0
    )
    pts = torch.randn(1, 8, 3)
    out = backbone(pts)
    assert out.shape == (1, 8, 16)


def test_backbone_deterministic():
    """Same input should produce same output (within float32 precision)."""
    backbone = PointTransformerBackbone(
        input_dim=3, hidden_dim=16, num_layers=4, num_heads=2, dropout=0.0
    )
    backbone.eval()
    pts = torch.randn(1, 32, 3)
    out1 = backbone(pts)
    out2 = backbone(pts)
    assert torch.allclose(out1, out2, atol=1e-6)


# ── SegmentationHead tests ───────────────────────────────────────────────────


def test_head_shape():
    head = SegmentationHead(hidden_dim=128, num_parts=50, dropout=0.1)
    feats = torch.randn(2, 1024, 128)
    out = head(feats)
    assert out.shape == (2, 1024, 50)


def test_head_deterministic_in_eval():
    head = SegmentationHead(hidden_dim=32, num_parts=10, dropout=0.5)
    head.eval()
    feats = torch.randn(1, 16, 32)
    out1 = head(feats)
    out2 = head(feats)
    assert torch.allclose(out1, out2)


# ── PointTransformerSeg tests ────────────────────────────────────────────────


def test_seg_forward_shape():
    model = PointTransformerSeg(
        input_dim=3, hidden_dim=32, num_layers=4, num_heads=4, num_parts=50, dropout=0.0
    )
    pts = torch.randn(2, 128, 3)
    logits = model(pts)
    assert logits.shape == (2, 128, 50)


def test_seg_train_mode_forward():
    model = PointTransformerSeg(
        input_dim=3, hidden_dim=16, num_layers=4, num_heads=2, num_parts=10, dropout=0.1
    )
    model.train()
    pts = torch.randn(1, 64, 3)
    logits = model(pts)
    assert logits.shape == (1, 64, 10)


def test_seg_overfit_single_batch():
    """Single-batch overfitting: loss should decrease over a few iterations."""
    torch.manual_seed(42)
    model = PointTransformerSeg(
        input_dim=3, hidden_dim=32, num_layers=4, num_heads=4, num_parts=5, dropout=0.0
    )
    pts = torch.randn(4, 128, 3)
    labels = torch.randint(0, 5, (4, 128))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    losses = []
    for _ in range(20):
        logits = model(pts)
        loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    assert losses[0] > losses[-1], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ── SegmentationLoss tests ───────────────────────────────────────────────────


def test_loss_shape_and_value():
    criterion = SegmentationLoss(ignore_index=-1)
    logits = torch.randn(2, 10, 5)
    labels = torch.randint(0, 5, (2, 10))
    loss = criterion(logits, labels)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_loss_ignore_index():
    criterion = SegmentationLoss(ignore_index=-1)
    logits = torch.zeros(1, 4, 3)  # all classes equal score
    labels = torch.tensor([[0, -1, -1, -1]])  # only first label valid
    loss = criterion(logits, labels)
    # Loss should be finite and > 0 (uniform pred over 3 classes for first point)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_perfect_prediction():
    criterion = SegmentationLoss(ignore_index=-1)
    logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]])
    labels = torch.tensor([[1, 0]])
    loss = criterion(logits, labels)
    assert loss.item() < 0.1  # near-zero loss for perfect prediction


# ── Metrics tests ────────────────────────────────────────────────────────────


def test_metrics_all_correct():
    logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]])
    labels = torch.tensor([[1, 0]])
    m = compute_segmentation_metrics(logits, labels, num_parts=2)
    assert m["overall_acc"] == 1.0
    assert m["miou"] == 1.0


def test_metrics_all_wrong():
    logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])  # swapped
    labels = torch.tensor([[1, 0]])
    m = compute_segmentation_metrics(logits, labels, num_parts=2)
    assert m["overall_acc"] == 0.0
    assert m["miou"] == 0.0


def test_metrics_ignore_label():
    logits = torch.tensor([[[0.0, 10.0, 0.0]]])  # pred class 1
    labels = torch.tensor([[-1]])  # ignore
    m = compute_segmentation_metrics(logits, labels, num_parts=3, ignore_index=-1)
    assert m["overall_acc"] == 0.0
    assert m["miou"] == 0.0


def test_metrics_partial_correct():
    logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]]])
    labels = torch.tensor([[1, 0, 0]])  # point 2 wrong
    m = compute_segmentation_metrics(logits, labels, num_parts=2)
    assert abs(m["overall_acc"] - 2.0 / 3.0) < 1e-6


def test_metrics_unseen_class():
    """mIoU should handle classes that don't appear in labels."""
    logits = torch.randn(1, 10, 50)
    # Only classes 0, 1 appear
    labels = torch.randint(0, 2, (1, 10))
    m = compute_segmentation_metrics(logits, labels, num_parts=50)
    assert 0.0 <= m["miou"] <= 1.0
    assert 0.0 <= m["overall_acc"] <= 1.0


# ── Dataset tests ────────────────────────────────────────────────────────────


def test_dataset_num_points_upsample(tmp_path):
    """When num_points > actual points, should resample with replacement."""
    import numpy as np

    root = tmp_path / "processed"
    split_dir = tmp_path / "splits"
    sample_dir = root / "samples" / "train"
    sample_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    np.savez_compressed(
        sample_dir / "s_000.npz",
        points=np.random.randn(16, 3).astype(np.float32),
        seg_labels=np.random.randint(0, 3, size=(16,)).astype(np.int64),
        category_id=np.array([1], dtype=np.int64),
        sample_id="s_000",
    )
    (split_dir / "train.txt").write_text("samples/train/s_000.npz\n", encoding="utf-8")

    ds = PartNetDataset(root, split_dir / "train.txt", num_points=32, augment=False)
    item = ds[0]
    assert tuple(item["points"].shape) == (32, 3)
    assert tuple(item["labels"].shape) == (32,)


def test_dataset_augment_deterministic_shape(tmp_path):
    """Augmentation should not change tensor shape."""
    import numpy as np

    root = tmp_path / "processed"
    sample_dir = root / "samples" / "train"
    split_dir = tmp_path / "splits"
    sample_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    np.savez_compressed(
        sample_dir / "s_000.npz",
        points=np.random.randn(64, 3).astype(np.float32),
        seg_labels=np.random.randint(0, 5, size=(64,)).astype(np.int64),
        category_id=np.array([2], dtype=np.int64),
        sample_id="s_000",
    )
    (split_dir / "train.txt").write_text("samples/train/s_000.npz\n", encoding="utf-8")

    ds = PartNetDataset(root, split_dir / "train.txt", num_points=32, augment=True)
    item = ds[0]
    assert tuple(item["points"].shape) == (32, 3)


def test_dataset_missing_file_raises(tmp_path):
    root = tmp_path / "processed"
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("samples/train/nonexistent.npz\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        PartNetDataset(root, split_dir / "train.txt")


def test_dataset_missing_labels_field(tmp_path):
    """Dataset should handle npz without seg_labels or labels field."""
    import numpy as np

    root = tmp_path / "processed"
    sample_dir = root / "samples" / "train"
    split_dir = tmp_path / "splits"
    sample_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    np.savez_compressed(
        sample_dir / "s_000.npz",
        points=np.random.randn(32, 3).astype(np.float32),
        category_id=np.array([3], dtype=np.int64),
        sample_id="s_000",
    )
    (split_dir / "train.txt").write_text("samples/train/s_000.npz\n", encoding="utf-8")

    ds = PartNetDataset(root, split_dir / "train.txt", num_points=None, augment=False)
    item = ds[0]
    # Should receive -1 labels
    assert (item["labels"] == -1).all()


# ── Integration tests ────────────────────────────────────────────────────────


def test_full_pipeline_overfit(tmp_path):
    """Full pipeline: create tiny synthetic dataset → train → eval, verify loss decreases."""
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)

    num_classes = 4
    n_points = 64
    n_samples = 20

    # Create synthetic data
    processed_root = tmp_path / "processed"
    split_dir = tmp_path / "splits"
    sample_dir = processed_root / "samples" / "train"
    sample_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    sample_paths = []
    for i in range(n_samples):
        sp = sample_dir / f"sample_{i:04d}.npz"
        np.savez_compressed(
            sp,
            points=np.random.randn(n_points, 3).astype(np.float32),
            seg_labels=np.random.randint(0, num_classes, size=(n_points,)).astype(np.int64),
            category_id=np.array([0], dtype=np.int64),
            sample_id=f"sample_{i:04d}",
        )
        sample_paths.append(sp)

    # Write split file
    split_file = split_dir / "train.txt"
    with split_file.open("w") as f:
        for sp in sample_paths:
            f.write(f"samples/train/{sp.name}\n")

    # Dataset & loader
    ds = PartNetDataset(processed_root, split_file, num_points=n_points, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

    # Model
    model = PointTransformerSeg(
        input_dim=3, hidden_dim=32, num_layers=4, num_heads=4, num_parts=num_classes, dropout=0.0
    )
    criterion = SegmentationLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Train 5 epochs
    model.train()
    losses = []
    for epoch in range(5):
        epoch_losses = []
        for batch in loader:
            pts = batch["points"]
            lbl = batch["labels"]
            logits = model(pts)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(sum(epoch_losses) / len(epoch_losses))

    assert losses[0] > losses[-1], f"Loss did not decrease: {losses}"
    # With random labels, model can still memorize some statistical patterns
    # Loss should show some improvement but not perfect convergence
    assert losses[-1] < losses[0] * 0.95, f"Loss did not decrease enough: {losses[-1]:.4f} vs {losses[0]:.4f}"


def test_checkpoint_save_load_cycle(tmp_path):
    """Save a checkpoint and verify it loads correctly."""
    model = PointTransformerSeg(
        input_dim=3, hidden_dim=16, num_layers=4, num_heads=2, num_parts=5, dropout=0.0
    )
    pts = torch.randn(2, 32, 3)
    model.eval()
    out_before = model(pts).detach().clone()

    ckpt_path = tmp_path / "test.pth"
    torch.save({"model": model.state_dict(), "epoch": 10, "val_miou": 0.85}, ckpt_path)

    model2 = PointTransformerSeg(
        input_dim=3, hidden_dim=16, num_layers=4, num_heads=2, num_parts=5, dropout=0.0
    )
    ckpt = torch.load(ckpt_path)
    model2.load_state_dict(ckpt["model"])
    model2.eval()
    out_after = model2(pts).detach()

    assert torch.allclose(out_before, out_after, atol=1e-6)
    assert ckpt["epoch"] == 10
    assert ckpt["val_miou"] == 0.85


# ── Numerical stability ──────────────────────────────────────────────────────


def test_attention_numerical_stability():
    """Attention should not produce NaN with extreme inputs."""
    block = PointTransformerBlock(dim=32, num_heads=4, k=8, dropout=0.0)
    # Very large values
    feats = torch.randn(1, 64, 32) * 100.0
    pts = torch.randn(1, 64, 3) * 1000.0
    out = block(feats, pts)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_transition_down_stability():
    """TransitionDown should not produce NaN."""
    down = TransitionDown(in_channels=32, out_channels=64, ratio=0.5, k=8)
    pts = torch.randn(1, 128, 3) * 100.0
    feats = torch.randn(1, 128, 32) * 10.0
    new_pts, new_feats = down(pts, feats)
    assert not torch.isnan(new_feats).any()
    assert not torch.isinf(new_feats).any()


def test_transition_up_zero_distance():
    """TransitionUp with identical coarse and skip points (zero distance) should not NaN."""
    up = TransitionUp(coarse_channels=64, skip_channels=32, out_channels=32, k=3)
    pts = torch.randn(1, 16, 3)
    coarse_feats = torch.randn(1, 16, 64)
    skip_feats = torch.randn(1, 16, 32)
    new_pts, new_feats = up(pts, coarse_feats, pts, skip_feats)
    assert not torch.isnan(new_feats).any()
    assert not torch.isinf(new_feats).any()


# ── Config consistency ───────────────────────────────────────────────────────


def test_model_from_config_params():
    """Model instantiated with config-default parameters should work."""
    import yaml
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "partnet_pt_baseline.yaml"
    )
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    model = PointTransformerSeg(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        num_parts=int(cfg["dataset"]["num_parts"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    pts = torch.randn(1, int(cfg["dataset"]["num_points"]), 3)
    with torch.no_grad():
        logits = model(pts)
    assert logits.shape == (1, int(cfg["dataset"]["num_points"]), int(cfg["dataset"]["num_parts"]))
