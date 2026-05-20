import torch

from utils.metrics.segmentation_metrics import compute_segmentation_metrics


def test_metrics_perfect_prediction():
    logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]])
    labels = torch.tensor([[1, 0]])

    metrics = compute_segmentation_metrics(logits, labels, num_parts=2, ignore_index=-1)

    assert metrics["overall_acc"] == 1.0
    assert metrics["miou"] == 1.0
