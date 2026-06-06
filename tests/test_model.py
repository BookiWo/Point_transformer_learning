import torch

from models.point_transformer_seg import PointTransformerSeg


def test_point_transformer_seg_forward_backward():
    model = PointTransformerSeg(
        input_dim=3,
        hidden_dim=32,
        num_layers=4,
        num_heads=4,
        num_parts=6,
        dropout=0.0,
    )
    points = torch.randn(2, 32, 3)
    labels = torch.randint(0, 6, (2, 32))

    logits = model(points)

    assert logits.shape == (2, 32, 6)

    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1))
    loss.backward()

    total_grad = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total_grad += float(parameter.grad.abs().sum().item())

    assert total_grad > 0.0