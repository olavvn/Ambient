"""표준 Cross-Entropy 손실"""
import torch.nn as nn


def make_loss_fn(pad_id: int = 0, label_smoothing: float = 0.1) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
