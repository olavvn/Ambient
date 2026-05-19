"""생성 모델 손실 함수"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThemeAwareCrossEntropy(nn.Module):
    """
    테마 토큰 위치에서 추가 가중치를 주는 Cross Entropy.
    [THEME_REF] 토큰 직후 예측에 더 높은 가중치 부여 →
    모델이 테마 참조 구간에서 더 정확하게 학습.
    """

    def __init__(self, vocab_size: int, label_smoothing: float = 0.1,
                 theme_ref_id: int = 6, theme_weight: float = 2.0,
                 pad_id: int = 0):
        super().__init__()
        self.theme_ref_id = theme_ref_id
        self.theme_weight = theme_weight
        self.pad_id = pad_id
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=pad_id,
            reduction='none',
        )

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, L, V)
            targets: (B, L)
        """
        B, L, V = logits.shape
        loss = self.ce(logits.view(B * L, V), targets.view(B * L))
        loss = loss.view(B, L)

        theme_ref_mask = (targets[:, :-1] == self.theme_ref_id).float()
        weight = torch.ones_like(loss)
        weight[:, 1:] += (self.theme_weight - 1) * theme_ref_mask

        pad_mask = (targets != self.pad_id).float()
        loss = (loss * weight * pad_mask).sum() / pad_mask.sum().clamp(min=1)
        return loss
