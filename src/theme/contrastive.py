"""
대조 학습 기반 음악 세그먼트 인코더
유사 패시지 → 임베딩 공간에서 가깝게
다른 패시지 → 멀게
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentEncoder(nn.Module):
    """
    음악 세그먼트를 고정 크기 벡터로 인코딩하는 Transformer 인코더.
    ThemeTransformer 논문의 contrastive encoder 구조를 따름.
    """

    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4,
                 max_seq_len: int = 128, proj_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L) 토큰 ID 시퀀스
            padding_mask: (B, L) True = padding 위치
        Returns:
            z: (B, proj_dim) 정규화된 임베딩
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        z = self.proj_head(h)
        return F.normalize(z, dim=-1)


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실.
    같은 원곡의 다른 구간 → positive pair,
    다른 원곡의 구간 → negative pair.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) 같은 원곡에서 나온 positive pair
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)
