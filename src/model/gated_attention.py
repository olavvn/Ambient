"""
Gated Parallel Attention (GPA) Module
ThemeTransformer 논문 Figure 3 구현

일반 self-attention 출력과 theme cross-attention 출력을
학습 가능한 sigmoid gate로 혼합.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GPAModule(nn.Module):
    """
    Gated Parallel Attention:
        output = self_attn_out + sigmoid(gate) * theme_cross_attn_out

    Args:
        d_model: 모델 차원
        n_heads: self-attention 헤드 수
        n_theme_heads: theme cross-attention 헤드 수
        dropout: 드롭아웃 비율
        gate_bias: 게이트 초기 바이어스 (음수 → 초기에 테마 영향 낮게)
    """

    def __init__(self, d_model: int, n_heads: int, n_theme_heads: int,
                 dropout: float = 0.1, gate_bias: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert d_model % n_theme_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_theme_heads = n_theme_heads
        self.head_dim = d_model // n_heads
        self.theme_head_dim = d_model // n_theme_heads

        # Causal Self-Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Theme Cross-Attention
        self.tq_proj = nn.Linear(d_model, d_model, bias=False)
        self.tk_proj = nn.Linear(d_model, d_model, bias=False)
        self.tv_proj = nn.Linear(d_model, d_model, bias=False)
        self.tout_proj = nn.Linear(d_model, d_model, bias=False)

        # Gate
        self.gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate.bias, gate_bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.theme_scale = math.sqrt(self.theme_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        theme_ctx: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        theme_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        # 1. Causal Self-Attention
        q = self._split_heads(self.q_proj(x), self.n_heads, self.head_dim)
        k = self._split_heads(self.k_proj(x), self.n_heads, self.head_dim)
        v = self._split_heads(self.v_proj(x), self.n_heads, self.head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if causal_mask is not None:
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        self_out = torch.matmul(attn, v)
        self_out = self._merge_heads(self_out)
        self_out = self.out_proj(self_out)

        # 2. Theme Cross-Attention
        tq = self._split_heads(self.tq_proj(x), self.n_theme_heads, self.theme_head_dim)
        tk = self._split_heads(self.tk_proj(theme_ctx), self.n_theme_heads, self.theme_head_dim)
        tv = self._split_heads(self.tv_proj(theme_ctx), self.n_theme_heads, self.theme_head_dim)

        t_attn = torch.matmul(tq, tk.transpose(-2, -1)) / self.theme_scale
        if theme_key_padding_mask is not None:
            t_attn = t_attn.masked_fill(
                theme_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        t_attn = F.softmax(t_attn, dim=-1)
        t_attn = self.dropout(t_attn)
        theme_out = torch.matmul(t_attn, tv)
        theme_out = self._merge_heads(theme_out)
        theme_out = self.tout_proj(theme_out)

        # 3. Gated Mixing
        gate = torch.sigmoid(self.gate(x))
        return self_out + gate * theme_out

    def _split_heads(self, x: torch.Tensor, n_heads: int,
                     head_dim: int) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, n_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)
