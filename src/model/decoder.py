"""앰비언트 음악 디코더 - GPA 모듈 통합"""
import torch
import torch.nn as nn
from typing import Optional
from src.model.gated_attention import GPAModule
from src.model.config import ModelConfig


class DecoderLayer(nn.Module):
    """Pre-LN 구조의 디코더 레이어"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.gpa = GPAModule(
            d_model=cfg.d_model,
            n_heads=cfg.decoder_heads,
            n_theme_heads=cfg.gpa_heads,
            dropout=cfg.dropout,
            gate_bias=cfg.gpa_gate_bias,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        theme_ctx: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        theme_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.gpa(x, theme_ctx, causal_mask, theme_pad_mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        return x


class AmbientMusicDecoder(nn.Module):
    """앰비언트 뮤직 자기회귀 디코더"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(cfg) for _ in range(cfg.decoder_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        token_ids: torch.Tensor,
        theme_ctx: torch.Tensor,
        theme_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns: logits (B, L, vocab_size)"""
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )

        for layer in self.layers:
            x = layer(x, theme_ctx, causal_mask, theme_pad_mask)

        x = self.norm_out(x)
        return self.lm_head(x)
