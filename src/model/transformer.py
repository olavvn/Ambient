"""AmbientFlowModel: 테마 인코더 + 앰비언트 디코더 통합"""
import torch
import torch.nn as nn
from typing import Optional
from src.model.config import ModelConfig
from src.model.decoder import AmbientMusicDecoder


class ThemeEncoder(nn.Module):
    """테마 토큰 시퀀스 → 컨텍스트 벡터 시퀀스"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_theme_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.theme_encoder_heads,
            dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.theme_encoder_layers, enable_nested_tensor=False
        )

    def forward(self, theme_tokens: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns: (B, T, D)"""
        B, T = theme_tokens.shape
        pos = torch.arange(T, device=theme_tokens.device)
        x = self.token_emb(theme_tokens) + self.pos_emb(pos)
        return self.encoder(x, src_key_padding_mask=padding_mask)


class AmbientFlowModel(nn.Module):
    """메인 생성 모델"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.theme_encoder = ThemeEncoder(cfg)
        self.decoder = AmbientMusicDecoder(cfg)

    def forward(
        self,
        theme_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        theme_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """훈련 시 전체 포워드 패스. Returns: logits (B, L, V)"""
        theme_ctx = self.theme_encoder(theme_tokens, theme_pad_mask)
        return self.decoder(target_tokens, theme_ctx, theme_pad_mask)

    @torch.no_grad()
    def generate_chunk(
        self,
        theme_tokens: torch.Tensor,
        context: torch.Tensor,
        n_new_tokens: int = 128,
        temperature: float = 0.95,
        top_p: float = 0.92,
        repetition_penalty: float = 1.05,
    ) -> torch.Tensor:
        """
        자기회귀 청크 생성.
        Returns: (1, n_new_tokens) 새로 생성된 토큰
        """
        theme_ctx = self.theme_encoder(theme_tokens)
        generated = context.clone()

        for _ in range(n_new_tokens):
            ctx_window = generated[:, -self.cfg.max_seq_len:]
            logits = self.decoder(ctx_window, theme_ctx)
            next_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in generated[0].tolist():
                    next_logits[0, token_id] /= repetition_penalty

            next_token = _sample_top_p(next_logits, temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, context.size(1):]


def _sample_top_p(logits: torch.Tensor, temperature: float,
                  top_p: float) -> torch.Tensor:
    """Top-p (nucleus) sampling"""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    next_id = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids.gather(-1, next_id)
