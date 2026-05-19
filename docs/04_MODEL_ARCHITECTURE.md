# 04 모델 아키텍처 (Model Architecture)

## 개요

ThemeTransformer의 **Gated Parallel Attention (GPA)** 모듈을 핵심으로 하는 seq2seq Transformer. 테마 인코더가 테마 토큰을 컨텍스트 벡터로 압축하고, 디코더의 GPA 모듈이 자기회귀 생성 중 항상 테마를 참조한다.

---

## 전체 모델 구조

```
┌─────────────────────────────────────────────────────┐
│                    ThemeEncoder                      │
│  theme_tokens → [Transformer Encoder] → theme_ctx   │
│                         (B, T_theme, D)              │
└─────────────────────────────────────────────────────┘
                          │ theme_ctx
                          ▼
┌─────────────────────────────────────────────────────┐
│              AmbientMusicDecoder                     │
│                                                      │
│  [token_emb] → ┌────────────────────────────────┐   │
│                 │  N × DecoderLayer              │   │
│                 │                                │   │
│                 │  ┌──────────────────────────┐  │   │
│                 │  │   Causal Self-Attention   │  │   │
│                 │  └─────────────┬────────────┘  │   │
│                 │                │                │   │
│                 │  ┌─────────────▼────────────┐  │   │
│                 │  │  Gated Parallel Attention │  │   │
│                 │  │  (Theme Cross-Attention)  │  │   │
│                 │  └─────────────┬────────────┘  │   │
│                 │                │                │   │
│                 │  ┌─────────────▼────────────┐  │   │
│                 │  │      Feed Forward         │  │   │
│                 │  └──────────────────────────┘  │   │
│                 └────────────────────────────────┘   │
│                          │                           │
│                    [Linear + Softmax]                 │
│                          │                           │
│                   next_token logits                  │
└─────────────────────────────────────────────────────┘
```

---

## 1. 설정 파일

### `src/model/config.py`

```python
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    # ── 공통 ──────────────────────────────────
    vocab_size: int = 238
    d_model: int = 512
    max_seq_len: int = 1024        # 생성 컨텍스트 윈도우

    # ── 테마 인코더 ──────────────────────────
    theme_encoder_layers: int = 4
    theme_encoder_heads: int = 8
    max_theme_len: int = 128       # 테마 토큰 최대 길이

    # ── 디코더 ───────────────────────────────
    decoder_layers: int = 8
    decoder_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1

    # ── GPA 설정 ─────────────────────────────
    gpa_heads: int = 8             # theme cross-attention 헤드
    gpa_gate_bias: float = 0.0    # 초기 게이트 바이어스

    # ── 앰비언트 생성 파라미터 ────────────────
    temperature: float = 0.95      # 부드러운 분포
    top_p: float = 0.92
    repetition_penalty: float = 1.05

    # ── 훈련 ──────────────────────────────────
    label_smoothing: float = 0.1
```

---

## 2. Gated Parallel Attention (GPA) 모듈

ThemeTransformer 논문의 핵심 기여. 기존 cross-attention에 **학습 가능한 게이트**를 추가하여, 디코더가 매 토큰마다 테마 참조 강도를 동적으로 조절한다.

### `src/model/gated_attention.py`

```python
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

        # ── Causal Self-Attention ───────────────────────────────────────
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ── Theme Cross-Attention ───────────────────────────────────────
        self.tq_proj = nn.Linear(d_model, d_model, bias=False)
        self.tk_proj = nn.Linear(d_model, d_model, bias=False)
        self.tv_proj = nn.Linear(d_model, d_model, bias=False)
        self.tout_proj = nn.Linear(d_model, d_model, bias=False)

        # ── Gate ────────────────────────────────────────────────────────
        # 위치별 스칼라 게이트 (d_model 크기)
        self.gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate.bias, gate_bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.theme_scale = math.sqrt(self.theme_head_dim)

    def forward(
        self,
        x: torch.Tensor,                # (B, L, D) 디코더 입력
        theme_ctx: torch.Tensor,         # (B, T, D) 테마 컨텍스트
        causal_mask: torch.Tensor = None, # (L, L) 인과 마스크
        theme_key_padding_mask: torch.Tensor = None,  # (B, T)
    ) -> torch.Tensor:
        B, L, D = x.shape

        # ── 1. Causal Self-Attention ────────────────────────────────────
        q = self._split_heads(self.q_proj(x), self.n_heads, self.head_dim)
        k = self._split_heads(self.k_proj(x), self.n_heads, self.head_dim)
        v = self._split_heads(self.v_proj(x), self.n_heads, self.head_dim)
        # q, k, v: (B, H, L, head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if causal_mask is not None:
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        self_out = torch.matmul(attn, v)
        self_out = self._merge_heads(self_out)
        self_out = self.out_proj(self_out)           # (B, L, D)

        # ── 2. Theme Cross-Attention ────────────────────────────────────
        tq = self._split_heads(self.tq_proj(x), self.n_theme_heads, self.theme_head_dim)
        tk = self._split_heads(self.tk_proj(theme_ctx), self.n_theme_heads, self.theme_head_dim)
        tv = self._split_heads(self.tv_proj(theme_ctx), self.n_theme_heads, self.theme_head_dim)

        t_attn = torch.matmul(tq, tk.transpose(-2, -1)) / self.theme_scale
        if theme_key_padding_mask is not None:
            t_attn = t_attn.masked_fill(
                theme_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        t_attn = F.softmax(t_attn, dim=-1)
        t_attn = self.dropout(t_attn)
        theme_out = torch.matmul(t_attn, tv)
        theme_out = self._merge_heads(theme_out)
        theme_out = self.tout_proj(theme_out)        # (B, L, D)

        # ── 3. Gated Mixing ─────────────────────────────────────────────
        gate = torch.sigmoid(self.gate(x))            # (B, L, D)
        output = self_out + gate * theme_out
        return output

    def _split_heads(self, x: torch.Tensor, n_heads: int,
                     head_dim: int) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, n_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)
```

---

## 3. 디코더 레이어

### `src/model/decoder.py`

```python
"""앰비언트 음악 디코더 - GPA 모듈 통합"""
import torch
import torch.nn as nn
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

    def forward(self, x, theme_ctx, causal_mask=None,
                theme_pad_mask=None):
        # Pre-LN: LayerNorm 먼저
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
        # Weight tying (선택)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        token_ids: torch.Tensor,       # (B, L)
        theme_ctx: torch.Tensor,        # (B, T, D) 테마 컨텍스트
        theme_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns: logits (B, L, vocab_size)"""
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)

        # Causal mask
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )

        for layer in self.layers:
            x = layer(x, theme_ctx, causal_mask, theme_pad_mask)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits
```

---

## 4. 전체 Theme-Conditioned Transformer

### `src/model/transformer.py`

```python
"""AmbientFlowModel: 테마 인코더 + 앰비언트 디코더 통합"""
import torch
import torch.nn as nn
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
            encoder_layer, num_layers=cfg.theme_encoder_layers
        )

    def forward(self, theme_tokens: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
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
        theme_tokens: torch.Tensor,   # (B, T)
        target_tokens: torch.Tensor,  # (B, L)
        theme_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """훈련 시 전체 포워드 패스. Returns: logits (B, L, V)"""
        theme_ctx = self.theme_encoder(theme_tokens, theme_pad_mask)
        logits = self.decoder(target_tokens, theme_ctx, theme_pad_mask)
        return logits

    @torch.no_grad()
    def generate_chunk(
        self,
        theme_tokens: torch.Tensor,   # (1, T)
        context: torch.Tensor,         # (1, C) 이전 생성 컨텍스트
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
            # 컨텍스트 윈도우 유지
            ctx_window = generated[:, -self.cfg.max_seq_len:]
            logits = self.decoder(ctx_window, theme_ctx)
            next_logits = logits[:, -1, :]  # 마지막 위치

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0].tolist():
                    next_logits[0, token_id] /= repetition_penalty

            # Temperature + top-p sampling
            next_token = _sample_top_p(next_logits, temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, context.size(1):]  # 새로 생성된 부분만


def _sample_top_p(logits: torch.Tensor, temperature: float,
                  top_p: float) -> torch.Tensor:
    """Top-p (nucleus) sampling"""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # top-p 경계 이후 제거
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    next_id = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids.gather(-1, next_id)
```

---

## 5. 모델 크기 요약

| 변형 | 파라미터 수 | 용도 |
|------|------------|------|
| AmbientFlow-Small | ~30M | 개발/테스트, CPU 실행 가능 |
| AmbientFlow-Base | ~120M | 기본 배포 (GPU 권장) |
| AmbientFlow-Large | ~350M | 최고 품질 (고사양 GPU) |

### Small 설정 (`configs/model_config.yaml`)
```yaml
vocab_size: 238
d_model: 256
max_seq_len: 512
theme_encoder_layers: 2
theme_encoder_heads: 4
max_theme_len: 64
decoder_layers: 4
decoder_heads: 4
ff_dim: 1024
dropout: 0.1
gpa_heads: 4
gpa_gate_bias: -1.0   # 초기에는 테마 영향 낮게 시작
temperature: 0.95
top_p: 0.92
repetition_penalty: 1.05
```

### Base 설정
```yaml
vocab_size: 238
d_model: 512
max_seq_len: 1024
theme_encoder_layers: 4
theme_encoder_heads: 8
max_theme_len: 128
decoder_layers: 8
decoder_heads: 8
ff_dim: 2048
dropout: 0.1
gpa_heads: 8
gpa_gate_bias: -0.5
temperature: 0.95
top_p: 0.92
repetition_penalty: 1.05
```
