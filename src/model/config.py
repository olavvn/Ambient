"""모델 하이퍼파라미터 설정"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 238
    d_model: int = 512
    max_seq_len: int = 1024

    theme_encoder_layers: int = 4
    theme_encoder_heads: int = 8
    max_theme_len: int = 128

    decoder_layers: int = 8
    decoder_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1

    gpa_heads: int = 8
    gpa_gate_bias: float = 0.0

    temperature: float = 0.95
    top_p: float = 0.92
    repetition_penalty: float = 1.05

    label_smoothing: float = 0.1
