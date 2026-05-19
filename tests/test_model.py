"""모델 포워드 패스 및 생성 테스트"""
import pytest
import torch
from src.model.config import ModelConfig
from src.model.transformer import AmbientFlowModel
from src.model.gated_attention import GPAModule


@pytest.fixture
def small_cfg():
    return ModelConfig(
        vocab_size=238, d_model=64, max_seq_len=64,
        theme_encoder_layers=1, theme_encoder_heads=2,
        max_theme_len=32, decoder_layers=2, decoder_heads=2,
        ff_dim=128, gpa_heads=2,
    )


@pytest.fixture
def model(small_cfg):
    return AmbientFlowModel(small_cfg)


def test_gpa_forward_shape():
    gpa = GPAModule(d_model=64, n_heads=4, n_theme_heads=4)
    x = torch.randn(2, 10, 64)
    theme_ctx = torch.randn(2, 8, 64)
    out = gpa(x, theme_ctx)
    assert out.shape == (2, 10, 64)


def test_gpa_gate_effect():
    gpa = GPAModule(d_model=64, n_heads=4, n_theme_heads=4, gate_bias=-100.0)
    x = torch.randn(1, 4, 64)
    theme = torch.randn(1, 4, 64)
    out_closed = gpa(x, theme)
    assert out_closed.dtype == torch.float32


def test_model_forward(model, small_cfg):
    B, T, L = 2, 16, 32
    theme = torch.randint(0, small_cfg.vocab_size, (B, T))
    target = torch.randint(0, small_cfg.vocab_size, (B, L))
    logits = model(theme, target)
    assert logits.shape == (B, L, small_cfg.vocab_size)


def test_model_generate_chunk(model, small_cfg):
    theme = torch.randint(0, small_cfg.vocab_size, (1, 16))
    context = torch.tensor([[small_cfg.vocab_size - 1]])
    new_tokens = model.generate_chunk(theme, context, n_new_tokens=10)
    assert new_tokens.shape == (1, 10)
    assert (new_tokens >= 0).all()
    assert (new_tokens < small_cfg.vocab_size).all()


def test_model_parameter_count(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    assert n_params < 5_000_000
