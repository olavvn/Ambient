"""전체 파이프라인 통합 테스트"""
import pytest
import torch
import pretty_midi
from src.theme.tokenizer import REMIAmbientTokenizer
from src.theme.extractor import ThemeExtractor


def make_test_midi(n_notes: int = 16, duration_sec: float = 8.0) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    piano = pretty_midi.Instrument(program=0)
    dt = duration_sec / n_notes
    for i in range(n_notes):
        pitch = 60 + (i % 12)
        note = pretty_midi.Note(velocity=80, pitch=pitch,
                                start=i * dt, end=i * dt + dt * 0.9)
        piano.notes.append(note)
    midi.instruments.append(piano)
    return midi


def test_full_pipeline_midi_to_theme():
    """MIDI → 토큰화 → 테마 추출 전 과정"""
    extractor = ThemeExtractor(encoder_checkpoint=None, device="cpu")
    midi = make_test_midi(32, 16.0)
    theme_tokens, spans = extractor.extract_from_midi(midi)

    assert isinstance(theme_tokens, list)
    assert len(theme_tokens) > 0
    assert all(isinstance(t, int) for t in theme_tokens)
    print(f"\nTheme tokens: {len(theme_tokens)}, Spans: {len(spans)}")


def test_tokenizer_theme_marker_pipeline():
    """테마 구간 마커가 포함된 전체 토큰 생성"""
    tokenizer = REMIAmbientTokenizer()
    midi = make_test_midi(16, 8.0)
    theme_spans = [(0.0, 2.0), (4.0, 6.0)]
    tokens = tokenizer.midi_to_tokens(midi, theme_spans=theme_spans)
    decoded = tokenizer.decode(tokens)
    theme_ref_count = decoded.count("[THEME_REF]")
    assert theme_ref_count > 0
    print(f"\nTheme refs: {theme_ref_count}")


def test_model_forward_with_theme():
    """모델 포워드 패스: 테마 → 생성"""
    from src.model.config import ModelConfig
    from src.model.transformer import AmbientFlowModel

    cfg = ModelConfig(
        vocab_size=238, d_model=64, max_seq_len=32,
        theme_encoder_layers=1, theme_encoder_heads=2,
        max_theme_len=16, decoder_layers=1, decoder_heads=2,
        ff_dim=128, gpa_heads=2,
    )
    model = AmbientFlowModel(cfg)
    theme = torch.randint(0, cfg.vocab_size, (1, 16))
    context = torch.tensor([[1]])  # BOS
    new_tokens = model.generate_chunk(theme, context, n_new_tokens=8)
    assert new_tokens.shape == (1, 8)
