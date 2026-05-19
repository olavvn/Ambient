"""테마 추출기 테스트 (CPU, 미학습 모델로도 동작 확인)"""
import pytest
import torch
import pretty_midi
import numpy as np
from src.theme.extractor import ThemeExtractor
from src.theme.contrastive import SegmentEncoder, NTXentLoss
from src.theme.clustering import cluster_segments, ThemeCandidate


@pytest.fixture
def extractor():
    return ThemeExtractor(encoder_checkpoint=None, device="cpu")


@pytest.fixture
def sample_midi():
    """반복 패턴이 있는 테스트 MIDI"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    piano = pretty_midi.Instrument(program=0)
    pattern = [(60, 0.5), (62, 0.5), (64, 0.5), (65, 0.5)]
    for rep in range(3):
        for i, (pitch, dur) in enumerate(pattern):
            t = rep * 2.0 + i * 0.5
            piano.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch, start=t, end=t + dur * 0.9
            ))
    midi.instruments.append(piano)
    return midi


def test_extractor_returns_tokens(extractor, sample_midi):
    theme_tokens, spans = extractor.extract_from_midi(sample_midi)
    assert isinstance(theme_tokens, list)
    assert len(theme_tokens) > 0


def test_extractor_tokens_valid_ids(extractor, sample_midi):
    theme_tokens, _ = extractor.extract_from_midi(sample_midi)
    assert all(0 <= t < extractor.tokenizer.vocab_size for t in theme_tokens)


def test_contrastive_loss_shape():
    loss_fn = NTXentLoss(temperature=0.07)
    z1 = torch.randn(8, 128)
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.randn(8, 128)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    loss = loss_fn(z1, z2)
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_clustering_returns_candidates():
    embeddings = np.random.randn(20, 64).astype(np.float32)
    candidates = cluster_segments(embeddings, method="kmeans", n_clusters=3)
    assert len(candidates) >= 1
    assert all(hasattr(c, "score") for c in candidates)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
