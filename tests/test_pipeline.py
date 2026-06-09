"""
통합 테스트: AnchorFlow 파이프라인 검증.
실제 모델 가중치 없이 더미 데이터로 구조 검증.
"""
import os
import sys
import pickle
import tempfile
import torch
import pytest
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import TSDTokenizer
from src.streaming.buffer import AnchorQueue, TokenQueue
from training.dataset import AmbientMIDIDataset
from training.losses import make_loss_fn


# ── 토크나이저 ─────────────────────────────────────────────────────────────

def test_tokenizer_vocab_size():
    tok = TSDTokenizer()
    assert tok.vocab_size == 274
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2


def test_tokenizer_roundtrip():
    tok = TSDTokenizer()
    midi = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=60, start=0.0, end=1.0))
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=64, start=1.0, end=2.0))
    midi.instruments.append(inst)

    tokens = tok.midi_to_tokens(midi)
    assert len(tokens) > 0
    assert tokens[0] == tok.bos_id
    assert tokens[-1] == tok.eos_id

    reconstructed = tok.tokens_to_midi(tokens)
    assert len(reconstructed.instruments) > 0


# ── 큐 ────────────────────────────────────────────────────────────────────

def test_anchor_queue():
    aq = AnchorQueue()
    assert not aq.has_pending()

    aq.push([1, 2, 3])
    aq.push([4, 5])
    assert aq.has_pending()

    result = aq.pop_all()
    assert result == [[1, 2, 3], [4, 5]]
    assert not aq.has_pending()


def test_token_queue():
    tq = TokenQueue(maxsize=4)
    tq.put([10, 20, 30])
    result = tq.get(timeout=1.0)
    assert result == [10, 20, 30]


# ── 데이터셋 ──────────────────────────────────────────────────────────────

def make_dummy_pkl_dir(n_files: int = 3, tokens_per_file: int = 3000):
    tmpdir = tempfile.mkdtemp()
    for i in range(n_files):
        path = os.path.join(tmpdir, f"track_{i:03d}.pkl")
        tokens = list(range(2, 2 + tokens_per_file))  # 더미 토큰 (0=PAD, 1=BOS 제외)
        with open(path, "wb") as f:
            pickle.dump({"tokens": tokens}, f)
    return tmpdir


def test_ambient_midi_dataset():
    data_dir = make_dummy_pkl_dir(n_files=3, tokens_per_file=3000)
    ds = AmbientMIDIDataset(data_dir, seq_len=512, stride=256, pad_id=0, bos_id=1)

    assert len(ds) > 0
    sample = ds[0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape == (511,)
    assert sample["labels"].shape == (511,)


# ── 손실 함수 ─────────────────────────────────────────────────────────────

def test_loss_fn():
    loss_fn = make_loss_fn(pad_id=0, label_smoothing=0.1)
    logits = torch.randn(2, 10, 274)
    labels = torch.randint(0, 274, (2, 10))
    labels[0, -2:] = 0  # PAD
    loss = loss_fn(logits.view(-1, 274), labels.view(-1))
    assert loss.item() > 0


# ── 모델 구조 (mock) ──────────────────────────────────────────────────────

def test_anchorflow_model_structure():
    """AnchorFlowModel이 올바른 인터페이스를 노출하는지 검증 (가중치 다운로드 없이)"""
    from unittest.mock import MagicMock, patch
    import torch.nn as nn

    # HF 모델을 mock으로 대체
    mock_hf = MagicMock()
    mock_hf.config.vocab_size = 274
    mock_embedding = nn.Embedding(274, 256)
    mock_hf.get_input_embeddings.return_value = mock_embedding
    mock_hf.get_output_embeddings.return_value = nn.Linear(256, 274, bias=False)

    from src.model.anchorflow import AnchorFlowModel
    model = AnchorFlowModel(mock_hf)

    assert hasattr(model, "generate")
    assert hasattr(model, "forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
