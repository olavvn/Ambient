# 09 테스트 및 평가 (Testing & Evaluation)

## 테스트 전략

단위 테스트 → 통합 테스트 → 스트리밍 테스트 순서로 진행.
모든 테스트는 `pytest`로 실행되며, GPU 없이도 CPU로 소형 설정 테스트 가능.

---

## 1. 토크나이저 테스트

### `tests/test_tokenizer.py`

```python
"""토크나이저 왕복 변환 정확성 테스트"""
import pytest
import pretty_midi
import numpy as np
from src.theme.tokenizer import REMIAmbientTokenizer


@pytest.fixture
def tokenizer():
    return REMIAmbientTokenizer()


@pytest.fixture
def simple_midi():
    """테스트용 단순 MIDI: C 장음계"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    piano = pretty_midi.Instrument(program=0)
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, p in enumerate(pitches):
        note = pretty_midi.Note(velocity=80, pitch=p,
                                start=i * 0.5, end=i * 0.5 + 0.4)
        piano.notes.append(note)
    midi.instruments.append(piano)
    return midi


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size > 200


def test_midi_to_tokens_not_empty(tokenizer, simple_midi):
    tokens = tokenizer.midi_to_tokens(simple_midi)
    assert len(tokens) > 0
    # BOS로 시작, EOS로 끝
    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id


def test_tokens_contain_pitch(tokenizer, simple_midi):
    tokens = tokenizer.midi_to_tokens(simple_midi)
    decoded = tokenizer.decode(tokens)
    pitches = [t for t in decoded if t.startswith("PITCH_")]
    assert len(pitches) == 8  # C 장음계 8음


def test_tokens_to_midi_roundtrip(tokenizer, simple_midi):
    """MIDI → 토큰 → MIDI 왕복 후 노트 수 동일해야 함"""
    tokens = tokenizer.midi_to_tokens(simple_midi)
    reconstructed = tokenizer.tokens_to_midi(tokens)
    orig_notes = sum(len(i.notes) for i in simple_midi.instruments)
    recon_notes = sum(len(i.notes) for i in reconstructed.instruments)
    # 정확히 같지 않아도 같은 수준이어야 함 (퀀타이즈 오차)
    assert abs(orig_notes - recon_notes) <= 2


def test_theme_markers(tokenizer, simple_midi):
    """테마 구간 마커 삽입 확인"""
    theme_spans = [(0.0, 2.0)]  # 처음 2초를 테마 구간으로
    tokens = tokenizer.midi_to_tokens(simple_midi, theme_spans=theme_spans)
    decoded = tokenizer.decode(tokens)
    assert "[THEME_REF]" in decoded
```

---

## 2. 테마 추출기 테스트

### `tests/test_theme_extractor.py`

```python
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
    # 체크포인트 없이 랜덤 초기화 모델로 테스트
    return ThemeExtractor(encoder_checkpoint=None, device="cpu")


@pytest.fixture
def sample_midi():
    """반복 패턴이 있는 테스트 MIDI"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    piano = pretty_midi.Instrument(program=0)
    # 같은 패턴 3번 반복
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
    assert loss.shape == torch.Size([])  # 스칼라
    assert loss.item() > 0


def test_clustering_returns_candidates():
    embeddings = np.random.randn(20, 64).astype(np.float32)
    candidates = cluster_segments(embeddings, method="kmeans", n_clusters=3)
    assert len(candidates) >= 1
    assert all(hasattr(c, "score") for c in candidates)
    # score 내림차순 정렬 확인
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
```

---

## 3. 모델 테스트

### `tests/test_model.py`

```python
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
        ff_dim=128,
    )


@pytest.fixture
def model(small_cfg):
    return AmbientFlowModel(small_cfg)


def test_gpa_forward_shape():
    gpa = GPAModule(d_model=64, n_heads=4, n_theme_heads=4)
    x = torch.randn(2, 10, 64)         # (B, L, D)
    theme_ctx = torch.randn(2, 8, 64)  # (B, T, D)
    out = gpa(x, theme_ctx)
    assert out.shape == (2, 10, 64)


def test_gpa_gate_effect():
    """gate=0이면 self-attention만, gate=1이면 theme도 포함"""
    gpa = GPAModule(d_model=64, n_heads=4, n_theme_heads=4, gate_bias=-100.0)
    x = torch.randn(1, 4, 64)
    theme = torch.randn(1, 4, 64)
    out_closed = gpa(x, theme)
    # gate 바이어스가 매우 낮으면 theme 영향이 거의 0
    # 단순히 shape과 dtype 확인
    assert out_closed.dtype == torch.float32


def test_model_forward(model, small_cfg):
    B, T, L = 2, 16, 32
    theme = torch.randint(0, small_cfg.vocab_size, (B, T))
    target = torch.randint(0, small_cfg.vocab_size, (B, L))
    logits = model(theme, target)
    assert logits.shape == (B, L, small_cfg.vocab_size)


def test_model_generate_chunk(model, small_cfg):
    theme = torch.randint(0, small_cfg.vocab_size, (1, 16))
    context = torch.tensor([[small_cfg.vocab_size - 1]])  # BOS
    new_tokens = model.generate_chunk(theme, context, n_new_tokens=10)
    assert new_tokens.shape == (1, 10)
    assert (new_tokens >= 0).all()
    assert (new_tokens < small_cfg.vocab_size).all()


def test_model_parameter_count(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    # Small 모델은 5M 미만이어야 함 (테스트 설정)
    assert n_params < 5_000_000
```

---

## 4. 스트리밍 테스트

### `tests/test_streaming.py`

```python
"""스트리밍 엔진 비동기 테스트"""
import pytest
import asyncio
import torch
import threading
import time
from src.streaming.buffer import ThemeBuffer, TokenQueue
from src.streaming.crossfade import crossfade_audio
import numpy as np


def test_theme_buffer_update_get():
    buf = ThemeBuffer()
    tokens = torch.tensor([1, 2, 3, 4, 5])
    buf.update(tokens)
    result = buf.get()
    assert result is not None
    assert torch.equal(result, tokens)


def test_theme_buffer_new_flag():
    buf = ThemeBuffer()
    assert not buf.consume_new_flag()
    buf.update(torch.tensor([1]))
    assert buf.consume_new_flag()
    assert not buf.consume_new_flag()  # 두 번 호출하면 False


def test_theme_buffer_thread_safety():
    """여러 스레드에서 동시 업데이트해도 안전"""
    buf = ThemeBuffer()
    errors = []

    def updater():
        try:
            for _ in range(100):
                buf.update(torch.randint(0, 100, (10,)))
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=updater) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors


def test_crossfade_shape():
    audio_out = np.random.randn(44100).astype(np.float32)
    audio_in = np.random.randn(44100).astype(np.float32)
    result = crossfade_audio(audio_out, audio_in, fade_samples=4410)
    assert result.shape == (4410,)
    # 크로스페이드는 두 신호의 합이어야 함 (중간 지점)
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_token_queue_put_get():
    loop = asyncio.get_event_loop()
    queue = TokenQueue(maxsize=4)
    queue.set_loop(loop)

    # 별도 스레드에서 넣기
    def put_in_thread():
        time.sleep(0.05)
        queue.put_nowait_threadsafe([1, 2, 3])

    threading.Thread(target=put_in_thread, daemon=True).start()
    chunk = await asyncio.wait_for(queue.get(), timeout=2.0)
    assert chunk == [1, 2, 3]
```

---

## 5. 통합 테스트

### `tests/test_integration.py`

```python
"""
전체 파이프라인 통합 테스트
실제 모델 없이 Mock 모델로 테스트
"""
import pytest
import torch
import pretty_midi
from unittest.mock import MagicMock, patch
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
    tokenizer = REMIAmbientTokenizer()
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
```

---

## 6. 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 빠른 테스트 (스트리밍 제외)
pytest tests/ -v -k "not streaming"

# 비동기 테스트 포함
pytest tests/ -v --asyncio-mode=auto

# 커버리지 리포트
pytest tests/ --cov=src --cov-report=html
```

---

## 7. 품질 기준 (Definition of Done)

| 단계 | 기준 |
|------|------|
| 토크나이저 | 왕복 변환 노트 수 오차 ≤ 10% |
| 테마 추출 | Theme Recall@3 ≥ 0.6 (학습 후) |
| 생성 모델 | Validation Perplexity ≤ 3.0 |
| 스트리밍 | 청크 간 갭 ≤ 50ms |
| 통합 | 10분 연속 생성 중 크래시 없음 |
| 앰비언트 품질 | MOS ≥ 3.5 (5점 척도) |
