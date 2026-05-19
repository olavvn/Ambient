"""토크나이저 왕복 변환 정확성 테스트"""
import pytest
import pretty_midi
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
    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id


def test_tokens_contain_pitch(tokenizer, simple_midi):
    tokens = tokenizer.midi_to_tokens(simple_midi)
    decoded = tokenizer.decode(tokens)
    pitches = [t for t in decoded if t.startswith("PITCH_")]
    assert len(pitches) == 8


def test_tokens_to_midi_roundtrip(tokenizer, simple_midi):
    """MIDI → 토큰 → MIDI 왕복 후 노트 수 동일해야 함"""
    tokens = tokenizer.midi_to_tokens(simple_midi)
    reconstructed = tokenizer.tokens_to_midi(tokens)
    orig_notes = sum(len(i.notes) for i in simple_midi.instruments)
    recon_notes = sum(len(i.notes) for i in reconstructed.instruments)
    assert abs(orig_notes - recon_notes) <= 2


def test_theme_markers(tokenizer, simple_midi):
    """테마 구간 마커 삽입 확인"""
    theme_spans = [(0.0, 2.0)]
    tokens = tokenizer.midi_to_tokens(simple_midi, theme_spans=theme_spans)
    decoded = tokenizer.decode(tokens)
    assert "[THEME_REF]" in decoded


def test_encode_decode_roundtrip(tokenizer):
    token_strs = ["[BOS]", "BAR", "POS_0", "PITCH_60", "DUR_4", "VEL_64", "[EOS]"]
    ids = tokenizer.encode(token_strs)
    back = tokenizer.decode(ids)
    assert back == token_strs


def test_closest_dur_token(tokenizer):
    assert tokenizer._closest_dur_token(1) == "DUR_1"
    assert tokenizer._closest_dur_token(64) == "DUR_64"
    assert tokenizer._closest_dur_token(50) == "DUR_48"


def test_closest_vel_token(tokenizer):
    assert tokenizer._closest_vel_token(0) == "VEL_8"
    assert tokenizer._closest_vel_token(127) == "VEL_106"
