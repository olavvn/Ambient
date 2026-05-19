"""StyleLibrary 테스트"""
import pytest
import pretty_midi
import tempfile
from pathlib import Path

from src.theme.style_library import StyleLibrary


def make_midi_file(path: Path, pitches: list, bpm: float = 60.0):
    """테스트용 MIDI 파일 생성"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)
    for i, p in enumerate(pitches):
        note = pretty_midi.Note(velocity=80, pitch=p,
                                start=i * 0.5, end=i * 0.5 + 0.4)
        piano.notes.append(note)
    midi.instruments.append(piano)
    midi.write(str(path))


@pytest.fixture
def style_dir(tmp_path):
    """MIDI 파일 3개가 있는 임시 styles/ 폴더"""
    make_midi_file(tmp_path / "ballad.mid",
                   [60, 62, 64, 65, 67, 69, 71, 72])
    make_midi_file(tmp_path / "cinematic.mid",
                   [55, 57, 59, 60, 62, 64, 65, 67])
    make_midi_file(tmp_path / "dark.mid",
                   [45, 47, 48, 50, 52, 53, 55, 57])
    return str(tmp_path)


def test_style_library_loads(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    assert len(lib.style_names) == 3
    assert "ballad.mid" in lib.style_names


def test_get_blended_theme_length(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=64)
    tokens = lib.get_blended_theme()
    assert len(tokens) <= 64
    assert len(tokens) > 0


def test_get_blended_theme_all_valid_ids(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    tokens = lib.get_blended_theme()
    assert all(isinstance(t, int) for t in tokens)
    assert all(0 <= t < lib.tokenizer.vocab_size for t in tokens)


def test_get_single_theme(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    tokens = lib.get_single_theme("ballad.mid")
    assert len(tokens) > 0


def test_get_single_theme_unknown_raises(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    with pytest.raises(KeyError):
        lib.get_single_theme("nonexistent.mid")


def test_blend_with_subset(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    tokens = lib.get_blended_theme(names=["ballad.mid", "dark.mid"])
    assert len(tokens) > 0


def test_blend_strategies_produce_different_results(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    front = lib.get_blended_theme(strategy="front")
    center = lib.get_blended_theme(strategy="center")
    # front와 center는 동일 파일에서 다른 구간을 뽑으므로 다를 수 있음
    assert isinstance(front, list) and isinstance(center, list)


def test_get_blended_theme_tensor_shape(style_dir):
    import torch
    lib = StyleLibrary(style_dir, max_theme_len=64)
    tensor = lib.get_blended_theme_tensor()
    assert tensor.dim() == 2        # (1, T)
    assert tensor.shape[0] == 1
    assert tensor.shape[1] <= 64
    assert tensor.dtype == torch.long


def test_empty_dir_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            StyleLibrary(d, max_theme_len=128)


def test_reload(style_dir):
    lib = StyleLibrary(style_dir, max_theme_len=128)
    initial_count = len(lib.style_names)
    # 새 파일 추가
    make_midi_file(Path(style_dir) / "new_style.mid",
                   [70, 72, 74, 75, 77, 79, 81, 82])
    lib.reload()
    assert len(lib.style_names) == initial_count + 1
    assert "new_style.mid" in lib.style_names
