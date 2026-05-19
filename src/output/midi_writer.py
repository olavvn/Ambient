"""MIDI 이벤트 파일 출력"""
from __future__ import annotations
import pretty_midi
from pathlib import Path


def save_midi(midi: pretty_midi.PrettyMIDI, path: str | Path):
    """PrettyMIDI → .mid 파일 저장"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
