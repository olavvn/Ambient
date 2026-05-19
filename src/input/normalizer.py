"""
모든 입력 경로를 통합하는 팩토리 함수
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pretty_midi

from src.input.midi_handler import MIDIFileHandler, RealtimeMIDIHandler
from src.input.audio_handler import AudioHandler


def load_input(
    path: Optional[str] = None,
    realtime_port: Optional[str] = None,
    realtime_sec: float = 4.0,
) -> pretty_midi.PrettyMIDI:
    """
    입력 경로에 따라 자동으로 적절한 핸들러 선택.

    Args:
        path: 파일 경로 (MIDI 또는 오디오)
        realtime_port: 실시간 MIDI 포트 이름 (None이면 첫 번째 포트)
        realtime_sec: 실시간 수집 시간 (초)
    """
    if path:
        ext = Path(path).suffix.lower()
        if ext in {'.mid', '.midi'}:
            return MIDIFileHandler.load(path)
        elif ext in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
            return AudioHandler().load(path)
        else:
            raise ValueError(f"알 수 없는 파일 형식: {ext}")
    else:
        handler = RealtimeMIDIHandler(realtime_port, realtime_sec)
        try:
            return handler.collect()
        finally:
            handler.close()
