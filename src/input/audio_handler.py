"""
오디오 파일 → MIDI 변환 (basic-pitch 사용)
"""
from __future__ import annotations
from pathlib import Path
import pretty_midi
from src.input.midi_handler import MIDIFileHandler


class AudioHandler:
    """
    오디오 파일을 MIDI로 변환 후 정규화.
    지원 포맷: .wav, .mp3, .flac, .ogg, .m4a
    """

    SUPPORTED = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    def __init__(self, onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold

    def load(self, path: str | Path) -> pretty_midi.PrettyMIDI:
        """오디오 파일 → 정규화된 PrettyMIDI"""
        path = Path(path)
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"지원하지 않는 오디오 포맷: {path.suffix}")

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError as e:
            raise ImportError("basic-pitch가 설치되어 있지 않습니다. pip install basic-pitch") from e

        print(f"기본음 추출 중: {path.name} ...")
        _, midi_data, _ = predict(
            str(path),
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        return MIDIFileHandler._normalize(midi_data)
