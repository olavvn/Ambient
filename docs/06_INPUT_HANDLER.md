# 06 입력 핸들러 (Input Handler)

## 개요

MIDI 파일, 실시간 MIDI 포트, 오디오 파일 세 가지 입력 경로를 통일된 내부 표현으로 변환한다. `basic-pitch`를 사용하여 오디오를 MIDI로 변환할 수 있다.

---

## 1. MIDI 파일 핸들러

### `src/input/midi_handler.py`

```python
"""
MIDI 파일 및 실시간 MIDI 포트 핸들러
"""
from __future__ import annotations
import pretty_midi
import rtmidi
import threading
import time
from typing import Optional, Callable
from pathlib import Path


class MIDIFileHandler:
    """MIDI 파일 → PrettyMIDI 로드"""

    @staticmethod
    def load(path: str | Path) -> pretty_midi.PrettyMIDI:
        path = str(path)
        if not path.endswith(('.mid', '.midi')):
            raise ValueError(f"MIDI 파일이 아닙니다: {path}")
        midi = pretty_midi.PrettyMIDI(path)
        return MIDIFileHandler._normalize(midi)

    @staticmethod
    def _normalize(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        박자 정규화: 32분음표 그리드로 퀀타이즈
        템포 범위 강제: 40~120 BPM
        """
        # 템포 클리핑
        tempo_times, tempos = midi.get_tempo_change_times()
        for i, tempo in enumerate(tempos):
            tempos[i] = max(40.0, min(120.0, tempo))

        # 노트 퀀타이즈 (32분음표 = 1/8 박자)
        quant_resolution = 1.0 / 8  # 8분의 1박자
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                note.start = round(note.start / quant_resolution) * quant_resolution
                note.end = max(note.start + quant_resolution,
                               round(note.end / quant_resolution) * quant_resolution)
        return midi


class RealtimeMIDIHandler:
    """
    실시간 MIDI 포트에서 N초간 노트를 수집하여 PrettyMIDI로 반환.
    ThemeExtractor가 처리할 수 있도록 충분한 분량 확보.
    """

    def __init__(self, port_name: Optional[str] = None,
                 collect_sec: float = 4.0):
        self.midi_in = rtmidi.MidiIn()
        self.collect_sec = collect_sec
        self._events: list = []
        self._collecting = False
        self._lock = threading.Lock()

        # 포트 열기
        ports = self.midi_in.get_ports()
        if not ports:
            raise RuntimeError("MIDI 입력 포트가 없습니다.")
        if port_name:
            idx = next((i for i, p in enumerate(ports) if port_name in p), 0)
        else:
            idx = 0
        self.midi_in.open_port(idx)
        self.midi_in.set_callback(self._on_message)
        print(f"MIDI 포트 오픈: {ports[idx]}")

    def _on_message(self, message_and_delta, _data=None):
        message, delta = message_and_delta
        with self._lock:
            if self._collecting:
                self._events.append((time.time(), message))

    def collect(self) -> pretty_midi.PrettyMIDI:
        """N초 동안 MIDI 수집 후 PrettyMIDI 반환"""
        with self._lock:
            self._events.clear()
            self._collecting = True
        print(f"MIDI 수집 중... ({self.collect_sec}초)")
        time.sleep(self.collect_sec)
        with self._lock:
            self._collecting = False
            events = list(self._events)

        return self._events_to_midi(events)

    def _events_to_midi(self, events: list) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        midi.instruments.append(instrument)
        if not events:
            return midi

        t0 = events[0][0]
        active: dict[int, float] = {}
        for t, msg in events:
            rel_t = t - t0
            status = msg[0] & 0xF0
            if status == 0x90 and len(msg) > 2 and msg[2] > 0:
                active[msg[1]] = rel_t
            elif (status == 0x80 or (status == 0x90 and msg[2] == 0)):
                if msg[1] in active:
                    start = active.pop(msg[1])
                    instrument.notes.append(pretty_midi.Note(
                        velocity=64, pitch=msg[1],
                        start=start, end=rel_t
                    ))
        instrument.notes.sort(key=lambda n: n.start)
        return MIDIFileHandler._normalize(midi)

    def close(self):
        self.midi_in.close_port()
```

---

## 2. 오디오 → MIDI 변환

### `src/input/audio_handler.py`

```python
"""
오디오 파일 → MIDI 변환 (basic-pitch 사용)
"""
from pathlib import Path
import pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from src.input.midi_handler import MIDIFileHandler


class AudioHandler:
    """
    오디오 파일을 MIDI로 변환 후 정규화.
    지원 포맷: .wav, .mp3, .flac, .ogg
    """

    def __init__(self, onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold

    def load(self, path: str | Path) -> pretty_midi.PrettyMIDI:
        """오디오 파일 → 정규화된 PrettyMIDI"""
        path = Path(path)
        supported = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        if path.suffix.lower() not in supported:
            raise ValueError(f"지원하지 않는 오디오 포맷: {path.suffix}")

        print(f"기본음 추출 중: {path.name} ...")
        _, midi_data, _ = predict(
            str(path),
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        # basic-pitch는 PrettyMIDI를 반환
        return MIDIFileHandler._normalize(midi_data)
```

---

## 3. 통합 입력 핸들러

### `src/input/normalizer.py`

```python
"""
모든 입력 경로를 통합하는 팩토리 함수
"""
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
        # 실시간 MIDI 수집
        handler = RealtimeMIDIHandler(realtime_port, realtime_sec)
        try:
            return handler.collect()
        finally:
            handler.close()
```
