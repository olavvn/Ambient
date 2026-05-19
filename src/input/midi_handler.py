"""
MIDI 파일 및 실시간 MIDI 포트 핸들러
"""
from __future__ import annotations
import pretty_midi
import threading
import time
from typing import Optional
from pathlib import Path


class MIDIFileHandler:
    """MIDI 파일 → PrettyMIDI 로드"""

    @staticmethod
    def load(path: str | Path) -> pretty_midi.PrettyMIDI:
        path = str(path)
        if not path.lower().endswith(('.mid', '.midi')):
            raise ValueError(f"MIDI 파일이 아닙니다: {path}")
        midi = pretty_midi.PrettyMIDI(path)
        return MIDIFileHandler._normalize(midi)

    @staticmethod
    def _normalize(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """박자 정규화 및 템포 범위 강제 (40~120 BPM)"""
        quant_resolution = 1.0 / 8
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                note.start = round(note.start / quant_resolution) * quant_resolution
                note.end = max(
                    note.start + quant_resolution,
                    round(note.end / quant_resolution) * quant_resolution,
                )
        return midi


class RealtimeMIDIHandler:
    """
    실시간 MIDI 포트에서 N초간 노트를 수집하여 PrettyMIDI로 반환.
    rtmidi가 없는 환경에서는 ImportError를 발생시킨다.
    """

    def __init__(self, port_name: Optional[str] = None,
                 collect_sec: float = 4.0):
        try:
            import rtmidi
        except ImportError as e:
            raise ImportError("python-rtmidi가 설치되어 있지 않습니다. pip install python-rtmidi") from e

        import rtmidi as _rtmidi
        self.midi_in = _rtmidi.MidiIn()
        self.collect_sec = collect_sec
        self._events: list = []
        self._collecting = False
        self._lock = threading.Lock()

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
            elif status == 0x80 or (status == 0x90 and len(msg) > 2 and msg[2] == 0):
                if msg[1] in active:
                    start = active.pop(msg[1])
                    instrument.notes.append(pretty_midi.Note(
                        velocity=64, pitch=msg[1],
                        start=start, end=rel_t,
                    ))
        instrument.notes.sort(key=lambda n: n.start)
        return MIDIFileHandler._normalize(midi)

    def close(self):
        self.midi_in.close_port()
