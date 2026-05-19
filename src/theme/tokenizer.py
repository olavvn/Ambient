"""
REMI-Ambient 토크나이저
앰비언트 음악에 최적화된 MIDI ↔ 토큰 변환
"""
from __future__ import annotations
import pretty_midi
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── 어휘 정의 ──────────────────────────────────────────────────────────
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]",
                  "[THEME_S]", "[THEME_E]", "[THEME_REF]", "[SEP]"]

BAR_TOKEN = "BAR"
BEAT_TOKENS = [f"BEAT_{i}" for i in range(1, 5)]
POS_TOKENS = [f"POS_{i}" for i in range(32)]
TEMPO_TOKENS = [f"TEMPO_{t}" for t in range(40, 125, 5)]
PITCH_TOKENS = [f"PITCH_{p}" for p in range(128)]
DUR_TOKENS = [f"DUR_{d}" for d in list(range(1, 33)) + [48, 64]]
VEL_TOKENS = [f"VEL_{v}" for v in range(8, 120, 14)]
SUSTAIN_TOKENS = ["SUSTAIN_ON", "SUSTAIN_OFF"]
INST_TOKENS = ["INST_PIANO", "INST_PAD", "INST_BASS", "INST_MELODY"]

ALL_TOKENS = (
    SPECIAL_TOKENS + [BAR_TOKEN] + BEAT_TOKENS + POS_TOKENS
    + TEMPO_TOKENS + PITCH_TOKENS + DUR_TOKENS
    + VEL_TOKENS + SUSTAIN_TOKENS + INST_TOKENS
)


@dataclass
class MusicEvent:
    """내부 음악 이벤트 표현"""
    event_type: str
    time: float
    pitch: Optional[int] = None
    duration: float = 0.0
    velocity: int = 64
    instrument: str = "INST_PIANO"


class REMIAmbientTokenizer:
    """REMI-Ambient 토크나이저"""

    def __init__(self, ticks_per_beat: int = 480, beats_per_bar: int = 4,
                 subdivisions: int = 32):
        self.ticks_per_beat = ticks_per_beat
        self.beats_per_bar = beats_per_bar
        self.subdivisions = subdivisions

        self.token2id: dict[str, int] = {t: i for i, t in enumerate(ALL_TOKENS)}
        self.id2token: dict[int, str] = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(ALL_TOKENS)

        self.pad_id = self.token2id["[PAD]"]
        self.bos_id = self.token2id["[BOS]"]
        self.eos_id = self.token2id["[EOS]"]
        self.theme_s_id = self.token2id["[THEME_S]"]
        self.theme_e_id = self.token2id["[THEME_E]"]
        self.bar_id = self.token2id[BAR_TOKEN]

    # ── MIDI → Tokens ──────────────────────────────────────────────────
    def midi_to_tokens(
        self,
        midi: pretty_midi.PrettyMIDI,
        instrument_label: str = "INST_PIANO",
        theme_spans: Optional[List[Tuple[float, float]]] = None,
    ) -> List[int]:
        """
        pretty_midi.PrettyMIDI → 토큰 ID 리스트

        Args:
            midi: 입력 MIDI
            instrument_label: 악기 레이블
            theme_spans: [(start_sec, end_sec), ...] 테마 구간
        """
        events = self._extract_events(midi, instrument_label)
        tokens = [self.bos_id]
        tokens += self._events_to_tokens(events, theme_spans or [])
        tokens.append(self.eos_id)
        return tokens

    def _extract_events(self, midi: pretty_midi.PrettyMIDI,
                        inst_label: str) -> List[MusicEvent]:
        events: List[MusicEvent] = []
        tempo_change_times, tempos = midi.get_tempo_changes()
        for t, tempo in zip(tempo_change_times, tempos):
            events.append(MusicEvent("tempo", t, velocity=int(tempo)))
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                events.append(MusicEvent(
                    "note", note.start,
                    pitch=note.pitch,
                    duration=note.end - note.start,
                    velocity=note.velocity,
                    instrument=inst_label,
                ))
        events.sort(key=lambda e: e.time)
        return events

    def _events_to_tokens(self, events: List[MusicEvent],
                          theme_spans: List[Tuple[float, float]]) -> List[int]:
        tokens: List[int] = []
        current_bar = -1

        def in_theme(t: float) -> bool:
            return any(s <= t < e for s, e in theme_spans)

        for ev in events:
            bar_idx = self._time_to_bar(ev.time)
            if bar_idx != current_bar:
                current_bar = bar_idx
                tokens.append(self.token2id[BAR_TOKEN])

            if in_theme(ev.time):
                tokens.append(self.token2id["[THEME_REF]"])

            pos = self._time_to_pos(ev.time)
            tokens.append(self.token2id[POS_TOKENS[pos % self.subdivisions]])

            if ev.event_type == "tempo":
                tempo = max(40, min(120, int(ev.velocity)))
                tempo_token = f"TEMPO_{(tempo // 5) * 5}"
                tokens.append(self.token2id.get(tempo_token, self.token2id["TEMPO_60"]))

            elif ev.event_type == "note":
                inst_suffix = ev.instrument.split('_')[-1]
                inst_key = f"INST_{inst_suffix}"
                tokens.append(
                    self.token2id.get(inst_key, self.token2id["INST_PIANO"])
                )
                tokens.append(self.token2id[f"PITCH_{ev.pitch}"])
                dur_units = max(1, min(64, round(ev.duration * self.subdivisions / 2)))
                dur_token = self._closest_dur_token(dur_units)
                tokens.append(self.token2id[dur_token])
                vel_token = self._closest_vel_token(ev.velocity)
                tokens.append(self.token2id[vel_token])

        return tokens

    # ── Tokens → MIDI ──────────────────────────────────────────────────
    def tokens_to_midi(self, token_ids: List[int],
                       bpm: float = 60.0) -> pretty_midi.PrettyMIDI:
        """토큰 ID 리스트 → pretty_midi.PrettyMIDI"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        piano = pretty_midi.Instrument(program=0, name="Piano")
        midi.instruments.append(piano)

        current_time = 0.0
        current_beat_dur = 60.0 / bpm
        tokens = [self.id2token.get(tid, "[PAD]") for tid in token_ids]

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == BAR_TOKEN:
                current_time += current_beat_dur * self.beats_per_bar
            elif tok.startswith("TEMPO_"):
                try:
                    bpm = float(tok.split("_")[1])
                    current_beat_dur = 60.0 / bpm
                except (ValueError, IndexError):
                    pass
            elif tok.startswith("PITCH_"):
                try:
                    pitch = int(tok.split("_")[1])
                    dur_tok = tokens[i + 1] if i + 1 < len(tokens) else "DUR_4"
                    vel_tok = tokens[i + 2] if i + 2 < len(tokens) else "VEL_64"
                    dur_units = int(dur_tok.split("_")[1]) if dur_tok.startswith("DUR_") else 4
                    vel = int(vel_tok.split("_")[1]) if vel_tok.startswith("VEL_") else 64
                    duration = dur_units * (current_beat_dur / (self.subdivisions / self.beats_per_bar))
                    note = pretty_midi.Note(
                        velocity=min(127, max(1, vel)),
                        pitch=min(127, max(0, pitch)),
                        start=current_time,
                        end=current_time + max(duration, 0.01),
                    )
                    piano.notes.append(note)
                    i += 2
                except (ValueError, IndexError):
                    pass
            i += 1

        piano.notes.sort(key=lambda n: n.start)
        return midi

    # ── Helper ─────────────────────────────────────────────────────────
    def _time_to_bar(self, time: float, bpm: float = 60.0) -> int:
        beat = time / (60.0 / bpm)
        return int(beat / self.beats_per_bar)

    def _time_to_pos(self, time: float, bpm: float = 60.0) -> int:
        beat = time / (60.0 / bpm)
        frac = beat % 1.0
        return int(frac * self.subdivisions)

    def _closest_dur_token(self, units: int) -> str:
        candidates = list(range(1, 33)) + [48, 64]
        closest = min(candidates, key=lambda x: abs(x - units))
        return f"DUR_{closest}"

    def _closest_vel_token(self, vel: int) -> str:
        levels = list(range(8, 120, 14))
        closest = min(levels, key=lambda x: abs(x - vel))
        return f"VEL_{closest}"

    def encode(self, token_strs: List[str]) -> List[int]:
        return [self.token2id.get(t, self.pad_id) for t in token_strs]

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id2token.get(i, "[PAD]") for i in token_ids]
