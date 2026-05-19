# 02 음악 표현 및 토크나이저 (Music Representation & Tokenizer)

## 설계 원칙

ThemeTransformer의 **REMI+** 토큰화 방식을 앰비언트 음악에 맞게 변형한다. 앰비언트의 특성상:
- 템포가 느리고 유동적 (40~90 BPM)
- 지속음(sustain), 패드, 텍스처 중심
- 다성부(polyphony) 표현 필수

---

## 1. 토큰 어휘 (Vocabulary)

### 1-1. 특수 토큰
```
[PAD]        # 패딩
[BOS]        # 시퀀스 시작
[EOS]        # 시퀀스 종료
[MASK]       # 마스킹 (훈련용)
[THEME_S]    # 테마 구간 시작
[THEME_E]    # 테마 구간 종료
[THEME_REF]  # 현재 위치가 테마를 참조·변형함을 표시
[SEP]        # 인코더-디코더 구분자
```

### 1-2. 음악 이벤트 토큰
```
# Bar & Beat
BAR                         # 새 마디 (1개)
BEAT_1 ~ BEAT_4             # 박자 위치 (4개)
POS_0 ~ POS_31              # 마디 내 세밀한 위치 (32개, 32분음표 단위)

# Tempo
TEMPO_40 ~ TEMPO_120        # 5 BPM 단계로 17개 토큰

# Pitch
PITCH_0 ~ PITCH_127         # 128개 (MIDI 표준)

# Duration  
DUR_1 ~ DUR_32              # 32분음표 단위, 1~32 (최대 1마디)
DUR_48, DUR_64              # 긴 음 (앰비언트용, 1.5, 2마디)

# Velocity
VEL_8, VEL_16 ... VEL_112  # 8단계 (8씩 증가)

# Sustain Pedal
SUSTAIN_ON
SUSTAIN_OFF

# Instrument (멀티트랙)
INST_PIANO                  # 어쿠스틱 피아노
INST_PAD                    # 스트링/신스 패드
INST_BASS                   # 베이스
INST_MELODY                 # 멜로디 리드
```

### 1-3. 어휘 크기 요약
```
특수 토큰:   8
BAR/BEAT/POS: 1 + 4 + 32 = 37
TEMPO:       17
PITCH:       128
DURATION:    34
VELOCITY:    8 (8단계)
SUSTAIN:     2
INSTRUMENT:  4
─────────────────
총 어휘 크기: ~238 토큰
```

---

## 2. 토크나이저 구현

### `src/theme/tokenizer.py`

```python
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
VEL_TOKENS = [f"VEL_{v}" for v in range(8, 120, 14)]   # 8단계
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
    event_type: str          # "note", "tempo", "sustain", "bar"
    time: float              # 절대 시간 (초)
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
        self.subdivisions = subdivisions  # 마디당 세분화

        # vocab → id 매핑
        self.token2id: dict[str, int] = {t: i for i, t in enumerate(ALL_TOKENS)}
        self.id2token: dict[int, str] = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(ALL_TOKENS)

        # 특수 토큰 id
        self.pad_id = self.token2id["[PAD]"]
        self.bos_id = self.token2id["[BOS]"]
        self.eos_id = self.token2id["[EOS]"]
        self.theme_s_id = self.token2id["[THEME_S]"]
        self.theme_e_id = self.token2id["[THEME_E]"]

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
        # Tempo events
        tempo_change_times, tempos = midi.get_tempo_change_times()
        for t, tempo in zip(tempo_change_times, tempos):
            events.append(MusicEvent("tempo", t, velocity=int(tempo)))
        # Note events
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
        tps = self.subdivisions  # ticks per subdivision unit

        def in_theme(t: float) -> bool:
            return any(s <= t < e for s, e in theme_spans)

        for ev in events:
            # 마디 경계 삽입
            bar_idx = self._time_to_bar(ev.time)
            if bar_idx != current_bar:
                current_bar = bar_idx
                tokens.append(self.token2id[BAR_TOKEN])

            # 테마 구간 마커
            if in_theme(ev.time):
                tokens.append(self.token2id["[THEME_REF]"])

            pos = self._time_to_pos(ev.time)
            tokens.append(self.token2id[POS_TOKENS[pos % self.subdivisions]])

            if ev.event_type == "tempo":
                tempo = max(40, min(120, int(ev.velocity)))
                tempo_token = f"TEMPO_{(tempo // 5) * 5}"
                tokens.append(self.token2id.get(tempo_token, self.token2id["TEMPO_60"]))

            elif ev.event_type == "note":
                tokens.append(self.token2id[f"INST_{ev.instrument.split('_')[-1]}"]
                               if f"INST_{ev.instrument.split('_')[-1]}" in self.token2id
                               else self.token2id["INST_PIANO"])
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
        i = 0
        tokens = [self.id2token.get(tid, "[PAD]") for tid in token_ids]

        while i < len(tokens):
            tok = tokens[i]
            if tok == BAR_TOKEN:
                current_time += current_beat_dur * self.beats_per_bar
            elif tok.startswith("TEMPO_"):
                bpm = float(tok.split("_")[1])
                current_beat_dur = 60.0 / bpm
            elif tok.startswith("PITCH_"):
                pitch = int(tok.split("_")[1])
                dur_tok = tokens[i + 1] if i + 1 < len(tokens) else "DUR_4"
                vel_tok = tokens[i + 2] if i + 2 < len(tokens) else "VEL_64"
                dur_units = int(dur_tok.split("_")[1]) if dur_tok.startswith("DUR_") else 4
                vel = int(vel_tok.split("_")[1]) if vel_tok.startswith("VEL_") else 64
                duration = dur_units * (current_beat_dur / (self.subdivisions / self.beats_per_bar))
                note = pretty_midi.Note(
                    velocity=vel, pitch=pitch,
                    start=current_time, end=current_time + duration
                )
                piano.notes.append(note)
                i += 2  # pitch + dur + vel 묶음
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
```

---

## 3. 데이터셋 준비 지침

### 사용 데이터셋 (우선순위 순)
1. **Maestro v3** (피아노 MIDI, 200시간) — 메인 훈련 데이터
2. **GiantMIDI-Piano** (피아노, 172시간)
3. **POP909** (팝 피아노, 909곡) — ThemeTransformer 원본 사용 데이터
4. **ATEPP** (앰비언트/클래식 피아노)

### 전처리 스크립트 위치
```
scripts/preprocess_data.py
```

### 전처리 파이프라인
```
Raw MIDI
  ↓ 박자 정규화 (quantize to 32분음표)
  ↓ 조성 정규화 (C장조 또는 A단조로 변환)
  ↓ 템포 필터링 (40~120 BPM만 유지)
  ↓ 청크 분할 (512 토큰 단위, 50% overlap)
  ↓ 테마 구간 자동 레이블링 (다음 문서 참조)
  ↓ Pickle 저장 (data_pkl/)
```

### `scripts/preprocess_data.py` 핵심 구조
```python
from src.theme.tokenizer import REMIAmbientTokenizer
from src.theme.extractor import ThemeExtractor
import pretty_midi, pickle, pathlib

tokenizer = REMIAmbientTokenizer()
extractor = ThemeExtractor()

def process_file(midi_path: str) -> dict:
    midi = pretty_midi.PrettyMIDI(midi_path)
    # 1. 테마 추출
    theme_spans = extractor.extract_theme_spans(midi)
    # 2. 토크나이즈
    tokens = tokenizer.midi_to_tokens(midi, theme_spans=theme_spans)
    # 3. 청크 분할
    chunks = [tokens[i:i+512] for i in range(0, len(tokens)-512, 256)]
    return {"path": midi_path, "tokens": tokens, "chunks": chunks,
            "theme_spans": theme_spans}

if __name__ == "__main__":
    data_dir = pathlib.Path("data/raw_midi")
    out_dir = pathlib.Path("data_pkl")
    out_dir.mkdir(exist_ok=True)
    for f in data_dir.glob("**/*.mid"):
        result = process_file(str(f))
        out_path = out_dir / (f.stem + ".pkl")
        with open(out_path, "wb") as fp:
            pickle.dump(result, fp)
        print(f"Processed: {f.name} → {len(result['chunks'])} chunks")
```
