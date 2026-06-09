"""
TSD Ambient Tokenizer
앰비언트 음악 특화 Time-Shift-Duration 토크나이저.
TimeShift 기반 시간 표현, 확장 템포(20-80 BPM), 확장 Duration(최대 32박), 16단계 벨로시티,
CC: Sustain(64), Volume(7), Expression(11), Brightness(74), Reverb(91), Chorus(93)
"""
from __future__ import annotations
import pretty_midi
from typing import List, Optional, Tuple

# ── Quantization tables (unit = 1/8 beat) ────────────────────────────────

_TS_TICKS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8,
    10, 12, 14, 16,
    20, 24, 28, 32,
    40, 48, 56, 64,
    80, 96, 112, 128,
    160, 192, 224, 256,
    320, 384, 448, 512,
]  # 32 values, 1/8 beat ~ 64 beats

_DUR_TICKS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8,
    10, 12, 14, 16,
    20, 24, 28, 32,
    40, 48, 56, 64,
    80, 96, 112, 128,
    160, 192, 224, 256,
]  # 28 values, 1/8 beat ~ 32 beats

_VEL_LEVELS: List[int] = [4 + i * 8 for i in range(16)]    # 4,12,...,124
_TEMPO_LEVELS: List[int] = list(range(20, 82, 2))            # 20,22,...,80 (31)
_CC16_BINS: List[int] = list(range(0, 128, 8))               # 0,8,...,120 (16)
_CC8_BINS: List[int] = [0, 18, 36, 54, 72, 90, 109, 127]    # 8 values

# ── Vocabulary ────────────────────────────────────────────────────────────

SPECIAL_TOKENS   = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"]
TIMESHIFT_TOKENS = [f"TS_{v}" for v in _TS_TICKS]
DURATION_TOKENS  = [f"DUR_{v}" for v in _DUR_TICKS]
PITCH_TOKENS     = [f"PITCH_{p}" for p in range(21, 109)]    # A0~C8, 88개
VELOCITY_TOKENS  = [f"VEL_{v}" for v in _VEL_LEVELS]
TEMPO_TOKENS_VOC = [f"TEMPO_{t}" for t in _TEMPO_LEVELS]
SUSTAIN_TOKENS   = ["SUSTAIN_ON", "SUSTAIN_OFF"]
VOL_TOKENS       = [f"VOL_{v}" for v in _CC16_BINS]
EXPR_TOKENS      = [f"EXPR_{v}" for v in _CC16_BINS]
BRIGHT_TOKENS    = [f"BRIGHT_{v}" for v in _CC16_BINS]
REVERB_TOKENS    = [f"REVERB_{v}" for v in _CC16_BINS]
CHORUS_TOKENS    = [f"CHORUS_{v}" for v in _CC8_BINS]

ALL_TOKENS: List[str] = (
    SPECIAL_TOKENS + TIMESHIFT_TOKENS + DURATION_TOKENS +
    PITCH_TOKENS + VELOCITY_TOKENS + TEMPO_TOKENS_VOC +
    SUSTAIN_TOKENS + VOL_TOKENS + EXPR_TOKENS +
    BRIGHT_TOKENS + REVERB_TOKENS + CHORUS_TOKENS
)
# 총계: 5+32+28+88+16+31+2+16+16+16+16+8 = 274

_CC_NUM_TO_PREFIX = {7: "VOL", 11: "EXPR", 74: "BRIGHT", 91: "REVERB", 93: "CHORUS"}
_CC_NUM_TO_BINS   = {7: _CC16_BINS, 11: _CC16_BINS, 74: _CC16_BINS,
                     91: _CC16_BINS, 93: _CC8_BINS}


class TSDTokenizer:
    """
    TSD-Ambient 토크나이저.
    인터페이스: midi_to_tokens / tokens_to_midi / vocab_size / pad_id / bos_id / eos_id
    """

    SUBDIVISION = 8  # 1 beat = 8 ticks

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {t: i for i, t in enumerate(ALL_TOKENS)}
        self.id2token: dict[int, str]  = {i: t for t, i in self.token2id.items()}
        self.vocab_size: int = len(ALL_TOKENS)
        self.pad_id  = self.token2id["[PAD]"]
        self.bos_id  = self.token2id["[BOS]"]
        self.eos_id  = self.token2id["[EOS]"]
        self.sep_id  = self.token2id["[SEP]"]

    # ── MIDI → Tokens ─────────────────────────────────────────────────────

    def midi_to_tokens(self, midi: pretty_midi.PrettyMIDI) -> List[int]:
        tempo_map = self._get_tempo_map(midi)
        events = self._extract_events(midi, tempo_map)
        events.sort(key=lambda e: (e["beats"], e["priority"]))

        tokens: List[int] = [self.bos_id]
        prev_beats = 0.0

        for ev in events:
            gap_ticks = round((ev["beats"] - prev_beats) * self.SUBDIVISION)
            if gap_ticks > 0:
                self._emit_timeshift(tokens, gap_ticks)
                prev_beats = ev["beats"]

            t = ev["type"]
            if t == "tempo":
                tokens.append(self.token2id[self._snap_tempo(ev["bpm"])])
            elif t == "note":
                p = max(21, min(108, ev["pitch"]))
                d = max(1, round(ev["dur_beats"] * self.SUBDIVISION))
                tokens.append(self.token2id[f"PITCH_{p}"])
                tokens.append(self.token2id[f"DUR_{self._snap_val(d, _DUR_TICKS)}"])
                tokens.append(self.token2id[f"VEL_{self._snap_val(ev['velocity'], _VEL_LEVELS)}"])
            elif t == "cc":
                cc_tok = self._cc_token(ev["num"], ev["value"])
                if cc_tok is not None:
                    tokens.append(self.token2id[cc_tok])

        tokens.append(self.eos_id)
        return tokens

    def _emit_timeshift(self, tokens: List[int], gap_ticks: int) -> None:
        while gap_ticks > 0:
            if gap_ticks >= 512:
                tokens.append(self.token2id["TS_512"])
                gap_ticks -= 512
            else:
                tokens.append(self.token2id[f"TS_{self._snap_val(gap_ticks, _TS_TICKS)}"])
                break

    def _get_tempo_map(self, midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, float]]:
        times, tempos = midi.get_tempo_changes()
        if len(times) == 0:
            return [(0.0, 120.0)]
        return list(zip(times.tolist(), tempos.tolist()))

    def _seconds_to_beats(self, t_sec: float,
                           tempo_map: List[Tuple[float, float]]) -> float:
        beats, prev_t, prev_bpm = 0.0, 0.0, tempo_map[0][1]
        for chg_t, bpm in tempo_map:
            if chg_t >= t_sec:
                break
            beats += (chg_t - prev_t) * (prev_bpm / 60.0)
            prev_t, prev_bpm = chg_t, bpm
        beats += (t_sec - prev_t) * (prev_bpm / 60.0)
        return max(0.0, beats)

    def _bpm_at(self, t_sec: float, tempo_map: List[Tuple[float, float]]) -> float:
        bpm = tempo_map[0][1]
        for chg_t, new_bpm in tempo_map:
            if chg_t > t_sec:
                break
            bpm = new_bpm
        return bpm

    def _extract_events(self, midi: pretty_midi.PrettyMIDI,
                        tempo_map: List[Tuple[float, float]]) -> List[dict]:
        events: List[dict] = []

        for t, bpm in tempo_map:
            events.append({
                "type": "tempo", "beats": self._seconds_to_beats(t, tempo_map),
                "bpm": bpm, "priority": 0,
            })

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                b = self._seconds_to_beats(note.start, tempo_map)
                dur_b = (note.end - note.start) * (self._bpm_at(note.start, tempo_map) / 60.0)
                events.append({
                    "type": "note", "beats": b, "priority": 2,
                    "pitch": note.pitch, "dur_beats": dur_b, "velocity": note.velocity,
                })
            for cc in inst.control_changes:
                if cc.number in (7, 11, 64, 74, 91, 93):
                    events.append({
                        "type": "cc", "beats": self._seconds_to_beats(cc.time, tempo_map),
                        "priority": 1, "num": cc.number, "value": cc.value,
                    })

        return events

    # ── Tokens → MIDI ─────────────────────────────────────────────────────

    def tokens_to_midi(self, token_ids: List[int],
                       bpm: float = 60.0) -> pretty_midi.PrettyMIDI:
        init_bpm = bpm
        for tid in token_ids:
            tok = self.id2token.get(tid, "[PAD]")
            if tok.startswith("TEMPO_"):
                init_bpm = float(tok[6:])
                break

        midi = pretty_midi.PrettyMIDI(initial_tempo=init_bpm)
        piano = pretty_midi.Instrument(program=0, name="Piano")
        midi.instruments.append(piano)

        cur_secs = 0.0
        cur_bpm  = bpm
        tokens = [self.id2token.get(tid, "[PAD]") for tid in token_ids]
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok.startswith("TS_"):
                cur_secs += int(tok[3:]) / self.SUBDIVISION * (60.0 / cur_bpm)

            elif tok.startswith("TEMPO_"):
                cur_bpm = float(tok[6:])

            elif tok.startswith("PITCH_"):
                pitch = int(tok[6:])
                dur_ticks, vel = 8, 64
                if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                    dur_ticks = int(tokens[i + 1][4:])
                    i += 1
                if i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_"):
                    vel = int(tokens[i + 1][4:])
                    i += 1
                dur_secs = dur_ticks / self.SUBDIVISION * (60.0 / cur_bpm)
                piano.notes.append(pretty_midi.Note(
                    velocity=max(1, min(127, vel)),
                    pitch=max(0, min(127, pitch)),
                    start=cur_secs,
                    end=max(cur_secs + 0.01, cur_secs + dur_secs),
                ))

            elif tok == "SUSTAIN_ON":
                piano.control_changes.append(pretty_midi.ControlChange(64, 127, cur_secs))
            elif tok == "SUSTAIN_OFF":
                piano.control_changes.append(pretty_midi.ControlChange(64, 0, cur_secs))
            else:
                for prefix, cc_num in (("VOL_", 7), ("EXPR_", 11),
                                        ("BRIGHT_", 74), ("REVERB_", 91), ("CHORUS_", 93)):
                    if tok.startswith(prefix):
                        piano.control_changes.append(
                            pretty_midi.ControlChange(cc_num, int(tok[len(prefix):]), cur_secs))
                        break

            i += 1

        piano.notes.sort(key=lambda n: n.start)
        piano.control_changes.sort(key=lambda c: c.time)
        return midi

    # ── Utils ──────────────────────────────────────────────────────────────

    @staticmethod
    def _snap_val(value: float, table: List[int]) -> int:
        return min(table, key=lambda c: abs(c - value))

    def _snap_tempo(self, bpm: float) -> str:
        return f"TEMPO_{self._snap_val(round(bpm), _TEMPO_LEVELS)}"

    def _cc_token(self, cc_num: int, value: int) -> Optional[str]:
        if cc_num == 64:
            return "SUSTAIN_ON" if value >= 64 else "SUSTAIN_OFF"
        prefix = _CC_NUM_TO_PREFIX.get(cc_num)
        bins   = _CC_NUM_TO_BINS.get(cc_num)
        if prefix and bins:
            return f"{prefix}_{self._snap_val(value, bins)}"
        return None

    def encode(self, token_strs: List[str]) -> List[int]:
        return [self.token2id.get(t, self.pad_id) for t in token_strs]

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id2token.get(i, "[PAD]") for i in token_ids]
