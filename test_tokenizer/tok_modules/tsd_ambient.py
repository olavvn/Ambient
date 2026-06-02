"""
TSD-based ambient tokenizer using miditok.TSD.
Ambient-optimized: wide tempo range (20-80 BPM), long note support, more velocities.
CC events (91 Reverb, 74 Brightness) are round-tripped via a separate token layer.
"""
from __future__ import annotations
import pretty_midi
import symusic
import tempfile
import os
from typing import List, Tuple

from miditok import TSD, TokenizerConfig

NAME = "tsd_ambient"

# CC numbers to track for preservation rate metrics
TRACKED_CC = [64, 74, 91]

# CC quantization bins (16 levels = 8-step intervals)
CC_BINS = list(range(0, 128, 8))  # 16 bins: 0,8,16,...,120


def _build_tokenizer() -> TSD:
    config = TokenizerConfig(
        pitch_range=(21, 108),
        # Fine resolution for slow ambient notes: 8 subdivisions up to beat 8, 4 beyond
        beat_res={(0, 8): 8, (8, 64): 4},
        num_velocities=16,
        use_tempos=True,
        use_rests=True,
        use_programs=False,
        use_chords=False,
        use_time_signatures=False,
        tempo_range=(20, 80),
        num_tempos=30,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
    )
    return TSD(config)


tokenizer = _build_tokenizer()


def _quantize_cc(val: int) -> int:
    return min(CC_BINS, key=lambda b: abs(b - val))


def encode_midi(pm: pretty_midi.PrettyMIDI) -> Tuple[List[int], List[Tuple[int, float, int]]]:
    """
    Returns (token_ids, cc_events) where cc_events = [(cc_number, time_sec, value), ...]
    CC events are preserved separately since miditok TSD does not natively encode them.
    """
    cc_events: List[Tuple[int, float, int]] = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number in TRACKED_CC:
                cc_events.append((cc.number, cc.time, _quantize_cc(cc.value)))

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        pm.write(tmp_path)
        score = symusic.Score(tmp_path)
        tok_seq = tokenizer.encode(score)
        if isinstance(tok_seq, list):
            tok_seq = tok_seq[0]
        return tok_seq.ids, cc_events
    finally:
        os.unlink(tmp_path)


def decode_midi(token_ids: List[int],
                cc_events: List[Tuple[int, float, int]] = None) -> pretty_midi.PrettyMIDI:
    from miditok import TokSequence
    tok_seq = TokSequence(ids=token_ids)
    score = tokenizer.decode([tok_seq])
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        score.dump_midi(tmp_path)
        pm = pretty_midi.PrettyMIDI(tmp_path)
        # Re-inject CC events
        if cc_events and pm.instruments:
            inst = pm.instruments[0]
            for cc_num, t, val in cc_events:
                inst.control_changes.append(pretty_midi.ControlChange(cc_num, val, t))
            inst.control_changes.sort(key=lambda c: c.time)
        return pm
    finally:
        os.unlink(tmp_path)
