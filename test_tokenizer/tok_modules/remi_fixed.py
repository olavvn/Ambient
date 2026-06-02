"""
Fixed REMI tokenizer using miditok.REMI with ambient-optimized settings.
Fixes BUG-01/02/03 from remi_original by using the correct miditok implementation.
Ambient settings: tempo_range=(20,80), nb_velocities=16, use_rests=True
"""
from __future__ import annotations
import pretty_midi
import symusic
import tempfile
import os
from pathlib import Path
from typing import List

from miditok import REMI, TokenizerConfig

NAME = "remi_fixed"


def _build_tokenizer() -> REMI:
    config = TokenizerConfig(
        pitch_range=(21, 108),
        beat_res={(0, 4): 8, (4, 12): 4},
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
    return REMI(config)


tokenizer = _build_tokenizer()


def encode_midi(pm: pretty_midi.PrettyMIDI) -> List[int]:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        pm.write(tmp_path)
        score = symusic.Score(tmp_path)
        tok_seq = tokenizer.encode(score)
        if isinstance(tok_seq, list):
            tok_seq = tok_seq[0]
        return tok_seq.ids
    finally:
        os.unlink(tmp_path)


def decode_midi(token_ids: List[int]) -> pretty_midi.PrettyMIDI:
    from miditok import TokSequence
    tok_seq = TokSequence(ids=token_ids)
    score = tokenizer.decode([tok_seq])
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        score.dump_midi(tmp_path)
        return pretty_midi.PrettyMIDI(tmp_path)
    finally:
        os.unlink(tmp_path)
