"""
ThemeExtractor: MIDI / 토큰 시퀀스에서 테마를 추출하는 메인 인터페이스
"""
from __future__ import annotations
import torch
import numpy as np
import pretty_midi
from typing import List, Tuple, Optional

from src.theme.tokenizer import REMIAmbientTokenizer
from src.theme.contrastive import SegmentEncoder
from src.theme.clustering import cluster_segments, select_theme, ThemeCandidate


class ThemeExtractor:
    """
    입력 → 테마 토큰 시퀀스 추출 파이프라인.
    훈련된 SegmentEncoder 가중치를 로드하여 사용.
    """

    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        segment_len: int = 64,
        stride: int = 32,
        cluster_method: str = "dbscan",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tokenizer = REMIAmbientTokenizer()
        self.encoder = SegmentEncoder(vocab_size=self.tokenizer.vocab_size)
        if encoder_checkpoint:
            state = torch.load(encoder_checkpoint, map_location=device)
            self.encoder.load_state_dict(state)
        self.encoder.eval().to(device)

        self.segment_len = segment_len
        self.stride = stride
        self.cluster_method = cluster_method
        self.device = device

    # ── 메인 인터페이스 ────────────────────────────────────────────────

    def extract_from_midi(
        self, midi: pretty_midi.PrettyMIDI
    ) -> Tuple[List[int], List[Tuple[float, float]]]:
        """
        MIDI → (테마 토큰 시퀀스, 테마 구간 [(start_sec, end_sec)])
        """
        full_tokens = self.tokenizer.midi_to_tokens(midi)
        return self.extract_from_tokens(full_tokens, midi=midi)

    def extract_from_tokens(
        self, tokens: List[int],
        midi: Optional[pretty_midi.PrettyMIDI] = None,
    ) -> Tuple[List[int], List[Tuple[float, float]]]:
        """토큰 시퀀스 → (테마 토큰, 테마 구간)"""
        segments = self._sliding_window(tokens)
        if len(segments) < 2:
            return tokens, []

        embeddings = self._embed_segments(segments)

        candidates = cluster_segments(embeddings, method=self.cluster_method)
        if not candidates:
            return segments[0].tolist(), []
        theme = select_theme(candidates)

        theme_tokens = segments[theme.representative_idx].tolist()

        theme_spans: List[Tuple[float, float]] = []
        if midi:
            dur = midi.get_end_time()
            tok_per_sec = len(tokens) / max(dur, 1.0)
            for idx in theme.member_indices:
                start_tok = idx * self.stride
                end_tok = start_tok + self.segment_len
                start_sec = start_tok / tok_per_sec
                end_sec = min(end_tok / tok_per_sec, dur)
                theme_spans.append((start_sec, end_sec))

        return theme_tokens, theme_spans

    # ── Helper ─────────────────────────────────────────────────────────

    def _sliding_window(self, tokens: List[int]) -> np.ndarray:
        segs = []
        for start in range(0, len(tokens) - self.segment_len + 1, self.stride):
            seg = tokens[start: start + self.segment_len]
            segs.append(seg)
        if not segs:
            return np.array([], dtype=np.int64).reshape(0, self.segment_len)
        return np.array(segs, dtype=np.int64)

    @torch.no_grad()
    def _embed_segments(self, segments: np.ndarray) -> np.ndarray:
        batch_size = 64
        all_embs = []
        for i in range(0, len(segments), batch_size):
            batch = torch.tensor(
                segments[i: i + batch_size], dtype=torch.long, device=self.device
            )
            padding_mask = (batch == self.tokenizer.pad_id)
            emb = self.encoder(batch, padding_mask)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    def extract_theme_spans(
        self, midi: pretty_midi.PrettyMIDI
    ) -> List[Tuple[float, float]]:
        """테마 구간만 반환 (전처리 레이블링용 편의 메서드)"""
        _, spans = self.extract_from_midi(midi)
        return spans
