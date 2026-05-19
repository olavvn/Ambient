"""
StyleLibrary: styles/ 폴더의 MIDI 파일들을 로드하여 블렌딩된 테마 토큰을 제공.

ThemeTransformer와 달리 사람이 레이블한 데이터 대신,
사용자가 원하는 스타일의 MIDI 파일들을 직접 준비해 폴더에 넣어두면
그것들을 혼합하여 생성 조건으로 사용한다.

블렌딩 방식:
    각 MIDI 파일에서 대표 세그먼트를 균등하게 추출 →
    이어붙여 max_theme_len 길이의 하나의 토큰 시퀀스로 만듦 →
    ThemeEncoder가 혼합 스타일로 인코딩
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Dict

import pretty_midi
import torch

from src.theme.tokenizer import REMIAmbientTokenizer
from src.input.midi_handler import MIDIFileHandler


class StyleLibrary:
    """
    스타일 MIDI 파일 저장소.

    사용 예:
        lib = StyleLibrary("styles/", max_theme_len=128)
        theme_tokens = lib.get_blended_theme()     # 전체 블렌딩
        theme_tokens = lib.get_blended_theme(      # 특정 파일만 지정
            names=["ballad.mid", "cinematic.mid"]
        )
    """

    def __init__(
        self,
        style_dir: str,
        max_theme_len: int = 128,
        segment_per_file: Optional[int] = None,  # None이면 파일 수로 균등 분할
    ):
        self.style_dir = Path(style_dir)
        self.max_theme_len = max_theme_len
        self.segment_per_file = segment_per_file
        self.tokenizer = REMIAmbientTokenizer()

        # name → token list
        self._styles: Dict[str, List[int]] = {}
        self._load_all()

    # ── 로드 ───────────────────────────────────────────────────────────

    def _load_all(self):
        midi_files = list(self.style_dir.glob("*.mid")) + \
                     list(self.style_dir.glob("*.midi"))
        if not midi_files:
            raise FileNotFoundError(
                f"styles/ 폴더에 MIDI 파일이 없습니다: {self.style_dir}"
            )

        for path in sorted(midi_files):
            try:
                midi = MIDIFileHandler.load(path)
                tokens = self.tokenizer.midi_to_tokens(midi)
                # BOS/EOS 제거하고 음악 토큰만 보관
                tokens = self._strip_special(tokens)
                if tokens:
                    self._styles[path.name] = tokens
                    print(f"[StyleLibrary] 로드: {path.name} ({len(tokens)} 토큰)")
            except Exception as e:
                print(f"[StyleLibrary] 스킵: {path.name} — {e}")

        if not self._styles:
            raise RuntimeError("로드 가능한 스타일 MIDI 파일이 없습니다.")
        print(f"[StyleLibrary] 총 {len(self._styles)}개 스타일 준비 완료")

    def _strip_special(self, tokens: List[int]) -> List[int]:
        """BOS/EOS 등 특수 토큰 제거"""
        skip = {
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
            self.tokenizer.pad_id,
        }
        return [t for t in tokens if t not in skip]

    # ── 퍼블릭 API ─────────────────────────────────────────────────────

    @property
    def style_names(self) -> List[str]:
        """로드된 스타일 파일명 목록"""
        return list(self._styles.keys())

    def get_blended_theme(
        self,
        names: Optional[List[str]] = None,
        strategy: str = "front",
    ) -> List[int]:
        """
        여러 스타일 파일에서 균등하게 세그먼트를 추출해 이어붙인다.

        Args:
            names:    사용할 파일명 리스트. None이면 전체 사용.
            strategy: "front"  — 각 파일의 앞 부분 사용 (기본, 가장 특징적)
                      "center" — 각 파일의 중간 부분 사용
                      "random" — 각 파일의 랜덤 위치 사용

        Returns:
            theme_tokens: List[int], 길이 ≤ max_theme_len
        """
        pool = self._select_styles(names)
        n = len(pool)
        seg_len = max(1, self.max_theme_len // n)

        blended: List[int] = []
        for style_tokens in pool:
            seg = self._extract_segment(style_tokens, seg_len, strategy)
            blended.extend(seg)
            if len(blended) >= self.max_theme_len:
                break

        return blended[:self.max_theme_len]

    def get_blended_theme_tensor(
        self,
        names: Optional[List[str]] = None,
        strategy: str = "front",
        device: str = "cpu",
    ) -> torch.Tensor:
        """get_blended_theme()의 텐서 반환 버전. shape: (1, T)"""
        tokens = self.get_blended_theme(names=names, strategy=strategy)
        return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    def get_single_theme(self, name: str) -> List[int]:
        """특정 스타일 파일 하나의 테마 토큰 반환"""
        if name not in self._styles:
            raise KeyError(f"스타일 없음: {name}. 사용 가능: {self.style_names}")
        tokens = self._styles[name]
        return tokens[:self.max_theme_len]

    def reload(self):
        """styles/ 폴더를 다시 스캔 (핫 리로드)"""
        self._styles.clear()
        self._load_all()

    # ── 내부 유틸 ──────────────────────────────────────────────────────

    def _select_styles(self, names: Optional[List[str]]) -> List[List[int]]:
        if names is None:
            return list(self._styles.values())
        result = []
        for n in names:
            if n in self._styles:
                result.append(self._styles[n])
            else:
                print(f"[StyleLibrary] 경고: '{n}' 없음, 스킵")
        return result or list(self._styles.values())

    def _extract_segment(
        self, tokens: List[int], seg_len: int, strategy: str
    ) -> List[int]:
        if len(tokens) <= seg_len:
            return tokens

        if strategy == "front":
            start = 0
        elif strategy == "center":
            start = max(0, (len(tokens) - seg_len) // 2)
        elif strategy == "random":
            start = random.randint(0, len(tokens) - seg_len)
        else:
            start = 0

        return tokens[start: start + seg_len]

    def __repr__(self) -> str:
        return (f"StyleLibrary(dir={self.style_dir}, "
                f"styles={self.style_names}, "
                f"max_theme_len={self.max_theme_len})")
