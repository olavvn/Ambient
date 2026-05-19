"""
테마 전환 시 오디오 크로스페이드
MIDI 레벨이 아닌 오디오 버퍼 레벨에서 수행
"""
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.theme.tokenizer import REMIAmbientTokenizer


def crossfade_audio(
    audio_out: np.ndarray,
    audio_in: np.ndarray,
    fade_samples: int = 22050,
    sr: int = 44100,
) -> np.ndarray:
    """
    두 오디오 버퍼의 크로스페이드 구간 생성.
    앰비언트 특성에 맞게 equal-power 크로스페이드 사용.

    Returns:
        crossfaded: (fade_samples,) 크로스페이드 구간
    """
    fade_out = audio_out[-fade_samples:] if len(audio_out) >= fade_samples else audio_out
    fade_in = audio_in[:fade_samples] if len(audio_in) >= fade_samples else audio_in

    n = min(len(fade_out), len(fade_in), fade_samples)
    t = np.linspace(0, np.pi / 2, n)

    gain_out = np.cos(t)
    gain_in = np.sin(t)

    return (fade_out[:n] * gain_out + fade_in[:n] * gain_in).astype(np.float32)


class TokenLevelCrossfade:
    """
    MIDI 토큰 레벨 크로스페이드.
    현재 청크를 자연스러운 마디 경계(BAR 토큰)에서 종료하고
    새 테마의 생성으로 전환.
    """

    def __init__(self, bar_token_id: int):
        self.bar_token_id = bar_token_id

    def find_next_bar(self, tokens: List[int], from_pos: int,
                      max_lookahead: int = 64) -> int:
        """from_pos 이후 첫 번째 BAR 토큰 위치 반환"""
        for i in range(from_pos, min(from_pos + max_lookahead, len(tokens))):
            if tokens[i] == self.bar_token_id:
                return i
        return from_pos + max_lookahead

    def plan_transition(self, current_chunk: List[int],
                        current_position: int) -> int:
        """
        현재 재생 위치에서 가장 가까운 마디 경계까지 계속 재생 후 전환.
        Returns: 전환 시작 토큰 위치
        """
        return self.find_next_bar(current_chunk, current_position)
