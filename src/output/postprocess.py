"""앰비언트 오디오 후처리"""
import numpy as np
from scipy.signal import butter, filtfilt


def smooth_dynamics(audio: np.ndarray, window_size: int = 4410) -> np.ndarray:
    """다이나믹스 스무딩 (RMS 정규화)"""
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / (rms + 1e-9) * 0.3
    return audio.astype(np.float32)


def apply_lowpass(audio: np.ndarray, cutoff_hz: float = 8000.0,
                  sr: int = 44100) -> np.ndarray:
    """고주파 감쇠 (앰비언트 특성)"""
    nyq = sr / 2.0
    b, a = butter(4, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, audio).astype(np.float32)
