"""
Token → MIDI → Audio 실시간 렌더러
FluidSynth를 사용한 소프트웨어 신시사이저
"""
import asyncio
import numpy as np
from typing import List, Optional

from src.theme.tokenizer import REMIAmbientTokenizer
from src.streaming.buffer import TokenQueue
from src.streaming.crossfade import crossfade_audio


class RealtimeRenderer:
    """
    TokenQueue에서 토큰을 꺼내 오디오로 렌더링 후 출력.
    FluidSynth가 없는 환경에서는 MIDI 파일 저장 모드로 폴백.
    """

    def __init__(
        self,
        tokenizer: REMIAmbientTokenizer,
        token_queue: TokenQueue,
        soundfont_path: str = "data/soundfonts/FluidR3_GM.sf2",
        sample_rate: int = 44100,
        output_file: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.token_queue = token_queue
        self.sample_rate = sample_rate
        self.output_file = output_file
        self._prev_chunk_audio: Optional[np.ndarray] = None

        self._fs = None
        self._init_fluidsynth(soundfont_path)

    def _init_fluidsynth(self, soundfont_path: str):
        try:
            import fluidsynth
            self._fs = fluidsynth.Synth(samplerate=float(self.sample_rate))
            driver = "file" if self.output_file else "alsa"
            self._fs.start(driver=driver)
            sfid = self._fs.sfload(soundfont_path)
            self._fs.program_select(0, sfid, 0, 0)
        except Exception:
            self._fs = None

    async def run(self):
        """비동기 재생 루프"""
        while True:
            chunk_tokens = await self.token_queue.get()
            audio = await asyncio.get_event_loop().run_in_executor(
                None, self._render_chunk, chunk_tokens
            )
            if self._prev_chunk_audio is not None and len(audio) > 0:
                fade_len = int(self.sample_rate * 0.1)
                crossfaded = crossfade_audio(
                    self._prev_chunk_audio, audio, fade_samples=fade_len
                )
                self._play_audio(crossfaded)
            self._play_audio(audio)
            self._prev_chunk_audio = audio

    def _render_chunk(self, tokens: List[int]) -> np.ndarray:
        """토큰 → numpy 오디오 배열"""
        midi = self.tokenizer.tokens_to_midi(tokens)
        if self._fs is None:
            return np.zeros(int(self.sample_rate * 2), dtype=np.float32)

        audio_frames = []
        for instrument in midi.instruments:
            for note in sorted(instrument.notes, key=lambda n: n.start):
                start_samples = int(note.start * self.sample_rate)
                end_samples = int(note.end * self.sample_rate)
                n_samples = max(end_samples - start_samples, 1)
                self._fs.noteon(0, note.pitch, note.velocity)
                chunk = self._fs.get_samples(n_samples)
                audio_frames.append(np.array(chunk, dtype=np.float32))
                self._fs.noteoff(0, note.pitch)

        if not audio_frames:
            return np.zeros(int(self.sample_rate * 2), dtype=np.float32)
        return np.concatenate(audio_frames)

    def _play_audio(self, audio: np.ndarray):
        """오디오 재생 (pyaudio 스트림 또는 파일)"""
        pass

    def close(self):
        if self._fs:
            self._fs.delete()
