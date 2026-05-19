# 05 실시간 스트리밍 엔진 (Streaming Engine)

## 개요

Magenta RealTime의 **연속 생성 패러다임**을 구현한다. 생성 스레드와 재생 스레드를 분리하고, 비동기 큐로 연결하여 끊김 없는 오디오 스트림을 만든다. 새 입력이 들어오면 **크로스페이드**로 부드럽게 전환한다.

---

## 아키텍처 다이어그램

```
[Input Thread]         [Generation Thread]        [Playback Thread]
     │                        │                         │
     │ new MIDI/Audio          │                         │
     ▼                        │                         │
 ThemeExtractor               │                         │
     │ theme_tokens            │                         │
     ▼                        │                         │
 ThemeBuffer ──────────────▶ AmbientFlowModel          │
 (thread-safe)                │                         │
                              │ chunk tokens            │
                              ▼                         │
                          TokenQueue ──────────────▶ Renderer
                          (asyncio.Queue)               │
                                                        │ audio bytes
                                                        ▼
                                                   FluidSynth
                                                        │
                                                        ▼
                                                  Speaker / File
```

---

## 1. 비동기 버퍼 및 큐

### `src/streaming/buffer.py`

```python
"""
스레드 안전 버퍼 및 비동기 큐
생성 스레드 ↔ 재생 스레드 간 토큰 전달
"""
import asyncio
import threading
from typing import List, Optional
import torch


class ThemeBuffer:
    """
    현재 활성 테마를 안전하게 교체하는 스레드 안전 버퍼.
    새 입력 → 새 테마로 원자적 교체.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._theme_tokens: Optional[torch.Tensor] = None
        self._is_new = threading.Event()

    def update(self, theme_tokens: torch.Tensor):
        """새 테마 설정 (입력 스레드에서 호출)"""
        with self._lock:
            self._theme_tokens = theme_tokens.clone()
            self._is_new.set()

    def get(self) -> Optional[torch.Tensor]:
        """현재 테마 반환 (생성 스레드에서 호출)"""
        with self._lock:
            return self._theme_tokens.clone() if self._theme_tokens is not None else None

    def consume_new_flag(self) -> bool:
        """새 테마가 있는지 확인 후 플래그 초기화"""
        if self._is_new.is_set():
            self._is_new.clear()
            return True
        return False


class TokenQueue:
    """
    생성된 토큰 청크를 저장하는 asyncio 기반 큐.
    큐 크기 제한으로 생성 속도를 재생 속도에 맞춤 (backpressure).
    """

    def __init__(self, maxsize: int = 4):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def put_nowait_threadsafe(self, chunk: List[int]):
        """별도 스레드에서 호출 (생성 스레드)"""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self.queue.put(chunk), self._loop
            )

    async def get(self) -> List[int]:
        """재생 스레드에서 대기 후 반환"""
        return await self.queue.get()
```

---

## 2. 크로스페이드 모듈

### `src/streaming/crossfade.py`

```python
"""
테마 전환 시 오디오 크로스페이드
MIDI 레벨이 아닌 오디오 버퍼 레벨에서 수행
"""
import numpy as np
from scipy.signal import butter, filtfilt


def crossfade_audio(
    audio_out: np.ndarray,   # 현재 재생 중인 오디오 (sr=44100, mono)
    audio_in: np.ndarray,    # 새로 시작될 오디오
    fade_samples: int = 22050,  # 크로스페이드 길이 (0.5초 @ 44100)
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

    # 길이 맞춤
    n = min(len(fade_out), len(fade_in), fade_samples)
    t = np.linspace(0, np.pi / 2, n)

    # Equal-power crossfade (앰비언트에 적합)
    gain_out = np.cos(t)   # 1 → 0
    gain_in = np.sin(t)    # 0 → 1

    return fade_out[:n] * gain_out + fade_in[:n] * gain_in


class TokenLevelCrossfade:
    """
    MIDI 토큰 레벨 크로스페이드.
    현재 청크를 자연스러운 마디 경계(BAR 토큰)에서 종료하고
    새 테마의 생성으로 전환.
    """

    def __init__(self, tokenizer, bar_token_id: int):
        self.tokenizer = tokenizer
        self.bar_token_id = bar_token_id

    def find_next_bar(self, tokens: List[int], from_pos: int,
                      max_lookahead: int = 64) -> int:
        """from_pos 이후 첫 번째 BAR 토큰 위치 반환"""
        for i in range(from_pos, min(from_pos + max_lookahead, len(tokens))):
            if tokens[i] == self.bar_token_id:
                return i
        return from_pos + max_lookahead  # BAR 없으면 강제 종료

    def plan_transition(self, current_chunk: List[int],
                        current_position: int) -> int:
        """
        현재 재생 위치에서 가장 가까운 마디 경계까지 계속 재생 후 전환.
        Returns: 전환 시작 토큰 위치
        """
        return self.find_next_bar(current_chunk, current_position)
```

---

## 3. 청크 생성기

### `src/streaming/chunk_generator.py`

```python
"""
롤링 컨텍스트 윈도우 기반 청크 생성기
"""
import torch
from typing import List, Optional
from src.model.transformer import AmbientFlowModel
from src.model.config import ModelConfig
from src.streaming.buffer import ThemeBuffer, TokenQueue


class ChunkGenerator:
    """
    무한 루프로 청크를 생성하며 TokenQueue에 넣는다.

    - 롤링 컨텍스트: 이전 생성 토큰을 다음 생성의 입력으로
    - 테마 변경 감지: ThemeBuffer에서 새 테마 확인
    - 청크 크기: 128 토큰 = 약 8~16마디 분량
    """

    def __init__(
        self,
        model: AmbientFlowModel,
        cfg: ModelConfig,
        theme_buffer: ThemeBuffer,
        token_queue: TokenQueue,
        chunk_size: int = 128,
        context_len: int = 512,
        device: str = "cuda",
    ):
        self.model = model.eval().to(device)
        self.cfg = cfg
        self.theme_buffer = theme_buffer
        self.token_queue = token_queue
        self.chunk_size = chunk_size
        self.context_len = context_len
        self.device = device

        # 초기 컨텍스트: BOS 토큰
        bos_id = 1  # tokenizer.bos_id
        self.context = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        self._running = False

    def start(self):
        """무한 생성 루프 시작 (별도 스레드에서 실행)"""
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=5.0)

    def _generate_loop(self):
        while self._running:
            theme = self.theme_buffer.get()
            if theme is None:
                import time; time.sleep(0.1)
                continue

            theme = theme.to(self.device)
            if theme.dim() == 1:
                theme = theme.unsqueeze(0)  # (1, T)

            # 새 테마가 감지되면 컨텍스트 유지하되 플래그만 초기화
            if self.theme_buffer.consume_new_flag():
                # 컨텍스트는 유지 → 부드러운 전환 (자르지 않음)
                pass

            # 청크 생성
            new_tokens = self.model.generate_chunk(
                theme_tokens=theme,
                context=self.context,
                n_new_tokens=self.chunk_size,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
            )

            # 롤링 컨텍스트 업데이트
            self.context = torch.cat([self.context, new_tokens], dim=1)
            if self.context.size(1) > self.context_len:
                self.context = self.context[:, -self.context_len:]

            # 큐에 넣기
            chunk_list = new_tokens[0].tolist()
            self.token_queue.put_nowait_threadsafe(chunk_list)
```

---

## 4. 오디오 렌더러

### `src/output/renderer.py`

```python
"""
Token → MIDI → Audio 실시간 렌더러
FluidSynth를 사용한 소프트웨어 신시사이저
"""
import asyncio
import numpy as np
import fluidsynth
import pretty_midi
from src.theme.tokenizer import REMIAmbientTokenizer
from src.streaming.buffer import TokenQueue
from src.streaming.crossfade import crossfade_audio


class RealtimeRenderer:
    """
    TokenQueue에서 토큰을 꺼내 오디오로 렌더링 후 출력.
    """

    def __init__(
        self,
        tokenizer: REMIAmbientTokenizer,
        token_queue: TokenQueue,
        soundfont_path: str = "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        sample_rate: int = 44100,
        output_file: Optional[str] = None,  # None이면 실시간 재생
    ):
        self.tokenizer = tokenizer
        self.token_queue = token_queue
        self.sample_rate = sample_rate
        self.output_file = output_file

        # FluidSynth 초기화
        self.fs = fluidsynth.Synth(samplerate=float(sample_rate))
        self.fs.start(driver="alsa" if output_file is None else "file")
        sfid = self.fs.sfload(soundfont_path)
        self.fs.program_select(0, sfid, 0, 0)  # Piano

        self._audio_buffer = np.array([], dtype=np.float32)
        self._prev_chunk_audio: Optional[np.ndarray] = None

    async def run(self):
        """비동기 재생 루프"""
        while True:
            chunk_tokens = await self.token_queue.get()
            audio = await asyncio.get_event_loop().run_in_executor(
                None, self._render_chunk, chunk_tokens
            )
            # 크로스페이드 (이전 청크와)
            if self._prev_chunk_audio is not None and len(audio) > 0:
                fade_len = int(self.sample_rate * 0.1)  # 100ms 크로스페이드
                crossfaded = crossfade_audio(
                    self._prev_chunk_audio, audio, fade_samples=fade_len
                )
                self._play_audio(crossfaded)
            self._play_audio(audio)
            self._prev_chunk_audio = audio

    def _render_chunk(self, tokens: List[int]) -> np.ndarray:
        """토큰 → numpy 오디오 배열"""
        midi = self.tokenizer.tokens_to_midi(tokens)
        # MIDI 이벤트 → FluidSynth
        audio_frames = []
        for instrument in midi.instruments:
            for note in sorted(instrument.notes, key=lambda n: n.start):
                # note on → 기다림 → note off
                start_samples = int(note.start * self.sample_rate)
                end_samples = int(note.end * self.sample_rate)
                self.fs.noteon(0, note.pitch, note.velocity)
                chunk = self.fs.get_samples(end_samples - start_samples)
                audio_frames.append(np.array(chunk, dtype=np.float32))
                self.fs.noteoff(0, note.pitch)
        if not audio_frames:
            return np.zeros(int(self.sample_rate * 2), dtype=np.float32)
        return np.concatenate(audio_frames)

    def _play_audio(self, audio: np.ndarray):
        """오디오 재생 (pyaudio 또는 파일 출력)"""
        # pyaudio stream 또는 soundfile write
        pass  # 실제 구현에서는 pyaudio.PyAudio stream 사용

    def close(self):
        self.fs.delete()
```

---

## 5. 스트리밍 설정

### `configs/streaming_config.yaml`
```yaml
# 스트리밍 엔진 설정
streaming:
  chunk_size: 128          # 청크당 토큰 수
  context_len: 512         # 롤링 컨텍스트 윈도우 크기
  queue_maxsize: 4          # 최대 선읽기 청크 수 (레이턴시 vs 안정성)
  device: "cuda"

# 오디오 출력
audio:
  sample_rate: 44100
  channels: 1               # 모노 (앰비언트)
  buffer_size: 1024
  crossfade_ms: 200         # 크로스페이드 길이

# FluidSynth
fluidsynth:
  soundfont: "data/soundfonts/FluidR3_GM.sf2"
  reverb: true
  reverb_room: 0.8          # 큰 공간감 (앰비언트)
  reverb_wet: 0.4
  chorus: false
  gain: 0.7

# MIDI 입력
midi_input:
  port: null                # null이면 파일 입력만
  buffer_sec: 2.0           # 실시간 MIDI 수집 시간

# 테마 전환
transition:
  strategy: "bar_boundary"  # 마디 경계에서 전환
  max_wait_bars: 4          # 최대 4마디까지 기다린 후 전환
```
