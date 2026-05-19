# 08 메인 인터페이스 및 실행 (Main Interface & CLI)

## 개요

`main.py`가 진입점이며, CLI로 파일 입력 모드와 실시간 MIDI 모드를 모두 지원한다. 생성은 백그라운드에서 계속 실행되며, 사용자는 언제든 새 입력을 추가할 수 있다.

---

## 1. 메인 오케스트레이터

### `src/streaming/engine.py`

```python
"""
AmbientFlow 메인 엔진 - 모든 컴포넌트를 조율
"""
from __future__ import annotations
import asyncio
import threading
import torch
from pathlib import Path
from typing import Optional

from src.input.normalizer import load_input
from src.theme.extractor import ThemeExtractor
from src.theme.tokenizer import REMIAmbientTokenizer
from src.model.transformer import AmbientFlowModel
from src.model.config import ModelConfig
from src.streaming.buffer import ThemeBuffer, TokenQueue
from src.streaming.chunk_generator import ChunkGenerator
from src.output.renderer import RealtimeRenderer


class AmbientFlowEngine:
    """
    전체 파이프라인을 관리하는 엔진.
    사용 패턴:
        engine = AmbientFlowEngine(model_ckpt="checkpoints/best_model.pt")
        engine.start()
        engine.feed_input("my_melody.mid")   # 처음 또는 새 입력
        engine.feed_input("new_theme.wav")   # 실행 중 새 입력
        engine.stop()
    """

    def __init__(
        self,
        model_ckpt: str,
        encoder_ckpt: Optional[str] = None,
        config_path: str = "configs/model_config.yaml",
        streaming_config: str = "configs/streaming_config.yaml",
        output_file: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        import yaml
        with open(config_path) as f:
            model_cfg_dict = yaml.safe_load(f)
        with open(streaming_config) as f:
            self.stream_cfg = yaml.safe_load(f)

        self.device = device
        self.cfg = ModelConfig(**model_cfg_dict)

        # 컴포넌트 초기화
        self.tokenizer = REMIAmbientTokenizer()
        self.extractor = ThemeExtractor(
            encoder_checkpoint=encoder_ckpt, device=device
        )
        # 모델 로드
        self.model = AmbientFlowModel(self.cfg)
        ckpt = torch.load(model_ckpt, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval().to(device)

        # 스트리밍 컴포넌트
        self.theme_buffer = ThemeBuffer()
        self.token_queue = TokenQueue(
            maxsize=self.stream_cfg["streaming"]["queue_maxsize"]
        )
        self.chunk_gen = ChunkGenerator(
            model=self.model,
            cfg=self.cfg,
            theme_buffer=self.theme_buffer,
            token_queue=self.token_queue,
            chunk_size=self.stream_cfg["streaming"]["chunk_size"],
            context_len=self.stream_cfg["streaming"]["context_len"],
            device=device,
        )
        self.renderer = RealtimeRenderer(
            tokenizer=self.tokenizer,
            token_queue=self.token_queue,
            soundfont_path=self.stream_cfg["fluidsynth"]["soundfont"],
            sample_rate=self.stream_cfg["audio"]["sample_rate"],
            output_file=output_file,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None

    def start(self):
        """엔진 시작 - 생성 루프와 재생 루프를 백그라운드로"""
        # asyncio 이벤트 루프를 별도 스레드에서 실행
        self._loop = asyncio.new_event_loop()
        self.token_queue.set_loop(self._loop)

        def _run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._async_thread = threading.Thread(target=_run_loop, daemon=True)
        self._async_thread.start()

        # 재생 태스크 등록
        asyncio.run_coroutine_threadsafe(self.renderer.run(), self._loop)

        # 생성 스레드 시작
        self.chunk_gen.start()
        print("🎵 AmbientFlow 엔진 시작됨")

    def feed_input(
        self,
        path: Optional[str] = None,
        realtime_port: Optional[str] = None,
        realtime_sec: float = 4.0,
    ):
        """
        새 입력 주입. 현재 재생 중에도 호출 가능.
        테마 버퍼를 업데이트 → 다음 청크부터 새 테마 반영.
        """
        print(f"📥 입력 처리 중: {path or '실시간 MIDI'}")
        midi = load_input(path, realtime_port, realtime_sec)
        theme_tokens, theme_spans = self.extractor.extract_from_midi(midi)

        if not theme_tokens:
            print("⚠️  테마 추출 실패. 이전 테마 유지.")
            return

        theme_tensor = torch.tensor(theme_tokens, dtype=torch.long)
        self.theme_buffer.update(theme_tensor)
        print(f"✅ 테마 업데이트: {len(theme_tokens)} 토큰, "
              f"{len(theme_spans)} 구간 감지")

    def stop(self):
        """엔진 정지"""
        self.chunk_gen.stop()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.renderer.close()
        print("⏹  AmbientFlow 엔진 정지")
```

---

## 2. CLI 인터페이스

### `main.py`

```python
"""
AmbientFlow 메인 CLI
사용법:
  python main.py --input melody.mid
  python main.py --input song.mp3 --output output.wav
  python main.py --realtime --port "MIDI Controller"
"""
import click
import threading
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from src.streaming.engine import AmbientFlowEngine

console = Console()


@click.command()
@click.option("--input", "-i", "input_path", default=None,
              help="초기 입력 파일 경로 (MIDI 또는 오디오)")
@click.option("--output", "-o", "output_path", default=None,
              help="출력 파일 경로 (None이면 실시간 재생)")
@click.option("--model", "-m", default="checkpoints/best_model.pt",
              help="생성 모델 체크포인트")
@click.option("--encoder", "-e", default="checkpoints/contrastive/best_encoder.pt",
              help="인코더 체크포인트")
@click.option("--realtime", is_flag=True, default=False,
              help="실시간 MIDI 입력 모드")
@click.option("--port", default=None,
              help="MIDI 포트 이름 (--realtime과 함께 사용)")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "cuda", "cpu"]),
              help="연산 장치")
def main(input_path, output_path, model, encoder, realtime, port, device):
    """🎵 AmbientFlow: 테마 기반 실시간 앰비언트 음악 생성기"""

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(Panel.fit(
        "[bold cyan]AmbientFlow[/bold cyan]\n"
        "Theme-based Real-time Ambient Music Generator",
        border_style="cyan"
    ))
    console.print(f"장치: [yellow]{device}[/yellow]")

    # 엔진 초기화
    engine = AmbientFlowEngine(
        model_ckpt=model,
        encoder_ckpt=encoder,
        output_file=output_path,
        device=device,
    )
    engine.start()

    # 초기 입력
    if input_path:
        engine.feed_input(path=input_path)
    elif realtime:
        console.print(f"실시간 MIDI 모드: {port or '첫 번째 포트'} ({4}초 수집)")
        engine.feed_input(realtime_port=port, realtime_sec=4.0)
    else:
        console.print("[yellow]입력 없이 시작 - 'f' 명령으로 파일 로드 가능[/yellow]")

    # 인터랙티브 루프
    console.print("\n[dim]명령어: [f]ile 새 파일, [r]ealtime MIDI, [q]uit[/dim]\n")
    _interactive_loop(engine, port)


def _interactive_loop(engine: AmbientFlowEngine, default_port: Optional[str]):
    """실행 중 사용자 명령 처리"""
    while True:
        try:
            cmd = Prompt.ask("", default="").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd.startswith("f ") or cmd.startswith("file "):
            path = cmd.split(None, 1)[1].strip()
            engine.feed_input(path=path)
        elif cmd == "f":
            path = Prompt.ask("파일 경로")
            engine.feed_input(path=path)
        elif cmd in ("r", "realtime"):
            console.print("MIDI 수집 시작 (4초)...")
            threading.Thread(
                target=lambda: engine.feed_input(
                    realtime_port=default_port, realtime_sec=4.0
                ), daemon=True
            ).start()
        elif cmd == "":
            pass
        else:
            console.print(f"[red]알 수 없는 명령: {cmd}[/red]")

    engine.stop()
    console.print("[green]종료[/green]")


if __name__ == "__main__":
    main()
```

---

## 3. 실행 예시

### 파일 입력 → 실시간 재생
```bash
python main.py --input examples/moonlight_sonata.mid --device cuda
```

### 오디오 입력 → 파일 저장
```bash
python main.py \
  --input examples/guitar_loop.wav \
  --output outputs/ambient_generated.wav \
  --device cuda
```

### 실시간 MIDI 키보드 모드
```bash
python main.py --realtime --port "USB MIDI Keyboard" --device cuda
```

### 빠른 테스트 (CPU, 소형 모델)
```bash
python main.py \
  --input examples/test.mid \
  --model checkpoints/small_model.pt \
  --device cpu
```

---

## 4. Python API (코드에서 직접 사용)

```python
from src.streaming.engine import AmbientFlowEngine

# 엔진 초기화 및 시작
engine = AmbientFlowEngine(
    model_ckpt="checkpoints/best_model.pt",
    encoder_ckpt="checkpoints/contrastive/best_encoder.pt",
    output_file="outputs/session1.wav",  # 파일로 저장
)
engine.start()

# 초기 테마 설정
engine.feed_input(path="inputs/theme1.mid")

import time
time.sleep(30)  # 30초 생성

# 테마 교체
engine.feed_input(path="inputs/theme2.wav")
time.sleep(30)  # 추가 30초

engine.stop()
```
