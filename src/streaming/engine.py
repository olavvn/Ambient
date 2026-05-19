"""
AmbientFlow 메인 엔진 - 모든 컴포넌트를 조율

테마(스타일) 소스와 입력 멜로디를 분리한다:
  - style_dir 있음 → StyleLibrary에서 블렌딩된 테마 사용
  - style_dir 없음 → 기존 방식 (입력에서 ThemeExtractor로 추출)

입력 멜로디는 항상 생성의 context 시드로 사용된다.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Optional, List

import torch

from src.input.normalizer import load_input
from src.theme.extractor import ThemeExtractor
from src.theme.tokenizer import REMIAmbientTokenizer
from src.model.transformer import AmbientFlowModel
from src.model.config import ModelConfig
from src.streaming.buffer import ThemeBuffer, TokenQueue
from src.streaming.chunk_generator import ChunkGenerator
from src.output.renderer import RealtimeRenderer
from src.utils.config_loader import load_yaml


class AmbientFlowEngine:
    """
    전체 파이프라인을 관리하는 엔진.

    스타일 폴더 모드:
        engine = AmbientFlowEngine(
            model_ckpt="checkpoints/best_model.pt",
            style_dir="styles/",          # 스타일 MIDI 폴더
        )
        engine.start()
        engine.set_style()                 # 폴더 전체 블렌딩
        engine.feed_melody("melody.mid")   # 멜로디는 context 시드로만 사용

    기존 모드 (입력에서 테마 추출):
        engine = AmbientFlowEngine(model_ckpt="checkpoints/best_model.pt")
        engine.start()
        engine.feed_input("my_melody.mid")
    """

    def __init__(
        self,
        model_ckpt: str,
        encoder_ckpt: Optional[str] = None,
        style_dir: Optional[str] = None,
        config_path: str = "configs/model_config.yaml",
        streaming_config: str = "configs/streaming_config.yaml",
        output_file: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        model_cfg_dict = load_yaml(config_path)
        self.stream_cfg = load_yaml(streaming_config)

        self.device = device
        self.cfg = ModelConfig(**model_cfg_dict)

        self.tokenizer = REMIAmbientTokenizer()

        # 스타일 라이브러리 (선택)
        self.style_library = None
        if style_dir:
            from src.theme.style_library import StyleLibrary
            self.style_library = StyleLibrary(
                style_dir=style_dir,
                max_theme_len=self.cfg.max_theme_len,
            )

        # 기존 방식 인코더 (style_dir 없을 때 사용)
        self.extractor = ThemeExtractor(
            encoder_checkpoint=encoder_ckpt, device=device
        )

        self.model = AmbientFlowModel(self.cfg)
        ckpt = torch.load(model_ckpt, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval().to(device)

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

    # ── 엔진 제어 ──────────────────────────────────────────────────────

    def start(self):
        """엔진 시작 — 생성 루프와 재생 루프를 백그라운드로"""
        self._loop = asyncio.new_event_loop()
        self.token_queue.set_loop(self._loop)

        def _run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._async_thread = threading.Thread(target=_run_loop, daemon=True)
        self._async_thread.start()

        asyncio.run_coroutine_threadsafe(self.renderer.run(), self._loop)
        self.chunk_gen.start()
        print("AmbientFlow 엔진 시작됨")

    def stop(self):
        """엔진 정지"""
        self.chunk_gen.stop()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.renderer.close()
        print("AmbientFlow 엔진 정지")

    # ── 스타일 폴더 모드 ───────────────────────────────────────────────

    def set_style(
        self,
        names: Optional[List[str]] = None,
        strategy: str = "front",
    ):
        """
        styles/ 폴더의 MIDI 파일들을 블렌딩하여 테마로 설정.
        입력 멜로디와 독립적으로 호출 가능.

        Args:
            names:    블렌딩할 파일명 리스트. None이면 폴더 전체 사용.
            strategy: "front" | "center" | "random"

        사용 예:
            engine.set_style()                          # 전체 블렌딩
            engine.set_style(["ballad.mid"])             # 단일 파일
            engine.set_style(["a.mid", "b.mid"], "center")
        """
        if not self.style_library:
            raise RuntimeError(
                "style_dir이 설정되지 않았습니다. "
                "AmbientFlowEngine(style_dir='styles/') 로 초기화하세요."
            )

        theme_tensor = self.style_library.get_blended_theme_tensor(
            names=names, strategy=strategy, device=self.device
        )
        self.theme_buffer.update(theme_tensor.squeeze(0))

        used = names or self.style_library.style_names
        print(f"스타일 설정: {used} (strategy={strategy}, "
              f"{theme_tensor.shape[1]} 토큰)")

    def reload_styles(self):
        """styles/ 폴더를 다시 스캔하여 새 파일 반영"""
        if not self.style_library:
            raise RuntimeError("style_dir이 설정되지 않았습니다.")
        self.style_library.reload()

    def list_styles(self) -> List[str]:
        """로드된 스타일 파일명 목록 반환"""
        if not self.style_library:
            return []
        return self.style_library.style_names

    def feed_melody(
        self,
        path: Optional[str] = None,
        realtime_port: Optional[str] = None,
        realtime_sec: float = 4.0,
    ):
        """
        입력 멜로디를 받아 생성 컨텍스트 시드로 사용.
        테마(스타일)는 변경하지 않는다 — set_style()로 독립 설정.

        Args:
            path:          MIDI 또는 오디오 파일 경로
            realtime_port: 실시간 MIDI 포트 이름
            realtime_sec:  실시간 수집 시간 (초)
        """
        print(f"멜로디 입력: {path or '실시간 MIDI'}")
        midi = load_input(path, realtime_port, realtime_sec)
        tokens = self.tokenizer.midi_to_tokens(midi)

        # 토큰을 context 시드로 삽입 (max_seq_len 이내로 자름)
        max_ctx = self.cfg.max_seq_len
        seed = tokens[:max_ctx]
        seed_tensor = torch.tensor(seed, dtype=torch.long,
                                   device=self.device).unsqueeze(0)

        # ChunkGenerator의 컨텍스트를 멜로디 시드로 교체
        self.chunk_gen.reset_context(seed_tensor)
        print(f"멜로디 시드: {len(seed)} 토큰")

    # ── 기존 모드 (하위 호환) ──────────────────────────────────────────

    def feed_input(
        self,
        path: Optional[str] = None,
        realtime_port: Optional[str] = None,
        realtime_sec: float = 4.0,
    ):
        """
        기존 방식: 입력에서 테마를 추출하고 context 시드로도 사용.
        style_dir 없이 사용할 때의 호환 메서드.
        """
        print(f"입력 처리 중: {path or '실시간 MIDI'}")
        midi = load_input(path, realtime_port, realtime_sec)
        theme_tokens, _ = self.extractor.extract_from_midi(midi)

        if not theme_tokens:
            print("테마 추출 실패. 이전 테마 유지.")
            return

        theme_tensor = torch.tensor(theme_tokens, dtype=torch.long)
        self.theme_buffer.update(theme_tensor)
        print(f"테마 업데이트: {len(theme_tokens)} 토큰")
