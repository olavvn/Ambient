"""
롤링 컨텍스트 윈도우 기반 청크 생성기
"""
import threading
import time
import torch
from typing import Optional

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

        bos_id = 1
        self.context = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        self._context_lock = threading.Lock()
        self._running = False

    def reset_context(self, new_context: torch.Tensor):
        """
        생성 컨텍스트를 외부에서 교체한다 (멜로디 시드 주입용).
        스레드 안전하게 처리.

        Args:
            new_context: (1, L) 토큰 텐서
        """
        new_context = new_context.to(self.device)
        if new_context.size(1) > self.context_len:
            new_context = new_context[:, -self.context_len:]
        with self._context_lock:
            self.context = new_context

    def start(self):
        """무한 생성 루프 시작 (별도 스레드에서 실행)"""
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
                time.sleep(0.1)
                continue

            theme = theme.to(self.device)
            if theme.dim() == 1:
                theme = theme.unsqueeze(0)

            self.theme_buffer.consume_new_flag()

            with self._context_lock:
                current_context = self.context

            new_tokens = self.model.generate_chunk(
                theme_tokens=theme,
                context=current_context,
                n_new_tokens=self.chunk_size,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
            )

            with self._context_lock:
                self.context = torch.cat([self.context, new_tokens], dim=1)
                if self.context.size(1) > self.context_len:
                    self.context = self.context[:, -self.context_len:]

            chunk_list = new_tokens[0].tolist()
            self.token_queue.put_nowait_threadsafe(chunk_list)
