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
