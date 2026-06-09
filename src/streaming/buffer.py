"""
스레드 안전 큐: AnchorQueue (사용자 anchor 입력) + TokenQueue (출력 토큰)
"""
import queue
import threading
from typing import List, Optional


class AnchorQueue:
    """
    사용자가 입력한 anchor 토큰들의 FIFO 큐.
    feed_anchor()로 추가, ChunkGenerator가 다음 청크 시작 시 소비.
    """

    def __init__(self):
        self._queue = queue.Queue()

    def push(self, tokens: List[int]):
        """사용자 입력 anchor를 큐에 추가 (thread-safe)"""
        self._queue.put(tokens)

    def pop_all(self) -> List[List[int]]:
        """대기 중인 모든 anchor를 한 번에 가져옴"""
        result = []
        while True:
            try:
                result.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return result

    def has_pending(self) -> bool:
        return not self._queue.empty()


class TokenQueue:
    """
    생성된 토큰 청크를 저장하는 스레드 안전 큐.
    consumer가 get()으로 읽는다.
    """

    def __init__(self, maxsize: int = 8):
        self._queue = queue.Queue(maxsize=maxsize)

    def put(self, tokens: List[int], block: bool = True, timeout: Optional[float] = None):
        self._queue.put(tokens, block=block, timeout=timeout)

    def get(self, timeout: Optional[float] = None) -> List[int]:
        return self._queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()
