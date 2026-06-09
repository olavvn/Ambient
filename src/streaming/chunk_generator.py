"""
자기회귀 청크 생성기 + anchor 리터럴 삽입.
상태머신 없음. 단순 무한 루프.
"""
import threading
import torch
from typing import Optional

from src.streaming.buffer import AnchorQueue, TokenQueue


class ChunkGenerator:
    """
    자기회귀 청크 생성 + anchor 리터럴 삽입.

    매 청크 시작 시 AnchorQueue를 확인하여:
    - 대기 중인 anchor가 있으면 context에 리터럴 삽입 + TokenQueue로 송출
    - 다음 청크를 자기회귀로 생성하여 context 확장 + TokenQueue로 송출
    """

    def __init__(
        self,
        model,
        tokenizer,
        anchor_queue: AnchorQueue,
        token_queue: TokenQueue,
        chunk_size: int = 64,
        max_context: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.anchor_queue = anchor_queue
        self.token_queue = token_queue
        self.chunk_size = chunk_size
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k if top_k > 0 else None
        self.device = device

        self.context = torch.tensor(
            [[self.tokenizer.bos_id]], dtype=torch.long, device=device
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)

    def _loop(self):
        while self._running:
            # 1단계: 대기 중인 anchor 처리
            pending_anchors = self.anchor_queue.pop_all()
            for anchor_tokens in pending_anchors:
                anchor_tensor = torch.tensor(
                    [anchor_tokens], dtype=torch.long, device=self.device
                )
                self.context = torch.cat([self.context, anchor_tensor], dim=1)
                self.token_queue.put(anchor_tokens)

            # 2단계: 다음 청크 자기회귀 생성
            new_tokens = self._generate_chunk(self.chunk_size)
            self.context = torch.cat([self.context, new_tokens], dim=1)
            self.token_queue.put(new_tokens[0].tolist())

            # 3단계: context 길이 관리
            self._trim_context()

    @torch.no_grad()
    def _generate_chunk(self, n_tokens: int) -> torch.Tensor:
        gen_kwargs = dict(
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_id,
            eos_token_id=self.tokenizer.eos_id,
        )
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k

        generated = self.model.generate(self.context, **gen_kwargs)
        return generated[:, self.context.size(1):]

    def _trim_context(self):
        if self.context.size(1) > self.max_context:
            keep = self.max_context - self.chunk_size * 2
            self.context = self.context[:, -keep:]
