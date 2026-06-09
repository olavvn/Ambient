"""
AnchorFlowEngine - 단일 공개 인터페이스.
feed_anchor() + start() + stop() + get_output_tokens()
"""
from __future__ import annotations

import queue
from typing import Optional, List

import pretty_midi
import torch

from src.tokenizer import TSDTokenizer
from src.model.anchorflow import AnchorFlowModel
from src.streaming.buffer import AnchorQueue, TokenQueue
from src.streaming.chunk_generator import ChunkGenerator
from src.utils.config_loader import load_yaml


class AnchorFlowEngine:
    """
    실시간 앰비언트 MIDI 생성 엔진.

    사용 예:
        engine = AnchorFlowEngine(
            model_ckpt="checkpoints/anchorflow_best.pt",
            config_path="configs/model_config.yaml",
        )
        engine.feed_anchor("inputs/intro_chord.mid")
        engine.start()
        # 30초 후
        engine.feed_anchor("inputs/mood_change.mid")
        engine.stop()
    """

    def __init__(
        self,
        model_ckpt: str,
        config_path: str = "configs/model_config.yaml",
        streaming_config: str = "configs/streaming_config.yaml",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_cfg = load_yaml(config_path)
        stream_cfg = load_yaml(streaming_config).get("streaming", {})

        self.tokenizer = TSDTokenizer()
        self.model = AnchorFlowModel.from_checkpoint(model_ckpt, device=device)
        self.model.eval()
        self.model.to(device)

        self.anchor_queue = AnchorQueue()
        self.token_queue = TokenQueue(maxsize=stream_cfg.get("queue_maxsize", 8))
        self.chunk_gen = ChunkGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            anchor_queue=self.anchor_queue,
            token_queue=self.token_queue,
            chunk_size=stream_cfg.get("chunk_size", 64),
            max_context=stream_cfg.get("max_context", 2048),
            temperature=stream_cfg.get("temperature", 1.0),
            top_p=stream_cfg.get("top_p", 0.9),
            top_k=stream_cfg.get("top_k", 0),
            device=device,
        )

    def feed_anchor(self, midi_path: str) -> None:
        """
        사용자 MIDI 입력을 anchor로 등록.
        호출 즉시 반환 (실제 삽입은 다음 청크 경계에서).

        - 첫 호출: 곡의 시작점
        - 이후 호출: 화성 변화 지점
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        tokens = self.tokenizer.midi_to_tokens(midi)
        self.anchor_queue.push(tokens)
        print(f"[AnchorFlow] Anchor queued: {len(tokens)} tokens from {midi_path}")

    def feed_anchor_tokens(self, tokens: List[int]) -> None:
        """토큰 리스트를 직접 anchor로 등록 (테스트/고급 사용)"""
        self.anchor_queue.push(tokens)
        print(f"[AnchorFlow] Anchor queued: {len(tokens)} tokens")

    def start(self) -> None:
        """생성 시작. 최소 1개의 anchor가 큐에 있어야 함."""
        if not self.anchor_queue.has_pending():
            raise RuntimeError(
                "최소 1개의 anchor가 필요합니다. feed_anchor()를 먼저 호출하세요."
            )
        self.chunk_gen.start()
        print("[AnchorFlow] 생성 시작")

    def stop(self) -> None:
        self.chunk_gen.stop()
        print("[AnchorFlow] 생성 정지")

    def get_output_tokens(self, timeout: float = 0.1) -> Optional[List[int]]:
        """출력 토큰 청크 가져오기. 없으면 None 반환."""
        try:
            return self.token_queue.get(timeout=timeout)
        except queue.Empty:
            return None
