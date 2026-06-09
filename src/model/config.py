"""모델 설정"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # 사전학습 모델
    pretrained_model_id: str = "stanford-crfm/music-medium-800k"
    vocab_size: int = 274          # TSD Ambient 어휘 크기
    reinit_embeddings: bool = True

    # 생성 파라미터
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    label_smoothing: float = 0.1
