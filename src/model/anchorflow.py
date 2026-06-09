"""
AnchorFlow 모델 래퍼.
HuggingFace의 AMT(stanford-crfm/music-medium-800k) 사전학습 가중치를 로드하고
TSD Ambient 토크나이저 어휘 크기에 맞게 임베딩/LM head를 교체한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional


def load_pretrained_amt(model_id: str = "stanford-crfm/music-medium-800k"):
    """HuggingFace에서 AMT 체크포인트 로드"""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model


def resize_and_reinit_embeddings(model, new_vocab_size: int) -> None:
    """
    임베딩 레이어와 LM head를 새 어휘 크기로 교체 후 재초기화.
    트랜스포머 블록 가중치는 유지된다.
    """
    model.resize_token_embeddings(new_vocab_size)
    nn.init.normal_(model.get_input_embeddings().weight, mean=0.0, std=0.02)
    nn.init.normal_(model.get_output_embeddings().weight, mean=0.0, std=0.02)


class AnchorFlowModel(nn.Module):
    """
    AMT 사전학습 가중치 + TSD Ambient 어휘를 결합한 래퍼.
    generate() 메서드는 HuggingFace의 generate()를 그대로 위임한다.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stanford-crfm/music-medium-800k",
        vocab_size: Optional[int] = None,
        reinit_embeddings: bool = True,
    ) -> "AnchorFlowModel":
        """사전학습 모델 로드 + 선택적 임베딩 교체"""
        hf_model = load_pretrained_amt(model_id)
        if vocab_size is not None and vocab_size != hf_model.config.vocab_size:
            resize_and_reinit_embeddings(hf_model, vocab_size)
        elif vocab_size is not None and not reinit_embeddings:
            hf_model.resize_token_embeddings(vocab_size)
        return cls(hf_model)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "AnchorFlowModel":
        """파인튜닝된 체크포인트에서 로드"""
        ckpt = torch.load(ckpt_path, map_location=device)
        config = ckpt.get("config", {})
        model_id = config.get("pretrained_model_id", "stanford-crfm/music-medium-800k")
        vocab_size = config.get("vocab_size", 274)

        hf_config = AutoConfig.from_pretrained(model_id)
        hf_config.vocab_size = vocab_size
        hf_model = AutoModelForCausalLM.from_config(hf_config)
        hf_model.load_state_dict(ckpt["model_state_dict"])
        return cls(hf_model)

    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
