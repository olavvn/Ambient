# 07 훈련 파이프라인 (Training Pipeline)

## 훈련 단계 개요

```
Phase 1: SegmentEncoder 대조 학습 (약 100 에폭)
              ↓
Phase 2: 테마 레이블 자동 생성 (전처리)
              ↓
Phase 3: AmbientFlowModel 생성 훈련 (약 200 에폭)
              ↓
Phase 4: 앰비언트 특화 파인튜닝 (약 50 에폭)
```

---

## 1. 데이터셋 클래스

### `training/dataset.py`

```python
"""
훈련용 데이터셋 클래스
"""
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import random


class ContrastiveDataset(Dataset):
    """
    Phase 1: 대조 학습용 - 같은 곡의 두 세그먼트 쌍
    """

    def __init__(self, data_dir: str, segment_len: int = 64,
                 hard_negative: bool = True):
        self.segment_len = segment_len
        self.hard_negative = hard_negative
        self.all_pieces = []

        for pkl_path in Path(data_dir).glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            tokens = data["tokens"]
            if len(tokens) >= segment_len * 2:
                self.all_pieces.append(tokens)

        print(f"ContrastiveDataset: {len(self.all_pieces)} 곡 로드")

    def __len__(self):
        return len(self.all_pieces) * 10  # 에폭당 과다표집

    def __getitem__(self, idx):
        piece = self.all_pieces[idx % len(self.all_pieces)]
        # 같은 곡에서 두 랜덤 구간 선택
        max_start = len(piece) - self.segment_len
        start1 = random.randint(0, max_start)
        start2 = random.randint(0, max_start)
        seg1 = piece[start1: start1 + self.segment_len]
        seg2 = piece[start2: start2 + self.segment_len]
        return torch.tensor(seg1, dtype=torch.long), \
               torch.tensor(seg2, dtype=torch.long)


class ThemeGenerationDataset(Dataset):
    """
    Phase 3: 생성 모델 훈련용 - (theme_tokens, target_tokens) 쌍
    테마 레이블이 있는 PKL 파일 필요
    """

    def __init__(self, data_dir: str,
                 max_theme_len: int = 128,
                 max_target_len: int = 512,
                 pad_id: int = 0):
        self.max_theme_len = max_theme_len
        self.max_target_len = max_target_len
        self.pad_id = pad_id
        self.samples: List[Tuple[List[int], List[int]]] = []

        for pkl_path in Path(data_dir).glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if "theme_tokens" not in data:
                continue
            theme = data["theme_tokens"][:max_theme_len]
            # 청크별로 샘플 생성
            for chunk in data.get("chunks", []):
                if len(chunk) >= 64:
                    self.samples.append((theme, chunk[:max_target_len]))

        print(f"ThemeGenerationDataset: {len(self.samples)} 샘플")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        theme, target = self.samples[idx]
        # 패딩
        theme_padded = theme + [self.pad_id] * (self.max_theme_len - len(theme))
        target_padded = target + [self.pad_id] * (self.max_target_len - len(target))
        theme_mask = [False] * len(theme) + [True] * (self.max_theme_len - len(theme))
        return {
            "theme_tokens": torch.tensor(theme_padded, dtype=torch.long),
            "target_tokens": torch.tensor(target_padded, dtype=torch.long),
            "theme_pad_mask": torch.tensor(theme_mask, dtype=torch.bool),
        }
```

---

## 2. 손실 함수

### `training/losses.py`

```python
"""
생성 모델 손실 함수
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThemeAwareCrossEntropy(nn.Module):
    """
    테마 토큰 위치에서 추가 가중치를 주는 Cross Entropy.
    [THEME_REF] 토큰 직후 예측에 더 높은 가중치 부여 →
    모델이 테마 참조 구간에서 더 정확하게 학습.
    """

    def __init__(self, vocab_size: int, label_smoothing: float = 0.1,
                 theme_ref_id: int = 6, theme_weight: float = 2.0,
                 pad_id: int = 0):
        super().__init__()
        self.theme_ref_id = theme_ref_id
        self.theme_weight = theme_weight
        self.pad_id = pad_id
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=pad_id,
            reduction='none'
        )

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, L, V)
            targets: (B, L)
        """
        B, L, V = logits.shape
        loss = self.ce(logits.view(B * L, V), targets.view(B * L))
        loss = loss.view(B, L)

        # [THEME_REF] 직후 위치에 가중치
        theme_ref_mask = (targets[:, :-1] == self.theme_ref_id).float()
        weight = torch.ones_like(loss)
        weight[:, 1:] += (self.theme_weight - 1) * theme_ref_mask

        # 패딩 마스크
        pad_mask = (targets != self.pad_id).float()
        loss = (loss * weight * pad_mask).sum() / pad_mask.sum().clamp(min=1)
        return loss
```

---

## 3. 생성 모델 훈련

### `training/train_generator.py`

```python
"""
AmbientFlowModel 훈련 스크립트
"""
import torch
import wandb
from torch.utils.data import DataLoader
from src.model.transformer import AmbientFlowModel
from src.model.config import ModelConfig
from training.dataset import ThemeGenerationDataset
from training.losses import ThemeAwareCrossEntropy


def train(config_path: str = "configs/model_config.yaml",
          training_config: str = "configs/training_config.yaml"):
    import yaml
    with open(config_path) as f:
        model_cfg_dict = yaml.safe_load(f)
    with open(training_config) as f:
        train_cfg = yaml.safe_load(f)

    cfg = ModelConfig(**model_cfg_dict)
    model = AmbientFlowModel(cfg).cuda()

    dataset = ThemeGenerationDataset(
        train_cfg["data_dir"],
        max_theme_len=cfg.max_theme_len,
        max_target_len=cfg.max_seq_len,
    )
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg["lr"],
        total_steps=train_cfg["epochs"] * len(loader)
    )
    loss_fn = ThemeAwareCrossEntropy(cfg.vocab_size, cfg.label_smoothing)

    wandb.init(project="ambientflow", config={**model_cfg_dict, **train_cfg})

    best_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in loader:
            theme = batch["theme_tokens"].cuda()
            target = batch["target_tokens"].cuda()
            theme_mask = batch["theme_pad_mask"].cuda()

            # Teacher forcing: input은 target[:-1], label은 target[1:]
            logits = model(theme, target[:, :-1], theme_mask)
            loss = loss_fn(logits, target[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"epoch": epoch, "loss": avg_loss,
                   "lr": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": model_cfg_dict,
            }, "checkpoints/best_model.pt")

    wandb.finish()


if __name__ == "__main__":
    train()
```

---

## 4. 훈련 설정

### `configs/training_config.yaml`

```yaml
# Phase 1: 대조 학습
contrastive:
  data_dir: "data_pkl/"
  epochs: 100
  batch_size: 256
  lr: 3.0e-4
  temperature: 0.07
  segment_len: 64
  checkpoint_dir: "checkpoints/contrastive/"
  save_every: 10

# Phase 3: 생성 모델
generation:
  data_dir: "data_pkl/"
  epochs: 200
  batch_size: 32           # seq2seq는 메모리 사용량 높음
  lr: 1.0e-4
  warmup_steps: 1000
  checkpoint_dir: "checkpoints/generation/"
  save_every: 5
  eval_every: 5
  gradient_clip: 1.0

# Phase 4: 앰비언트 파인튜닝
finetune:
  data_dir: "data_pkl/ambient/"   # 앰비언트 특화 데이터
  epochs: 50
  batch_size: 16
  lr: 5.0e-5               # 낮은 LR로 파인튜닝
  checkpoint_dir: "checkpoints/finetune/"
```

---

## 5. 평가 지표

### `scripts/evaluate.py`에 구현할 지표

| 지표 | 설명 | 도구 |
|------|------|------|
| **Theme Recall@K** | 생성 결과에서 테마가 K번 이상 반복되는 비율 | 코사인 유사도 + 임계값 |
| **Perplexity** | 언어 모델 품질 | `torch.exp(cross_entropy)` |
| **Pitch Class Entropy** | 음고 다양성 (앰비언트는 낮을수록 좋음) | numpy |
| **Inter-Onset Interval** | 리듬 패턴 규칙성 | pretty_midi |
| **MOS (주관적)** | 사람 평가 (1~5점) | 설문 |
| **FAD** | Fréchet Audio Distance (오디오 품질) | `fadtk` 라이브러리 |

```python
# scripts/evaluate.py 핵심 함수
def compute_theme_recall(
    generated_tokens: List[int],
    theme_tokens: List[int],
    encoder: SegmentEncoder,
    threshold: float = 0.7,
    k: int = 3,
) -> float:
    """생성 결과에서 테마 유사 구간이 k번 이상 등장하는지"""
    ...
```
