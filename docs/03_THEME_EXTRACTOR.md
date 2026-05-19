# 03 테마 추출기 (Theme Extractor)

## 개요

ThemeTransformer의 핵심 아이디어를 기반으로, 입력 MIDI/오디오에서 **반복·변형되는 주제적 재료(thematic material)**를 자동으로 감지한다. 앰비언트 음악 특성을 고려하여 짧은 멜로디 모티브뿐 아니라 **하모니 텍스처 패턴**도 테마로 인식할 수 있도록 설계한다.

---

## 파이프라인 구조

```
입력 토큰 시퀀스
       ↓
  [1] 세그먼트 분할
  (고정 길이 윈도우 슬라이딩)
       ↓
  [2] 대조 학습 인코더
  (유사 패턴 → 임베딩 공간 근접)
       ↓
  [3] 클러스터링 (DBSCAN or K-Means)
  (반복 등장 패턴 그룹화)
       ↓
  [4] 테마 선택
  (가장 큰/중심적 클러스터의 centroid)
       ↓
  테마 토큰 시퀀스 + 테마 구간 레이블
```

---

## 1. 대조 학습 인코더

### `src/theme/contrastive.py`

```python
"""
대조 학습 기반 음악 세그먼트 인코더
유사 패시지 → 임베딩 공간에서 가깝게
다른 패시지 → 멀게
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SegmentEncoder(nn.Module):
    """
    음악 세그먼트를 고정 크기 벡터로 인코딩하는 Transformer 인코더.
    ThemeTransformer 논문의 contrastive encoder 구조를 따름.
    """

    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4,
                 max_seq_len: int = 128, proj_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Projection head (NT-Xent 학습용)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L) 토큰 ID 시퀀스
            padding_mask: (B, L) True = padding 위치
        Returns:
            z: (B, proj_dim) 정규화된 임베딩
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        # CLS-like pooling: 패딩 제외 평균
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        z = self.proj_head(h)
        return F.normalize(z, dim=-1)  # unit sphere로 정규화


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실.
    같은 원곡의 다른 구간 → positive pair,
    다른 원곡의 구간 → negative pair.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) 같은 원곡에서 나온 positive pair
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)              # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # 자기 자신 제외
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # positive: z1[i] ↔ z2[i]
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device)
        ])
        loss = F.cross_entropy(sim, labels)
        return loss
```

---

## 2. 클러스터링 모듈

### `src/theme/clustering.py`

```python
"""
임베딩 공간에서 반복 패턴을 클러스터링하여 테마 후보를 선별
"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ThemeCandidate:
    """테마 후보"""
    centroid: np.ndarray      # 클러스터 중심 임베딩
    member_indices: List[int] # 이 클러스터에 속하는 세그먼트 인덱스
    representative_idx: int   # 대표 세그먼트 인덱스 (centroid에 가장 가까운 것)
    score: float              # 테마 점수 (크기 × 분산 역수)


def cluster_segments(
    embeddings: np.ndarray,        # (N, D) 세그먼트 임베딩
    method: str = "dbscan",        # "dbscan" or "kmeans"
    eps: float = 0.3,              # DBSCAN epsilon (코사인 거리)
    min_samples: int = 3,          # DBSCAN 최소 클러스터 크기
    n_clusters: int = 5,           # KMeans k
) -> List[ThemeCandidate]:
    """
    임베딩 클러스터링 → 테마 후보 목록 반환
    결과는 score 내림차순 정렬
    """
    embeddings = normalize(embeddings)  # 코사인 거리용 L2 정규화

    if method == "dbscan":
        labels = DBSCAN(
            eps=eps, min_samples=min_samples,
            metric="cosine", algorithm="brute"
        ).fit_predict(embeddings)
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=42,
                        n_init="auto").fit_predict(embeddings)

    candidates: List[ThemeCandidate] = []
    unique_labels = set(labels) - {-1}  # -1 = 노이즈

    for lbl in unique_labels:
        member_idx = np.where(labels == lbl)[0].tolist()
        if len(member_idx) < 2:
            continue
        cluster_embs = embeddings[member_idx]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        # 대표 세그먼트: centroid에 가장 가까운 것
        dists = 1 - cluster_embs @ centroid
        rep_idx = member_idx[int(np.argmin(dists))]

        # 점수: 클러스터 크기 / 평균 내부 거리 (응집도)
        intra_dist = float(dists.mean())
        score = len(member_idx) / (intra_dist + 1e-6)

        candidates.append(ThemeCandidate(
            centroid=centroid,
            member_indices=member_idx,
            representative_idx=rep_idx,
            score=score,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_theme(candidates: List[ThemeCandidate],
                 strategy: str = "top1") -> ThemeCandidate:
    """
    테마 후보 중 최종 테마 선택.

    strategy:
        "top1"   - 가장 높은 score (기본)
        "diverse" - 상위 3개 중 가장 초반에 등장하는 것
    """
    if not candidates:
        raise ValueError("No theme candidates found")

    if strategy == "top1" or len(candidates) == 1:
        return candidates[0]
    elif strategy == "diverse":
        top3 = candidates[:3]
        return min(top3, key=lambda c: min(c.member_indices))
    else:
        return candidates[0]
```

---

## 3. 통합 테마 추출기

### `src/theme/extractor.py`

```python
"""
ThemeExtractor: MIDI / 토큰 시퀀스에서 테마를 추출하는 메인 인터페이스
"""
from __future__ import annotations
import torch
import numpy as np
import pretty_midi
from typing import List, Tuple, Optional

from src.theme.tokenizer import REMIAmbientTokenizer
from src.theme.contrastive import SegmentEncoder
from src.theme.clustering import cluster_segments, select_theme, ThemeCandidate


class ThemeExtractor:
    """
    입력 → 테마 토큰 시퀀스 추출 파이프라인.
    훈련된 SegmentEncoder 가중치를 로드하여 사용.
    """

    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        segment_len: int = 64,       # 세그먼트 길이 (토큰 수)
        stride: int = 32,            # 슬라이딩 윈도우 보폭
        cluster_method: str = "dbscan",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tokenizer = REMIAmbientTokenizer()
        self.encoder = SegmentEncoder(vocab_size=self.tokenizer.vocab_size)
        if encoder_checkpoint:
            state = torch.load(encoder_checkpoint, map_location=device)
            self.encoder.load_state_dict(state)
        self.encoder.eval().to(device)

        self.segment_len = segment_len
        self.stride = stride
        self.cluster_method = cluster_method
        self.device = device

    # ── 메인 인터페이스 ────────────────────────────────────────────────

    def extract_from_midi(
        self, midi: pretty_midi.PrettyMIDI
    ) -> Tuple[List[int], List[Tuple[float, float]]]:
        """
        MIDI → (테마 토큰 시퀀스, 테마 구간 [(start_sec, end_sec)])

        Returns:
            theme_tokens: 테마를 대표하는 토큰 리스트
            theme_spans : MIDI 시간 기준 테마 구간 목록
        """
        full_tokens = self.tokenizer.midi_to_tokens(midi)
        return self.extract_from_tokens(full_tokens, midi=midi)

    def extract_from_tokens(
        self, tokens: List[int],
        midi: Optional[pretty_midi.PrettyMIDI] = None,
    ) -> Tuple[List[int], List[Tuple[float, float]]]:
        """토큰 시퀀스 → (테마 토큰, 테마 구간)"""
        # 1. 세그먼트 분할
        segments = self._sliding_window(tokens)
        if len(segments) < 2:
            # 짧은 입력: 전체를 테마로 사용
            return tokens, []

        # 2. 임베딩
        embeddings = self._embed_segments(segments)

        # 3. 클러스터링 + 테마 선택
        candidates = cluster_segments(embeddings, method=self.cluster_method)
        if not candidates:
            # 클러스터 없으면 첫 세그먼트 반환
            return segments[0].tolist(), []
        theme = select_theme(candidates)

        # 4. 테마 토큰 추출
        theme_tokens = segments[theme.representative_idx].tolist()

        # 5. 테마 구간 계산 (MIDI 시간으로 역산, 근사치)
        theme_spans: List[Tuple[float, float]] = []
        if midi:
            dur = midi.get_end_time()
            tok_per_sec = len(tokens) / max(dur, 1.0)
            for idx in theme.member_indices:
                start_tok = idx * self.stride
                end_tok = start_tok + self.segment_len
                start_sec = start_tok / tok_per_sec
                end_sec = min(end_tok / tok_per_sec, dur)
                theme_spans.append((start_sec, end_sec))

        return theme_tokens, theme_spans

    # ── Helper ─────────────────────────────────────────────────────────

    def _sliding_window(self, tokens: List[int]) -> np.ndarray:
        """슬라이딩 윈도우로 세그먼트 생성"""
        segs = []
        for start in range(0, len(tokens) - self.segment_len + 1, self.stride):
            seg = tokens[start: start + self.segment_len]
            segs.append(seg)
        return np.array(segs, dtype=np.int64)

    @torch.no_grad()
    def _embed_segments(self, segments: np.ndarray) -> np.ndarray:
        """세그먼트 배열 → numpy 임베딩 배열"""
        batch_size = 64
        all_embs = []
        for i in range(0, len(segments), batch_size):
            batch = torch.tensor(
                segments[i: i + batch_size], dtype=torch.long, device=self.device
            )
            padding_mask = (batch == self.tokenizer.pad_id)
            emb = self.encoder(batch, padding_mask)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    def extract_theme_spans(
        self, midi: pretty_midi.PrettyMIDI
    ) -> List[Tuple[float, float]]:
        """테마 구간만 반환 (전처리 레이블링용 편의 메서드)"""
        _, spans = self.extract_from_midi(midi)
        return spans
```

---

## 4. 대조 학습 훈련

### `training/train_contrastive.py` 핵심 루프

```python
"""대조 학습으로 SegmentEncoder 훈련"""
import torch, wandb
from torch.utils.data import DataLoader
from src.theme.contrastive import SegmentEncoder, NTXentLoss
from training.dataset import ContrastiveDataset

# 같은 곡의 서로 다른 구간 쌍을 positive pair로 구성
dataset = ContrastiveDataset("data_pkl/", segment_len=64)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

encoder = SegmentEncoder(vocab_size=238, d_model=256, proj_dim=128).cuda()
loss_fn = NTXentLoss(temperature=0.07)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    total_loss = 0.0
    for seg1, seg2 in loader:   # (B, 64) 각각 같은 곡의 다른 구간
        seg1, seg2 = seg1.cuda(), seg2.cuda()
        z1 = encoder(seg1)
        z2 = encoder(seg2)
        loss = loss_fn(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
    torch.save(encoder.state_dict(), f"checkpoints/encoder_ep{epoch}.pt")
```

### `training/dataset.py` - ContrastiveDataset 구조
```python
class ContrastiveDataset(torch.utils.data.Dataset):
    """
    같은 곡 내의 두 랜덤 세그먼트를 positive pair로 반환.
    Positive pair 선택 전략:
    - 일반: 같은 곡의 임의 두 구간
    - Hard: 같은 곡 내 테마 레이블이 있는 구간 vs 비테마 구간 (Hard Negative)
    """
    ...
```
