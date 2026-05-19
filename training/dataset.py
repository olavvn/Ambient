"""훈련용 데이터셋 클래스"""
import pickle
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple


class ContrastiveDataset(Dataset):
    """Phase 1: 대조 학습용 - 같은 곡의 두 세그먼트 쌍"""

    def __init__(self, data_dir: str, segment_len: int = 64,
                 hard_negative: bool = True):
        self.segment_len = segment_len
        self.hard_negative = hard_negative
        self.all_pieces: List[List[int]] = []

        for pkl_path in Path(data_dir).glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            tokens = data["tokens"]
            if len(tokens) >= segment_len * 2:
                self.all_pieces.append(tokens)

        print(f"ContrastiveDataset: {len(self.all_pieces)} 곡 로드")

    def __len__(self):
        return len(self.all_pieces) * 10

    def __getitem__(self, idx):
        piece = self.all_pieces[idx % len(self.all_pieces)]
        max_start = len(piece) - self.segment_len
        start1 = random.randint(0, max_start)
        start2 = random.randint(0, max_start)
        seg1 = piece[start1: start1 + self.segment_len]
        seg2 = piece[start2: start2 + self.segment_len]
        return (
            torch.tensor(seg1, dtype=torch.long),
            torch.tensor(seg2, dtype=torch.long),
        )


class ThemeGenerationDataset(Dataset):
    """Phase 3: 생성 모델 훈련용 - (theme_tokens, target_tokens) 쌍"""

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
            for chunk in data.get("chunks", []):
                if len(chunk) >= 64:
                    self.samples.append((theme, chunk[:max_target_len]))

        print(f"ThemeGenerationDataset: {len(self.samples)} 샘플")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        theme, target = self.samples[idx]
        theme_padded = theme + [self.pad_id] * (self.max_theme_len - len(theme))
        target_padded = target + [self.pad_id] * (self.max_target_len - len(target))
        theme_mask = [False] * len(theme) + [True] * (self.max_theme_len - len(theme))
        return {
            "theme_tokens": torch.tensor(theme_padded, dtype=torch.long),
            "target_tokens": torch.tensor(target_padded, dtype=torch.long),
            "theme_pad_mask": torch.tensor(theme_mask, dtype=torch.bool),
        }
