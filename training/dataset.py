"""표준 자기회귀 학습 데이터셋. 슬라이딩 윈도우."""
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class AmbientMIDIDataset(Dataset):
    """
    앰비언트 MIDI 토큰 시퀀스를 슬라이딩 윈도우로 분할하는 데이터셋.

    특별한 anchor 마킹 없음. 데이터 그대로 학습.
    data_dir/*.pkl 파일에서 {"tokens": [int, ...]} 포맷을 읽는다.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 2048,
        stride: int = 1024,
        pad_id: int = 0,
        bos_id: int = 1,
    ):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.samples: List[List[int]] = []

        pkl_paths = list(Path(data_dir).glob("*.pkl"))
        for pkl_path in pkl_paths:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            tokens = [self.bos_id] + list(data["tokens"])

            for start in range(0, max(1, len(tokens) - seq_len), stride):
                chunk = tokens[start : start + seq_len]
                if len(chunk) < seq_len // 2:
                    continue
                self.samples.append(chunk)

        print(f"AmbientMIDIDataset: {len(pkl_paths)} 파일, {len(self.samples)} 샘플")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tokens = self.samples[idx]
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.pad_id] * (self.seq_len - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels":    tokens[1:],
        }
