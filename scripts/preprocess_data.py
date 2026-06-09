"""
MIDI 파일 → pkl 변환 스크립트.
지정한 폴더의 MIDI 파일을 TSDTokenizer로 토큰화하여
out_dir/train, out_dir/val 폴더에 pkl로 저장한다.

사용법:
    python scripts/preprocess_data.py \
        --midi_dir  /path/to/midi_files \
        --out_dir   /path/to/data/processed \
        --train_ratio 0.9
"""
import argparse
import pickle
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi
from src.tokenizer import TSDTokenizer


def preprocess(midi_dir: str, out_dir: str, train_ratio: float = 0.9):
    tok = TSDTokenizer()

    midi_paths = sorted(Path(midi_dir).glob("**/*.mid")) + \
                 sorted(Path(midi_dir).glob("**/*.midi"))

    if not midi_paths:
        print(f"[ERROR] MIDI 파일을 찾을 수 없습니다: {midi_dir}")
        sys.exit(1)

    print(f"MIDI 파일 수: {len(midi_paths)}")

    split_idx = max(1, int(len(midi_paths) * train_ratio))
    splits = {
        "train": midi_paths[:split_idx],
        "val":   midi_paths[split_idx:] if split_idx < len(midi_paths) else midi_paths[-1:],
    }

    total_saved = 0
    for split_name, paths in splits.items():
        out_split = Path(out_dir) / split_name
        out_split.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, p in enumerate(paths):
            try:
                midi   = pretty_midi.PrettyMIDI(str(p))
                tokens = tok.midi_to_tokens(midi)
                if len(tokens) < 32:
                    print(f"  SKIP (짧음): {p.name}")
                    continue
                out_path = out_split / f"{p.stem}_{i:04d}.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump({"tokens": tokens}, f)
                saved += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{split_name}] {i+1}/{len(paths)} 처리 중...")
            except Exception as e:
                print(f"  WARN [{p.name}]: {e}")

        print(f"{split_name}: {saved}개 저장 → {out_split}")
        total_saved += saved

    print(f"\n완료: 총 {total_saved}개 pkl 파일 생성")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir",    required=True,  help="MIDI 파일 폴더")
    parser.add_argument("--out_dir",     required=True,  help="출력 폴더 (train/, val/ 생성됨)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="train 비율 (기본 0.9)")
    args = parser.parse_args()

    preprocess(args.midi_dir, args.out_dir, args.train_ratio)
