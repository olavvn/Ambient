"""
데이터 확인 스크립트.
data/processed/train, val 폴더의 pkl 파일 수와 샘플 내용을 출력한다.

사용법:
    python scripts/check_data.py --data_dir /path/to/data/processed
"""
import argparse
import pickle
from pathlib import Path


def check(data_dir: str):
    for split in ("train", "val"):
        split_dir = Path(data_dir) / split
        pkls = sorted(split_dir.glob("*.pkl"))
        print(f"[{split:5s}] {len(pkls)} 파일  ({split_dir})")
        if pkls:
            try:
                d = pickle.load(open(pkls[0], "rb"))
                tokens = d["tokens"]
                print(f"         샘플: {pkls[0].name}")
                print(f"         토큰 수: {len(tokens)}")
                print(f"         첫 토큰: {tokens[:8]}")
            except Exception as e:
                print(f"         읽기 오류: {e}")
        else:
            print(f"         ⚠️  pkl 파일 없음 — 변환 셀을 다시 실행하세요")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="data/processed 폴더 경로")
    args = parser.parse_args()
    check(args.data_dir)
