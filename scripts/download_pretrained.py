"""
AMT 사전학습 가중치 다운로드 스크립트.
stanford-crfm/music-medium-800k를 HuggingFace Hub에서 로컬 캐시로 다운로드한다.
"""
import argparse
from pathlib import Path


def download_amt(model_id: str = "stanford-crfm/music-medium-800k", cache_dir: str = None):
    from transformers import AutoModelForCausalLM, AutoConfig
    from huggingface_hub import snapshot_download

    print(f"다운로드 중: {model_id}")
    cache_path = snapshot_download(model_id, cache_dir=cache_dir)
    print(f"캐시 경로: {cache_path}")

    print("모델 로드 검증 중...")
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print("다운로드 완료.")
    return cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="stanford-crfm/music-medium-800k")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    download_amt(args.model_id, args.cache_dir)
