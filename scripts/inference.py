"""
추론 스크립트 — anchor MIDI를 입력받아 앰비언트 MIDI를 생성한다.

사용법:
    python scripts/inference.py \
        --ckpt       checkpoints/phase2_best.pt \
        --anchor     inputs/chord.mid \
        --out        outputs/generated.mid \
        --n_tokens   256 \
        --temperature 1.0 \
        --top_p      0.9
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pretty_midi
from src.model.anchorflow import AnchorFlowModel
from src.tokenizer import TSDTokenizer


def generate(
    ckpt_path: str,
    anchor_path: str,
    out_path: str,
    n_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = TSDTokenizer()

    print(f"모델 로드: {ckpt_path}")
    model = AnchorFlowModel.from_checkpoint(ckpt_path, device=device)
    model.eval()
    model.to(device)

    print(f"Anchor 로드: {anchor_path}")
    anchor_midi   = pretty_midi.PrettyMIDI(anchor_path)
    anchor_tokens = tokenizer.midi_to_tokens(anchor_midi)
    print(f"  → {len(anchor_tokens)} 토큰")

    context        = [tokenizer.bos_id] + anchor_tokens
    context_tensor = torch.tensor([context], dtype=torch.long, device=device)

    gen_kwargs = dict(
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id,
    )
    if top_k > 0:
        gen_kwargs["top_k"] = top_k

    print(f"생성 중... (max_new_tokens={n_tokens})")
    with torch.no_grad():
        output = model.generate(context_tensor, **gen_kwargs)

    new_tokens  = output[0, len(context):].tolist()
    all_tokens  = context + new_tokens
    out_midi    = tokenizer.tokens_to_midi(all_tokens)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_midi.write(out_path)

    note_count = sum(len(inst.notes) for inst in out_midi.instruments)
    print(f"저장: {out_path}")
    print(f"  음표 수: {note_count}")
    print(f"  길이:   {out_midi.get_end_time():.2f}초")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,  help="체크포인트 경로")
    parser.add_argument("--anchor",      required=True,  help="anchor MIDI 파일 경로")
    parser.add_argument("--out",         default="outputs/generated.mid", help="출력 MIDI 경로")
    parser.add_argument("--n_tokens",    type=int,   default=256,  help="생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=1.0,  help="샘플링 온도")
    parser.add_argument("--top_p",       type=float, default=0.9,  help="nucleus sampling p")
    parser.add_argument("--top_k",       type=int,   default=0,    help="top-k (0=비활성)")
    parser.add_argument("--device",      default=None, help="cuda / cpu (기본: 자동)")
    args = parser.parse_args()

    generate(
        ckpt_path=args.ckpt,
        anchor_path=args.anchor,
        out_path=args.out,
        n_tokens=args.n_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    )
