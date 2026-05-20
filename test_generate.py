import torch
import pretty_midi
import click
from src.model.config import ModelConfig
from src.model.transformer import AmbientFlowModel
from src.theme.tokenizer import REMIAmbientTokenizer
from src.input.normalizer import load_input
from src.theme.extractor import ThemeExtractor
from src.utils.config_loader import load_yaml

@click.command()
@click.option("--input", "input_path", required=True, help="입력 미디 파일 경로")
@click.option("--output", "output_path", required=True, help="출력될 생성 미디 파일 경로")
@click.option("--model_ckpt", default="checkpoints/best_model.pt", help="생성 모델 체크포인트 경로")
@click.option("--tokens", "n_new_tokens", default=256, help="생성할 새로운 토큰 개수")
@click.option("--style-dir", "style_dir", default=None, help="스타일 MIDI 폴더 경로 (이것을 설정하면 입력 멜로디 대신 여기서 테마를 뽑습니다)")
def main(input_path, output_path, model_ckpt, n_new_tokens, style_dir):
    """
    입력 MIDI를 받아 테마를 추출하고, 새로운 MIDI 구간을 이어서 생성하는 테스트 스크립트.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[설정] 연산 장치: {device}")
    
    # 1. 모델과 토크나이저 초기화
    cfg_dict = load_yaml("configs/model_config.yaml")
    cfg = ModelConfig(**cfg_dict)
    model = AmbientFlowModel(cfg).to(device)
    try:
        state = torch.load(model_ckpt, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"[완료] 모델 체크포인트 로드: {model_ckpt}")
    except Exception as e:
        print(f"[경고] 모델 체크포인트를 불러오지 못했습니다. (학습되지 않은 가중치로 진행합니다): {e}")
    model.eval()
    
    tokenizer = REMIAmbientTokenizer()
    
    # 2. 인풋 미디 로드 및 토큰 변환
    print(f"\n[진행] 입력 파일 로드 중: {input_path}")
    midi = load_input(path=input_path, realtime_port=None, realtime_sec=0)
    full_tokens = tokenizer.midi_to_tokens(midi)
    
    # 3. 테마 및 컨텍스트 추출
    if style_dir:
        from src.theme.style_library import StyleLibrary
        print(f"[진행] 스타일 폴더에서 테마 블렌딩 중: {style_dir}")
        style_lib = StyleLibrary(style_dir=style_dir, max_theme_len=cfg.max_theme_len)
        theme_tensor = style_lib.get_blended_theme_tensor(device=device)
        theme_tokens = theme_tensor[0].cpu().tolist()
    else:
        extractor = ThemeExtractor(device=device)
        theme_tokens, _ = extractor.extract_from_tokens(full_tokens, midi=midi)
        if not theme_tokens:
            theme_tokens = full_tokens[:64]
        theme_tensor = torch.tensor(theme_tokens, dtype=torch.long, device=device).unsqueeze(0)

    
    # 컨텍스트 (최대 길이로 자름)
    max_ctx = cfg.max_seq_len
    seed_tokens = full_tokens[:max_ctx]
    context_tensor = torch.tensor(seed_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # 4. MIDI 텍스트(토큰) 생성
    print(f"[진행] 생성 중... (테마 토큰: {len(theme_tokens)}개, 시드 토큰: {len(seed_tokens)}개 -> 추가 생성: {n_new_tokens}개)")
    with torch.no_grad():
        generated_tokens_tensor = model.generate_chunk(
            theme_tokens=theme_tensor, 
            context=context_tensor, 
            n_new_tokens=n_new_tokens, 
            temperature=0.95, 
            top_p=0.92
        )
    
    # 5. 결과물 저장
    generated_tokens = generated_tokens_tensor[0].cpu().tolist()
    final_tokens = seed_tokens + generated_tokens # 원본 멜로디에 이어서 생성된 토큰을 붙임
    
    print(f"\n[진행] 생성 완료! (전체 토큰: {len(final_tokens)}개). MIDI로 변환 중...")
    out_midi = tokenizer.tokens_to_midi(final_tokens)
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_midi.write(output_path)
    
    print(f"[성공] 생성된 MIDI가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()
