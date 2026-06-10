import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
from pathlib import Path
import pretty_midi
import mido
import time
from src.tokenizer import TSDTokenizer

def play_midi_file(file_path: str, label: str = ""):
    """Play the MIDI file using mido (system MIDI output port)"""
    try:
        outputs = mido.get_output_names()
        if not outputs:
            print("[오류] 사용 가능한 MIDI 출력 장치를 찾을 수 없습니다.")
            return
            
        port_name = outputs[0]  # 보통 Windows에서는 'Microsoft GS Wavetable Synth 0'
        print(f"\n🎵 [재생 시작] {label} MIDI 파일: {file_path}")
        print(f"🔊 [출력 장치] {port_name}")
        print("정지하려면 Ctrl+C를 누르세요.\n")
        
        mid = mido.MidiFile(file_path)
        with mido.open_output(port_name) as port:
            for msg in mid.play():
                if not msg.is_meta:
                    port.send(msg)
        print(f"✅ {label} [재생 완료]")
    except KeyboardInterrupt:
        print(f"\n⏹️ {label} [재생 중지됨]")
    except Exception as e:
        print(f"\n❌ {label} 재생 중 오류가 발생했습니다: {e}")

def display_ipython_audio(original_midi: pretty_midi.PrettyMIDI, reconstructed_midi: pretty_midi.PrettyMIDI, has_original: bool):
    """Jupyter/Colab 환경에서 HTML5 오디오 플레이어 2개를 출력창에 렌더링"""
    try:
        from IPython.display import Audio, display, HTML
        import numpy as np
        
        fs = 22050  # 합성속도와 음질의 타협점
        
        print("\n🎹 오디오 플레이어 합성 중... (잠시만 기다려주세요)")
        
        # 1. 복원된 MIDI 합성
        recon_audio = reconstructed_midi.synthesize(fs=fs)
        
        # 2. 원본 MIDI 합성 (원본이 존재하는 경우)
        orig_audio = None
        if has_original and original_midi is not None:
            orig_audio = original_midi.synthesize(fs=fs)
            
        print("🔊 Jupyter/Colab 출력 창에 오디오 플레이어를 표시합니다.")
        
        if has_original and orig_audio is not None:
            # 원본 플레이어 표시
            display(HTML("<div style='margin-bottom: 15px;'><strong>▶️ Original MIDI (원본)</strong></div>"))
            display(Audio(orig_audio, rate=fs))
            
            # 복원된 플레이어 표시
            display(HTML("<div style='margin-top: 15px; margin-bottom: 15px;'><strong>▶️ Reconstructed MIDI (토큰화 후 복원)</strong></div>"))
            display(Audio(recon_audio, rate=fs))
        else:
            display(HTML("<div style='margin-bottom: 15px;'><strong>▶️ Reconstructed MIDI (토큰화 후 복원)</strong></div>"))
            display(Audio(recon_audio, rate=fs))
            
    except ImportError:
        print("\n[알림] IPython 패키지가 설치되어 있지 않아 터미널 모드로 동작합니다.")
    except Exception as e:
        print(f"\n[오류] 오디오 플레이어 생성 중 에러가 발생했습니다: {e}")

def main():
    parser = argparse.ArgumentParser(description="TSD Tokenizer Check, Reconstruction, and Comparison Script")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing MIDI or .pkl files")
    parser.add_argument("--file_path", type=str, default=None, help="Specific MIDI or .pkl file to tokenize/decode")
    parser.add_argument("--output_midi", type=str, default="outputs/reconstructed.mid", help="Path to save the reconstructed MIDI file")
    parser.add_argument("--play", action="store_true", help="Play the reconstructed MIDI file (Terminal Mido)")
    parser.add_argument("--compare", action="store_true", help="Play both original and reconstructed MIDI files sequentially (Terminal Mido)")
    args = parser.parse_args()

    tokenizer = TSDTokenizer()
    target_file = None
    is_pkl = False

    # 1. Determine the target file
    if args.file_path:
        target_file = Path(args.file_path)
        if not target_file.exists():
            print(f"[ERROR] Specified file not found: {target_file}")
            return
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"[ERROR] Specified directory not found: {data_dir}")
            return
        
        # Search for MIDI files first
        midi_files = sorted(list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi")))
        if midi_files:
            target_file = midi_files[0]
            print(f"Found {len(midi_files)} MIDI file(s) in {data_dir}. Using the first one: {target_file.name}")
        else:
            # Search for .pkl files
            pkl_files = sorted(list(data_dir.glob("**/*.pkl")))
            if pkl_files:
                target_file = pkl_files[0]
                is_pkl = True
                print(f"Found {len(pkl_files)} .pkl file(s) in {data_dir}. Using the first one: {target_file.name}")
            else:
                print(f"[ERROR] No MIDI (.mid, .midi) or preprocessed (.pkl) files found in {data_dir}")
                return
    else:
        # Default fallback to examples/theme.mid
        default_path = Path("examples/theme.mid")
        if default_path.exists():
            target_file = default_path
            print(f"No arguments provided. Falling back to default: {default_path}")
        else:
            print("[ERROR] Please provide --data_dir or --file_path.")
            return

    # Set whether the source is a pkl file
    if target_file.suffix.lower() == '.pkl':
        is_pkl = True

    # 2. Tokenize (MIDI) or Decode (pkl)
    original_midi = None
    if is_pkl:
        print(f"Loading preprocessed tokens from: {target_file}")
        with open(target_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "tokens" in data:
            tokens = data["tokens"]
        elif isinstance(data, list):
            tokens = data
        else:
            print("[ERROR] Invalid .pkl format. Expected list of tokens or dict with 'tokens' key.")
            return
    else:
        print(f"Loading MIDI file: {target_file}")
        try:
            original_midi = pretty_midi.PrettyMIDI(str(target_file))
        except Exception as e:
            print(f"[ERROR] Failed to load MIDI file: {e}")
            return
        
        print("Tokenizing MIDI...")
        tokens = tokenizer.midi_to_tokens(original_midi)

    # 3. Print Token info
    print(f"Total tokens: {len(tokens)}")
    
    # Print tokens
    print("\n--- Tokens (ID and String) ---")
    token_strs = tokenizer.decode(tokens)
    for idx, (tid, tstr) in enumerate(zip(tokens, token_strs)):
        print(f"{idx:02d}: ID={tid:<5} -> {tstr}")
        
    # Count token type frequencies
    print("\n--- Token Types Counts ---")
    types = {}
    for tstr in token_strs:
        prefix = tstr.split("_")[0] if "_" in tstr else tstr
        types[prefix] = types.get(prefix, 0) + 1
    for k, v in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<15} : {v} occurrences")

    # 4. Reconstruct MIDI from Tokens
    print(f"\nReconstructing MIDI from tokens...")
    try:
        reconstructed_midi = tokenizer.tokens_to_midi(tokens)
        
        # Ensure output directory exists
        output_path = Path(args.output_midi)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        reconstructed_midi.write(str(output_path))
        print(f"📂 복원된 MIDI 파일이 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to reconstruct or save MIDI: {e}")
        return

    # 5. IPython Audio Player Rendering (Jupyter/Colab 지원)
    # 현재 세션이 Jupyter 환경인지 체크하여 자동으로 플레이어 2개 렌더링
    is_jupyter = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            is_jupyter = True
    except:
        pass

    if is_jupyter:
        display_ipython_audio(original_midi, reconstructed_midi, has_original=(not is_pkl))
    else:
        # 터미널 환경인 경우 기존처럼 mido 재생
        if args.compare:
            if is_pkl:
                print("⚠️ [경고] .pkl 파일은 원본 MIDI 파일 정보가 없으므로 복원된 MIDI만 재생합니다.")
                play_midi_file(str(output_path), label="[복원된]")
            else:
                print("\n==================================================")
                print("🔊 원본 MIDI와 복원된 MIDI의 비교 재생을 시작합니다.")
                print("==================================================")
                play_midi_file(str(target_file), label="[원본]")
                print("\n⏸️  2초간 대기 후 복원된 MIDI를 재생합니다...")
                time.sleep(2.0)
                play_midi_file(str(output_path), label="[복원된]")
        elif args.play:
            play_midi_file(str(output_path), label="[복원된]")

if __name__ == "__main__":
    main()
